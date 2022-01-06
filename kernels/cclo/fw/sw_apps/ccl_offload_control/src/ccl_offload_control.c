/*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# *******************************************************************************/

#include <stdbool.h>
#include "ccl_offload_control.h"

#ifdef ACCL_SYNTHESIS
#include "xparameters.h"
#include "mb_interface.h"
#include "microblaze_interrupts_i.h"
#include <setjmp.h>
static jmp_buf excp_handler;
#else

sem_t mb_irq_mutex;
void microblaze_disable_interrupts(){sem_wait(&mb_irq_mutex);};
void microblaze_enable_interrupts(){sem_post(&mb_irq_mutex);};

#endif

static volatile int 		 use_tcp = 1;
static volatile int 		 dma_tag;
static volatile unsigned int timeout = 1 << 28;
static volatile unsigned int dma_tag_lookup [MAX_DMA_TAGS]; //index of the spare buffer that has been issued with that dma tag. -1 otherwise
static volatile	unsigned int dma_transaction_size 	= DMA_MAX_BTT;
static volatile unsigned int max_dma_in_flight 		= DMA_MAX_TRANSACTIONS;

static datapath_arith_config arcfg;
static communicator world;

#ifndef ACCL_SYNTHESIS
uint32_t sim_cfgmem[END_OF_EXCHMEM/4];
uint32_t *cfgmem = sim_cfgmem;
hlslib::Stream<ap_axiu<32,0,0,0>, 512> cmd_fifos[4];
hlslib::Stream<ap_axiu<32,0,0,0>, 512> sts_fifos[4];
#else
uint32_t *cfgmem = (uint32_t *)(EXCHMEM_BASEADDR);
#endif

//circular buffer operation
void cb_init(circular_buffer *cb, unsigned int capacity){
    cb->write_idx=0;
    cb->read_idx=0;
    cb->occupancy=0;
    cb->capacity= (capacity < MAX_CIRCULAR_BUFFER_SIZE) ? capacity : MAX_CIRCULAR_BUFFER_SIZE;
}

void cb_push(circular_buffer *cb, unsigned int item){
    cb->buffer[cb->write_idx] = item;
    cb->write_idx = (cb->write_idx + 1) % cb->capacity;
    cb->occupancy++;
}

unsigned int cb_pop(circular_buffer *cb){
    unsigned int item = cb->buffer[cb->read_idx];
    cb->read_idx = (cb->read_idx + 1) % cb->capacity;
    cb->occupancy--;
    return item;
}

bool cb_empty(circular_buffer *cb){
    return (cb->occupancy == 0);
}

bool cb_full(circular_buffer *cb){
    return (cb->occupancy == cb->capacity);
}

//retrieves the communicator
static inline communicator find_comm(unsigned int adr){
    communicator ret;
    ret.size 		= Xil_In32(adr);
    ret.local_rank 	= Xil_In32(adr+4);
    if(ret.size != 0 && ret.local_rank < ret.size){
        ret.ranks = (comm_rank*)(cfgmem+(adr+8)/4);
    } else {
        ret.size = 0;
        ret.local_rank = 0;
        ret.ranks = NULL;
    }
    return ret;
}

//Packetizer/Depacketizer
static inline void start_packetizer(unsigned int base_addr,unsigned int max_pktsize) {
    //get number of DATAPATH_WIDTH_BYTES transfers corresponding to max_pktsize
    unsigned int max_pkt_transfers = (max_pktsize+DATAPATH_WIDTH_BYTES-1)/DATAPATH_WIDTH_BYTES;
    Xil_Out32(base_addr+0x10, max_pkt_transfers);
    SET(base_addr, CONTROL_REPEAT_MASK | CONTROL_START_MASK);
}

static inline void start_depacketizer(unsigned int base_addr) {
    SET(base_addr, CONTROL_REPEAT_MASK | CONTROL_START_MASK );
}

//connection management

//establish connection with every other rank in the communicator
int openCon()
{
    unsigned int session 	= 0;
    unsigned int dst_ip 	= 0;
    unsigned int dst_port 	= 0;
    unsigned int success	= 0;
    int ret = 0;

    unsigned int cur_rank_ip	= 0;
    unsigned int cur_rank_port 	= 0;

    unsigned int size 		= world.size;
    unsigned int local_rank = world.local_rank;

    //open connection to all the ranks except for the local rank
    for (int i = 0; i < size; i++)
    {
        if (i != local_rank)
        {
            cur_rank_ip 	= world.ranks[i].ip;
            cur_rank_port 	= world.ranks[i].port;
            //send open connection request to the packetizer
            putd(CMD_NET_CON, cur_rank_ip);
            putd(CMD_NET_CON, cur_rank_port);
        }
    }
    ret = NO_ERROR;
    //wait until the connections status are all returned
    for (int i = 0; i < size -1 ; i++)
    {	
        //ask: tngetd or the stack will return a non-success? 
        session 	= getd(STS_NET_CON);
        dst_ip 		= getd(STS_NET_CON);
        dst_port 	= getd(STS_NET_CON);
        success 	= getd(STS_NET_CON);

        if (success)
        {
            //store the session ID into corresponding rank
            for (int j = 0; j < size ; ++j)
            {
                cur_rank_ip 	= world.ranks[j].ip;
                cur_rank_port 	= world.ranks[j].port;
                if ((dst_ip == cur_rank_ip) && (dst_port == cur_rank_port))
                {
                    world.ranks[j].session = session;
                    break;
                }
            }
        }
        else
        {
            ret = OPEN_CON_NOT_SUCCEEDED;
        }
    }
    return ret;
}
//open local port for listening
int openPort()
{
    int success = 0;

    //open port with only the local rank
    putd(CMD_NET_PORT, world.ranks[world.local_rank].port);
    success = getd(STS_NET_PORT);

    if (success)
        return NO_ERROR;
    else
        return OPEN_PORT_NOT_SUCCEEDED;

}

static inline unsigned int segment(unsigned int number_of_bytes,unsigned int segment_size){
    return  (number_of_bytes + segment_size - 1) / segment_size;
} 
/*
//configure datapath before calling this method
//starts one or more datamovers (AXI-MM DMs or Ethernet Packetizer)
//updates the config when done
static inline int start_move(
                    dm_config * cfg,
                    unsigned int dma_tag,
                    unsigned int dst_rank,
                    unsigned int msg_tag) {
    //BE CAREFUL!: if(count > DMA_MAX_BTT) return DMA_TRANSACTION_SIZE;
    //NOTE: count is the number of uncompressed elements to be transferred, which we convert to bytes

    //start DMAs 
    //OP0 from stream or DMA
    if( cfg->which_dm & USE_OP0_DM){
        if(cfg->stream & OP0_STREAM){
            start_krnl_message(cfg->op0_len);
        } else{
            dma_tag = (dma_tag + 1) & 0xf;
            dma_cmd(CMD_DMA0_RX, cfg->op0_len, cfg->op0_addr.ptr, dma_tag);
        }
    }
    //OP1 always from DMA
    if( cfg->which_dm & USE_OP1_DM){
        dma_tag = (dma_tag + 1) & 0xf;
        dma_cmd(CMD_DMA1_RX, cfg->op1_len, cfg->op1_addr.ptr, dma_tag);
    }
    //RES to stream, DMA, or packetizer
    if( cfg->which_dm & USE_RES_DM){
        if(cfg->remote & RES_REMOTE){
            start_packetizer_message(dst_rank, cfg->res_len, msg_tag, cfg->stream & RES_STREAM);
        } else if(!(cfg->stream & RES_STREAM)){
            dma_tag = (dma_tag + 1) & 0xf;
            dma_cmd(CMD_DMA1_TX, cfg->res_len, cfg->res_addr.ptr, dma_tag);
        }
        //if RES is STREAM, then we don't need to do anything,
        //data directly flows through the switch into the external kernel 
        //TODO: implement TDEST inserter to be able to target specific external kernel
    }
    return dma_tag;
}

static inline void ack_move(
    dm_config * cfg,
    unsigned int dst_rank) {
    unsigned int invalid, count, status;

    //wait for DMAs to complete where appropriate
    count = 0;
    do {
        if(timeout != 0 && count >= timeout )
            longjmp(excp_handler, DMA_TIMEOUT_ERROR);
        count ++;
        invalid = 0;
        if( cfg->which_dm & USE_OP0_DM){
            if(!(cfg->stream & OP0_STREAM)){
                invalid += tngetd(STS_DMA0_RX);
            }
        }
        if( cfg->which_dm & USE_OP1_DM){
            invalid += tngetd(STS_DMA1_RX);
        }
        if( cfg->which_dm & (USE_RES_DM)){
            if(!(cfg->stream & RES_STREAM) && !(cfg->remote & RES_REMOTE)){
                invalid += tngetd(STS_DMA1_TX);
            }
        }
    } while(invalid);

    if( cfg->which_dm & USE_OP0_DM){
        if(cfg->stream & OP0_STREAM){
            ack_krnl_message(cfg->op0_len);
        } else {
            status = getd(STS_DMA0_RX);
            dma_tag = (dma_tag + 1) & 0xf;
            check_DMA_status(status, cfg->op0_len, dma_tag, 0);
        }
    }
    if( cfg->which_dm & USE_OP1_DM){
        status = getd(STS_DMA1_RX);
        dma_tag = (dma_tag + 1) & 0xf;
        check_DMA_status(status, cfg->op1_len, dma_tag, 0);
    }
    if( cfg->which_dm & USE_RES_DM){
        if(cfg->remote & RES_REMOTE){
            ack_packetizer_message(dst_rank, !(cfg->stream & RES_STREAM));
        } else if(!(cfg->stream & RES_STREAM)){
            status = getd(STS_DMA1_TX);
            dma_tag = (dma_tag + 1) & 0xf;
            check_DMA_status(status, cfg->res_len, dma_tag, 0);
        }      
    }
}
*/
//configure datapath before calling this method
//instructs the data plane to move data
//use MOVE_IMMEDIATE
inline int move(
    uint32_t op0_opcode,
    uint32_t op1_opcode,
    uint32_t res_opcode,
    uint32_t compression_flags,
    uint32_t remote_flags,
    uint32_t func_id,
    uint32_t count,
    uint32_t comm_offset,
    uint32_t arcfg_offset,
    uint64_t op0_addr,
    uint64_t op1_addr,
    uint64_t res_addr,
    uint32_t op0_stride,
    uint32_t op1_stride,
    uint32_t res_stride,
    uint32_t rx_src_rank,
    uint32_t rx_tag,
    uint32_t tx_dst_rank,
    uint32_t tx_tag
) {
    uint32_t opcode = 0;
    opcode |= op0_opcode;
    opcode |= op1_opcode << 3;
    opcode |= res_opcode << 6;
    opcode |= remote_flags << 9;
    bool res_is_remote = (remote_flags == RES_REMOTE);
    opcode |= compression_flags << 10;
    opcode |= func_id << 13;
    putd(CMD_DMA_MOVE, opcode);
    putd(CMD_DMA_MOVE, count);

    //arith config offset, as an offset in a uint32_t array
    putd(CMD_DMA_MOVE, arcfg_offset/4);

    //get addr for op0, or equivalents
    if(op0_opcode == MOVE_IMMEDIATE){
        putd(CMD_DMA_MOVE, (uint32_t)op0_addr);
        putd(CMD_DMA_MOVE, (uint32_t)(op0_addr>>32));
    } else if(op0_opcode == MOVE_STRIDE){
        putd(CMD_DMA_MOVE, op0_stride);
    }
    //get addr for op1, or equivalents
    if(op1_opcode == MOVE_IMMEDIATE){
        putd(CMD_DMA_MOVE, (uint32_t)op1_addr);
        putd(CMD_DMA_MOVE, (uint32_t)(op1_addr>>32));
    } else if(op1_opcode == MOVE_ON_RECV){
        putd(CMD_DMA_MOVE, rx_src_rank);
        putd(CMD_DMA_MOVE, rx_tag);
    } else if(op1_opcode == MOVE_STRIDE){
        putd(CMD_DMA_MOVE, op1_stride);
    }
    //get addr for res, or equivalents
    if(res_opcode == MOVE_IMMEDIATE){
        putd(CMD_DMA_MOVE, (uint32_t)res_addr);
        putd(CMD_DMA_MOVE, (uint32_t)(res_addr>>32));
    } else if(res_opcode == MOVE_STRIDE){
        putd(CMD_DMA_MOVE, res_stride);
    }
    //get send related stuff, if result is remote or stream
    if(res_is_remote || res_opcode == MOVE_STREAM){
        putd(CMD_DMA_MOVE, tx_tag);
    }
    if(res_is_remote){
        putd(CMD_DMA_MOVE, comm_offset/4);
        putd(CMD_DMA_MOVE, tx_dst_rank);
    }
    return getd(STS_DMA_MOVE);
}
/*
//segment a logical move into multiple MOVE_IMMEDIATEs
int move_segmented(
    dm_config * cfg,
    unsigned dst_rank,
    unsigned dst_tag
) {
    unsigned int dma_tag_tmp, i;
    dma_tag_tmp = dma_tag;

    //set up datamover configs for emitting and retiring transfers
    //we need two, because we emit and retire in separate processes which are offset
    dm_config emit_dm_config = *cfg;
    dm_config ack_dm_config = *cfg;

    //1. issue at most max_dma_in_flight of dma_transaction_size
    for (i = 0; emit_dm_config.elems_remaining > 0 && i < max_dma_in_flight ; i++){
        //start DMAs
        dma_tag_tmp = start_move(&emit_dm_config, dma_tag_tmp, dst_rank, dst_tag);
        dm_config_update(&emit_dm_config);
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(emit_dm_config.elems_remaining > 0){
        //wait for DMAs to finish
        ack_move(&ack_dm_config, dst_rank);
        //start DMAs
        dma_tag_tmp = start_move(&emit_dm_config, dma_tag_tmp, dst_rank, dst_tag);
        //update configs
        dm_config_update(&emit_dm_config);
        dm_config_update(&ack_dm_config);
    }
    //3. finish ack the remaining
    while(ack_dm_config.elems_remaining > 0){
        //wait for DMAs to finish
        ack_move(&ack_dm_config, dst_rank);
        dm_config_update(&ack_dm_config);
    }
    return NO_ERROR;
}
*/

//performs a copy using DMA0. DMA0 rx reads while DMA1 tx overwrites
//use MOVE_IMMEDIATE
static inline int copy(	unsigned int count,
                        uint64_t src_addr,
                        uint64_t dst_addr,
                        unsigned int arcfg_offset,
                        unsigned int compression,
                        unsigned int stream) {
    return move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        MOVE_NONE, 
        (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        compression, RES_LOCAL, 0,
        count, 0, arcfg_offset, src_addr, 0, dst_addr, 0, 0, 0,
        0, 0, 0, 0
    );
}

//performs an accumulate using DMA1 and DMA0. DMA0 rx reads op1 DMA1 rx reads op2 while DMA1 tx back to dst buffer
//use MOVE_IMMEDIATE
int combine(unsigned int count,
            unsigned int function,
            uint64_t op0_addr,
            uint64_t op1_addr,
            uint64_t res_addr,
            unsigned int arcfg_offset,
            unsigned int compression,
            unsigned int stream) {
    return move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        MOVE_IMMEDIATE,
        (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        compression, RES_LOCAL, 0,
        count, 0, arcfg_offset, op0_addr, op1_addr, res_addr, 0, 0, 0,
        0, 0, 0, 0
    );
}

//transmits a buffer to a rank of the world communicator
//use MOVE_IMMEDIATE
int send(
    unsigned int dst_rank,
    unsigned int count,
    uint64_t src_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset,
    unsigned int dst_tag,
    unsigned int compression,
    unsigned int stream
){
    return move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        MOVE_NONE,
        MOVE_IMMEDIATE, 
        compression, RES_REMOTE, 0,
        count, comm_offset, arcfg_offset, src_addr, 0, 0, 0, 0, 0,
        0, 0, dst_rank, dst_tag
    );
}

//waits for a messages to come and move their contents in a buffer
//use MOVE_ON_RECV
int recv(	unsigned int src_rank,	
            unsigned int count,
            uint64_t dst_addr,
            unsigned int arcfg_offset,
            unsigned int src_tag,
            unsigned int compression){
    return move(
        MOVE_ON_RECV, 
        MOVE_NONE,
        MOVE_IMMEDIATE, 
        compression, RES_LOCAL, 0,
        count, 0, arcfg_offset, 0, 0, dst_addr, 0, 0, 0,
        src_rank, src_tag, 0, 0
    );
}

/*
//performs a reduction using DMA0 and DMA1 and then sends the result 
//through tx_subsystem to dst_rank
static inline int fused_reduce_send(
                    unsigned int dst_rank,
                    unsigned int count,
                    uint64_t op0_addr,
                    uint64_t op1_addr,
                    unsigned int dst_tag,
                    unsigned int compression) {
    configure_datapath(DATAPATH_OFFCHIP_REDUCTION, 0, compression, 0);
    dm_config cfg = dm_config_init(count, op0_addr, op1_addr, 0, USE_OP0_DM | USE_OP1_DM | USE_RES_DM, RES_REMOTE, compression, NO_STREAM);
    return move_segmented(&cfg, dst_rank, dst_tag);
}

//iterates over rx buffers until match is found or a timeout expires
//matches count, src and tag if tag is not ANY
//returns the index of the spare_buffer or -1 if not found
int seek_rx_buffer(
                        unsigned int src_rank,
                        unsigned int count,
                        unsigned int src_tag
                    ){
    unsigned int seq_num; //src_port TODO: use this variable to choose depending on session id or port
    int i;
    seq_num = world.ranks[src_rank].inbound_seq + 1;

    //TODO: use a list to store recent message received to avoid scanning in the entire spare_buffer_struct.
    //parse rx buffers until match is found
    //matches count, src
    //matches tag or tag is ANY
    //return buffer index 
    unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
    rx_buffer *rx_buf_list = (rx_buffer*)(cfgmem+RX_BUFFER_COUNT_OFFSET/4+1);
    for(i=0; i<nbufs; i++){	
        if(rx_buf_list[i].status == STATUS_RESERVED)
        {
            if((rx_buf_list[i].rx_src == src_rank) && (rx_buf_list[i].rx_len == count))
            {
                if(((rx_buf_list[i].rx_tag == src_tag) || (src_tag == TAG_ANY)) && (rx_buf_list[i].sequence_number == seq_num) )
                {
                    //only now advance sequence number
                    world.ranks[src_rank].inbound_seq++;
                    return i;
                }
            }
        }
    }
    return -1;
}

//iterates over rx buffers until match is found or a timeout expires
//matches count, src and tag if tag is not ANY
//returns the index of the spare_buffer
//timeout is jumps to exception handler
//useful as infrastructure for [I]MProbe/[I]MRecv
int wait_on_rx(	unsigned int src_rank,
                    unsigned int count,
                    unsigned int src_tag){
    int idx, i;
    for(i = 0; timeout == 0 || i < timeout; i++){
        idx = seek_rx_buffer(src_rank, count, src_tag);
        if(idx >= 0) return idx;
    }
    longjmp(excp_handler, RECEIVE_TIMEOUT_ERROR);
}
*/

/*
//1) receives from a rank
//2) sums with a a buffer 
//3) the result is saved in (possibly another) local buffer
//use MOVE_ON_RECV
int fused_recv_reduce(
        unsigned int src_rank,
        unsigned int count,
        unsigned int func,
        uint64_t op0_addr,
        uint64_t dst_addr,
        unsigned int src_tag,
        unsigned int compression)
    {
    unsigned int buf_idx;
    unsigned int dma_tag_tmp = dma_tag, i;
    rx_buffer *rx_buf_list = (rx_buffer*)(cfgmem+RX_BUFFER_COUNT_OFFSET/4+1);

    circular_buffer spare_buffer_queue;
    cb_init(&spare_buffer_queue, max_dma_in_flight);

    //figure out compression; essentially what we do here is a reduction where op1 is from the network
    //therefore OP1_COMPRESSED should be equal to ETH_COMPRESSED;
    compression = (compression & ETH_COMPRESSED) ? (compression | OP1_COMPRESSED) : compression;

    configure_datapath(DATAPATH_DMA_REDUCTION, func, compression, 0);

    //set up datamover configs for emitting and retiring transfers
    //we need two, because we emit and retire in separate processes which are offset
    dm_config emit_dm_config = dm_config_init(count, op0_addr, 0, dst_addr, USE_OP0_DM | USE_OP1_DM | USE_RES_DM, NO_REMOTE, compression, NO_STREAM);
    dm_config ack_dm_config = dm_config_init(count, 0, 0, 0, USE_OP0_DM | USE_OP1_DM | USE_RES_DM, NO_REMOTE, compression, NO_STREAM);

    //1. issue at most max_dma_in_flight of dma_transaction_size
    for (i = 0; emit_dm_config.elems_remaining > 0 && i < max_dma_in_flight ; i++){
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, emit_dm_config.op1_len, src_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        emit_dm_config.op1_addr.ptr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp  = start_move(&emit_dm_config, dma_tag_tmp, 0, 0);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(emit_dm_config.elems_remaining > 0){
        //wait for DMAs to finish
        ack_move(&ack_dm_config, 0);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        //enqueue other DMA movement
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, emit_dm_config.op1_len, src_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        emit_dm_config.op1_addr.ptr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp  = start_move(&emit_dm_config, dma_tag_tmp, 0, 0);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
    }
    //3. finish ack the remaining
    while(ack_dm_config.elems_remaining > 0){
        //wait for DMAs to finish
        ack_move(&ack_dm_config, 0);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
    }

    return NO_ERROR;
}

//1) receives from a rank 
//2) sums with a a buffer
//3) result is sent to another rank
//use MOVE_ON_RECV
int fused_recv_reduce_send(
        unsigned int src_rank,
        unsigned int dst_rank,
        unsigned int count,
        unsigned int func,
        uint64_t op0_addr,
        unsigned int mpi_tag,
        unsigned int compression)
    {
    unsigned int buf_idx;
    unsigned int dma_tag_tmp = dma_tag, i;
    rx_buffer *rx_buf_list = (rx_buffer*)(cfgmem+RX_BUFFER_COUNT_OFFSET/4+1);

    circular_buffer spare_buffer_queue;
    cb_init(&spare_buffer_queue, max_dma_in_flight);

    //figure out compression; essentially what we do here is a reduction where op1 is from the network
    //therefore OP1_COMPRESSED should be equal to ETH_COMPRESSED;
    compression = (compression & ETH_COMPRESSED) ? (compression | OP1_COMPRESSED) : compression;

    configure_datapath(DATAPATH_OFFCHIP_REDUCTION, func, compression, 0);

    //set up datamover configs for emitting and retiring transfers
    //we need two, because we emit and retire in separate processes which are offset
    dm_config emit_dm_config = dm_config_init(count, op0_addr, 0, 0, USE_OP0_DM | USE_OP1_DM, NO_REMOTE, compression, NO_STREAM);
    dm_config ack_dm_config = dm_config_init(count, 0, 0, 0, USE_OP0_DM | USE_OP1_DM, NO_REMOTE, compression, NO_STREAM);

    //1. issue at most max_dma_in_flight of dma_transaction_size
    for (i = 0; emit_dm_config.elems_remaining > 0 && i < max_dma_in_flight ; i++){
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, emit_dm_config.op1_len, mpi_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        emit_dm_config.op1_addr.ptr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp = start_move(&emit_dm_config, dma_tag_tmp, dst_rank, mpi_tag);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(emit_dm_config.elems_remaining > 0){
        //wait for DMAs to finish
        ack_move(&ack_dm_config, dst_rank);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        //enqueue other DMA movement
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, emit_dm_config.op1_len, mpi_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        emit_dm_config.op1_addr.ptr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp = start_move(&emit_dm_config, dma_tag_tmp, dst_rank, mpi_tag);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
    }
    //3. finish ack the remaining
    while(ack_dm_config.elems_remaining > 0){
        //wait for DMAs to finish
        ack_move(&ack_dm_config, dst_rank);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
    }

    return NO_ERROR;
}

//receives from a rank and forwards to another rank 
//use MOVE_ON_RECV
int relay(
        unsigned int src_rank,
        unsigned int dst_rank,
        unsigned int count, 
        unsigned int mpi_tag,
        unsigned int compression){
    unsigned int buf_idx;
    unsigned int dma_tag_tmp = dma_tag, i;
    rx_buffer *rx_buf_list = (rx_buffer*)(cfgmem+RX_BUFFER_COUNT_OFFSET/4+1);

    circular_buffer spare_buffer_queue;
    cb_init(&spare_buffer_queue, max_dma_in_flight);

    //compression adjustment: if ETH_COMPRESSED, then we will use OP0_COMPRESSED and ETH_COMPRESSED
    compression = (compression & ETH_COMPRESSED) ? (OP0_COMPRESSED | ETH_COMPRESSED) : NO_COMPRESSION;
    //configure datapath
    configure_datapath(DATAPATH_OFFCHIP_TX, 0, compression, 0);

    //set up datamover configs for emitting and retiring transfers
    //we need two, because we emit and retire in separate processes which are offset
    dm_config emit_dm_config = dm_config_init(count, 0, 0, 0, USE_OP0_DM | USE_RES_DM, RES_REMOTE, compression, NO_STREAM);
    dm_config ack_dm_config = dm_config_init(count, 0, 0, 0, USE_OP0_DM | USE_RES_DM, RES_REMOTE, compression, NO_STREAM);

    //1. issue at most max_dma_in_flight of dma_transaction_size
    for (i = 0; emit_dm_config.elems_remaining > 0 && i < max_dma_in_flight ; i++){
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, emit_dm_config.op0_len, mpi_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        emit_dm_config.op0_addr.ptr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp = start_move(&emit_dm_config, dma_tag_tmp, dst_rank, mpi_tag);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(emit_dm_config.elems_remaining > 0){
        //wait for DMAs to finish
        ack_move(&ack_dm_config, dst_rank);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        //enqueue other DMA movement
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, emit_dm_config.op0_len, mpi_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        emit_dm_config.op0_addr.ptr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp = start_move(&emit_dm_config, dma_tag_tmp, dst_rank, mpi_tag);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
    }
    //3. finish ack the remaining
    while(ack_dm_config.elems_remaining > 0){
        //wait for DMAs to finish
        ack_move(&ack_dm_config, dst_rank);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
    }

    return NO_ERROR;
}

//COLLECTIVES

//root read multiple times the same segment and send it to each rank before moving to the next part of the buffer to be transmitted.
//use MOVE_IMMEDIATE and MOVE_INCREMENTING and MOVE_REPEATING in sequence 
int broadcast(  unsigned int count,
                unsigned int src_rank,
                uint64_t buf_addr,
                unsigned int compression){
    int i, dma_tag_tmp=dma_tag;
    //determine if we're sending or receiving
    if(src_rank == world.local_rank){
        
        //on the root we only care about ETH_COMPRESSED and OP0_COMPRESSED
        //so mask out RES_COMPRESSED
        compression &= ~(RES_COMPRESSED);

        //configure datapath
        configure_datapath(DATAPATH_OFFCHIP_TX, 0, compression, 0);

        //send to all members of the communicator 1 segment of the buffer at the time
        dm_config cfg = dm_config_init(count, buf_addr, 0, 0, USE_OP0_DM | USE_RES_DM, RES_REMOTE, compression, NO_STREAM);

        int emit_dst_rank, ack_dst_rank;
        while(cfg.elems_remaining > 0){
            //1. issue at most max_dma_in_flight transfers to consecutive ranks
            for (i = 0, emit_dst_rank=0; emit_dst_rank < world.size && i < max_dma_in_flight ; i++, emit_dst_rank++){
                if(emit_dst_rank != src_rank){
                    //start DMAs
                    dma_tag_tmp = start_move(&cfg, dma_tag_tmp, emit_dst_rank, TAG_ANY);
                }
            }
            //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
            for(ack_dst_rank = 0; emit_dst_rank < world.size; ack_dst_rank++, emit_dst_rank++){
                if(ack_dst_rank != src_rank){
                    //wait for DMAs to finish
                    ack_move(&cfg, ack_dst_rank);
                }
                //enqueue other DMA movement
                if(emit_dst_rank != src_rank){
                    //start DMAs
                    dma_tag_tmp = start_move(&cfg, dma_tag_tmp, emit_dst_rank, TAG_ANY);
                }
            }
            //3. finish ack the remaining
            for(; ack_dst_rank < world.size; ack_dst_rank++){
                if(ack_dst_rank != src_rank){
                    //wait for DMAs to finish
                    ack_move(&cfg, ack_dst_rank);
                }
            }
            //advance config
            //TODO: this can be interleaved with command issuing to avoid pausing
            dm_config_update(&cfg);
        }
        return NO_ERROR;
    }else{
        //on non-root odes we only care about ETH_COMPRESSED and RES_COMPRESSED
        //so mask out OP0_COMPRESSED
        //NOTE: recv currently ignores OP0_COMPRESSED anyway, but this is safer and better for documentation
        compression &= ~(OP0_COMPRESSED);
        return recv(src_rank, count, buf_addr, TAG_ANY, compression);
    }
    
}

//scatter segment at a time. root sends each rank a segment in a round robin fashion 
//use MOVE_IMMEDIATE and MOVE_INCREMENTING and MOVE_STRIDE in sequence 
int scatter(unsigned int count,
            unsigned int src_rank,
            uint64_t src_buf_addr,
            uint64_t dst_buf_addr,
            unsigned int compression){

    unsigned int curr_count, i;
    unsigned int dma_tag_tmp = dma_tag;

    //determine if we're sending or receiving
    if(src_rank == world.local_rank){
        
        //on the root we only care about ETH_COMPRESSED and OP0_COMPRESSED
        //so mask out RES_COMPRESSED
        compression &= ~(RES_COMPRESSED);

        //configure datapath
        configure_datapath(DATAPATH_OFFCHIP_TX, 0, compression, 0);

        //send to all members of the communicator 1 segment of the buffer at the time
        dm_config cfg = dm_config_init(count, src_buf_addr, 0, 0, USE_OP0_DM | USE_RES_DM, RES_REMOTE, compression, NO_STREAM);

        int emit_dst_rank, ack_dst_rank;
        while(cfg.elems_remaining > 0){
            dm_config stride_emit_cfg = cfg;
            dm_config stride_ack_cfg = cfg;
            //1. issue at most max_dma_in_flight transfers to consecutive ranks
            for (i = 0, emit_dst_rank=0; emit_dst_rank < world.size && i < max_dma_in_flight ; i++, emit_dst_rank++){
                if(emit_dst_rank != src_rank){
                    //start DMAs
                    dma_tag_tmp = start_move(&stride_emit_cfg, dma_tag_tmp, emit_dst_rank, TAG_ANY);
                }
                dm_config_stride(&stride_emit_cfg, count);
            }
            //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
            for(ack_dst_rank = 0; emit_dst_rank < world.size; ack_dst_rank++, emit_dst_rank++){
                if(ack_dst_rank != src_rank){
                    //wait for DMAs to finish
                    ack_move(&stride_ack_cfg, ack_dst_rank);
                }
                dm_config_stride(&stride_ack_cfg, count);
                //enqueue other DMA movement
                if(emit_dst_rank != src_rank){
                    //start DMAs
                    dma_tag_tmp = start_move(&stride_emit_cfg, dma_tag_tmp, emit_dst_rank, TAG_ANY);
                }
                dm_config_stride(&stride_emit_cfg, count);
            }
            //3. finish ack the remaining
            for(; ack_dst_rank < world.size; ack_dst_rank++){
                if(ack_dst_rank != src_rank){
                    //wait for DMAs to finish
                    ack_move(&stride_ack_cfg, ack_dst_rank);
                }
                dm_config_stride(&stride_ack_cfg, count);
            }
            //advance config
            //TODO: this can be interleaved with command issuing to avoid pausing
            dm_config_update(&cfg);
        }
        //do copy to self last (it requires reconfiguring the datapath)
        uint64_t copy_addr = phys_addr_offset(src_buf_addr, count*src_rank, (compression & OP0_COMPRESSED) ? true : false);
        copy(count, copy_addr, dst_buf_addr, compression, NO_STREAM);
        //done
        return NO_ERROR;
    }else{
        //on non-root odes we only care about ETH_COMPRESSED and RES_COMPRESSED
        //so mask out OP0_COMPRESSED
        //NOTE: recv currently ignores OP0_COMPRESSED anyway, but this is safer and better for documentation
        compression &= ~(OP0_COMPRESSED);
        return recv(src_rank, count, dst_buf_addr, TAG_ANY, compression);
    }

}

//naive gather: non root relay data to the root. root copy segments in dst buffer as they come.
//on root, SEND_ON_RECV
//elsewhere MOVE_STRIDE then SEND_ON_RECV for relay
int gather( unsigned int count,
            unsigned int root_rank,
            uint64_t src_buf_addr,
            uint64_t dst_buf_addr,
            unsigned int compression){
    uint64_t tmp_buf_addr;
    unsigned int i,curr_pos, next_in_ring, prev_in_ring, number_of_shift;
    int ret = NO_ERROR;

    next_in_ring = (world.local_rank + 1			  ) % world.size	;
    prev_in_ring = (world.local_rank + world.size - 1 ) % world.size	;
    
    if(root_rank == world.local_rank){ //root ranks mainly receives

        //we need to compute correct compression schemes from the input compression flags
        //copy to self: keep all flags except ETH_COMPRESSED, which should be reset for safety
        //recv: keep RES_COMPRESSED and ETH_COMPRESSED, reset OP0_COMPRESSED

        //receive from all members of the communicator
        for(i=0, curr_pos = prev_in_ring; i<world.size && ret == NO_ERROR; i++, curr_pos = (curr_pos + world.size - 1 ) % world.size){
            //TODO: optimize receives; what we want is to move any data which arrives into the correct segment of the buffer
            //currently this will go sequentially through the segments, which is slow and also requires more spare RX buffers
            tmp_buf_addr = phys_addr_offset(dst_buf_addr, count*curr_pos, (compression & RES_COMPRESSED) ? true : false);
            if(curr_pos==world.local_rank)
            { // root copies
                ret	= copy(count, src_buf_addr, tmp_buf_addr, compression & ~(ETH_COMPRESSED), NO_STREAM);
            }else{
                ret = recv(prev_in_ring, count, tmp_buf_addr, TAG_ANY, compression & ~(OP0_COMPRESSED));
            }
        }
    }else{
        //non root ranks sends their data + relay others data to the next rank in sequence
        // as a daisy chain

        //we need to compute correct compression schemes from the input compression flags
        //send: keep all flags except RES_COMPRESSED, which should be reset for safety
        //relay: keep ETH_COMPRESSED, reset everything else

        number_of_shift = ((world.size+world.local_rank-root_rank)%world.size) - 1 ; //distance to the root
        ret += send(next_in_ring, count, src_buf_addr, TAG_ANY, compression & ~(RES_COMPRESSED), NO_STREAM);
        for (int i = 0; i < number_of_shift; i++)
        {	
            //relay the others 
            relay(prev_in_ring, next_in_ring, count, TAG_ANY, compression & ETH_COMPRESSED);
        }	
    }
    return ret;
}

//naive fused: 1) receive a segment 2) move in the dest buffer 3) relay to next rank 
int allgather(  unsigned int count,
                uint64_t src_buf_addr,
                uint64_t dst_buf_addr,
                unsigned int compression){
    uint64_t tmp_buf_addr;
    unsigned int i,curr_pos, next_in_ring, prev_in_ring;
    int ret = NO_ERROR;

    //compression is tricky for the relay: we've already received into the destination buffer 
    //with associated flag RES_COMPRESSED; this buffer becomes the source for a send
    //so if RES_COMPRESSED is set, OP0_COMPRESSED must be set for the send, and RES_COMPRESSED reset
    unsigned int relay_compression = (compression & RES_COMPRESSED) ? (compression | OP0_COMPRESSED) : compression;
    relay_compression &= ~(RES_COMPRESSED);

    next_in_ring = (world.local_rank+1			   ) % world.size	;
    prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;

    //send our data to next in ring
    ret += send(next_in_ring, count, src_buf_addr, TAG_ANY, compression & ~(RES_COMPRESSED), NO_STREAM);
    //receive from all members of the communicator
    for(i=0, curr_pos = world.local_rank; i<world.size && ret == NO_ERROR; i++, curr_pos = (curr_pos + world.size - 1 ) % world.size){
        tmp_buf_addr = phys_addr_offset(dst_buf_addr, count*curr_pos, (compression & RES_COMPRESSED) ? true : false);
        if(curr_pos==world.local_rank){
            ret	= copy(count, src_buf_addr, tmp_buf_addr, compression & ~(ETH_COMPRESSED), NO_STREAM);
        }else{
            ret = recv(prev_in_ring, count, tmp_buf_addr, TAG_ANY, compression & ~(OP0_COMPRESSED));
            //TODO: use the same stream to move data both to dst_buf_addr and tx_subsystem 
            //todo: use DMA1 RX to forward to next and DMA0/2 RX and DMA1 TX to copy ~ at the same time! 
            if(i+1 < world.size){ //if not the last data needed relay to the next in the sequence
                ret = send(next_in_ring, count, tmp_buf_addr, TAG_ANY, relay_compression, NO_STREAM);
            }
        }		
    }

    return ret;
}

//every rank receives a buffer it reduces its own buffer and forwards to next rank in the ring
int reduce( unsigned int count,
            unsigned int func,
            unsigned int root_rank,
            uint64_t src_addr,
            uint64_t dst_addr,
            unsigned int compression){
    int ret;

    unsigned int next_in_ring = (world.local_rank+1			   ) % world.size	;
    unsigned int prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
    //determine if we're sending or receiving
    if( prev_in_ring == root_rank){ 
        //non root ranks immediately after the root sends
        ret = send(next_in_ring, count, src_addr, TAG_ANY, compression & ~(RES_COMPRESSED), NO_STREAM);
    }else if (world.local_rank != root_rank){
        //non root ranks sends their data + data received from previous rank to the next rank in sequence as a daisy chain
        ret = fused_recv_reduce_send(prev_in_ring, next_in_ring, count, func, src_addr,  TAG_ANY, compression & ~(RES_COMPRESSED));
    }else{	
        //root only receive from previous node in the ring, add its local buffer and save in destination buffer 
        ret = fused_recv_reduce(prev_in_ring, count, func, src_addr, dst_addr, TAG_ANY, compression);
    }
    return ret;
}

//scatter_reduce: (a,b,c), (1,2,3), (X,Y,Z) -> (a+1+X,,) (,b+2+Y,) (,,c+3+Z)
//count == size of chunks
int scatter_reduce( unsigned int count,
                    unsigned int func,
                    uint64_t src_addr,
                    uint64_t dst_addr,
                    unsigned int compression){
    unsigned int ret,i;
    uint64_t curr_recv_addr, curr_send_addr;

    //convert the full element count into a normal and tail count
    unsigned int curr_count, count_tail = (count % world.size == 0) ? (count/world.size) : (count%world.size);
    count = count / world.size;

    unsigned int next_in_ring 	 = (world.local_rank + 1			) % world.size	;
    unsigned int prev_in_ring 	 = (world.local_rank + world.size-1 ) % world.size	;
    unsigned int curr_send_chunk = prev_in_ring;
    unsigned int curr_recv_chunk = (prev_in_ring + world.size-1 ) % world.size	;

    //figure out compression
    //the fused_recv_reduce_send reduces op0 (from src_addr) with op1 (from the network) and sends the result to the network
    //so keep OP0_COMPRESSED and ETH_COMPRESSED from the top level;
    //the fused_recv_reduce reduces from src_addr with op1 from the network and puts in dst_addr

    //1. each rank send its own chunk to the next rank in the ring
    curr_send_addr = phys_addr_offset(src_addr, count*curr_send_chunk, (compression & OP0_COMPRESSED) ? true : false);
    curr_count = (curr_send_chunk == (world.local_rank-1)) ? count_tail : count;   
    ret = send(next_in_ring, curr_count, curr_send_addr, TAG_ANY, compression & (ETH_COMPRESSED | OP0_COMPRESSED), NO_STREAM);
    if(ret != NO_ERROR) return ret;
    //2. for n-1 times sum the chunk that is coming from previous rank with your chunk at the same location. then forward the result to
    // the next rank in the ring
    for (i = 0; i < world.size-2; ++i, curr_recv_chunk=(curr_recv_chunk + world.size - 1) % world.size)
    {
        curr_recv_addr = phys_addr_offset(src_addr, count*curr_recv_chunk, (compression & OP0_COMPRESSED) ? true : false);
        //2. receive part of data from previous in rank and accumulate to the part you have at the same address
        //and forward to the next rank in the ring
        //TODO: figure out compression
        curr_count = (curr_recv_chunk == (world.local_rank-1)) ? count_tail : count;
        ret = fused_recv_reduce_send(prev_in_ring, next_in_ring, curr_count, func, curr_recv_addr, TAG_ANY, compression & (ETH_COMPRESSED | OP0_COMPRESSED)); 
        if(ret != NO_ERROR) return ret;
    }
    //at last iteration (n-1) you can sum and save the result at the same location (which is the right place to be)
    if (world.size > 1){
        curr_recv_addr = phys_addr_offset(src_addr, count*curr_recv_chunk, (compression & OP0_COMPRESSED) ? true : false);
        curr_send_addr = phys_addr_offset(dst_addr, count*curr_recv_chunk, (compression & RES_COMPRESSED) ? true : false);
        curr_count = (curr_recv_chunk == (world.local_rank-1)) ? count_tail : count;
        ret = fused_recv_reduce(prev_in_ring, curr_count, func, curr_recv_addr, curr_send_addr, TAG_ANY, compression); 
        if(ret != NO_ERROR) return ret;
    }
    return NO_ERROR;
}

//2 stage allreduce: distribute sums across the ranks. smaller bandwidth between ranks and use multiple arith at the same time. scatter_reduce+all_gather
int allreduce(  unsigned int count,
                unsigned int func,
                uint64_t src_addr,
                uint64_t dst_addr,
                unsigned int compression){
    unsigned int ret,i;
    uint64_t curr_recv_addr, curr_send_addr;

    int curr_send_chunk, curr_recv_chunk = world.local_rank;			 
    //divide into chunks, rounding up
    unsigned int curr_count, count_tail = (count % world.size == 0) ? (count/world.size) : (count%world.size);
    count = (count+world.size-1) / world.size; 
    unsigned int next_in_ring = (world.local_rank+1)%world.size;
    unsigned int prev_in_ring = (world.local_rank+world.size-1)%world.size;
    //scatter reduce - use top-level compression flags directly
    scatter_reduce(count, func, src_addr, dst_addr, compression);
    //allgather in place on dst_buffer
    for (i = 0; i < world.size -1; ++i)
    {
        curr_send_chunk =  curr_recv_chunk;
        curr_recv_chunk = (curr_recv_chunk + world.size - 1) % world.size;
        //TODO: 5.6. can be done together since we have 3 dmas (1rx read from spare, 1tx write in the final dest, 2rx can read and transmit to next in the ring)
        //5. send the local reduced results to next rank
        curr_send_addr = phys_addr_offset(src_addr, count*curr_recv_chunk, (compression & OP0_COMPRESSED) ? true : false);
        curr_count = (curr_send_chunk == (world.size - 1)) ? count_tail : count;
        ret = send(next_in_ring, curr_count, curr_send_addr, TAG_ANY, compression, NO_STREAM);
        if(ret != NO_ERROR) return ret;
        //6. receive the reduced results from previous rank and put into correct dst buffer position
        curr_recv_addr = phys_addr_offset(dst_addr, count*curr_recv_chunk, (compression & RES_COMPRESSED) ? true : false);
        curr_count = (curr_recv_chunk == (world.size - 1)) ? count_tail : count;
        ret = recv(prev_in_ring, curr_count, curr_recv_addr, TAG_ANY, compression);
        if(ret != NO_ERROR) return ret;
    }

    return NO_ERROR;
}
*/

//startup and main

void check_hwid(void){
    // read HWID from hardware and copy it to host-accessible memory
    // TODO: check the HWID against expected
    unsigned int hwid = Xil_In32(GPIO2_DATA_REG);
    Xil_Out32(HWID_OFFSET, hwid);
}

//initialize the system
void init(void) {
    // initialize exchange memory to zero.
    int myoffset;
    for ( myoffset = EXCHMEM_BASEADDR; myoffset < END_OF_EXCHMEM; myoffset +=4) {
        Xil_Out32(myoffset, 0);
    }
    // Check hardware ID
    check_hwid();
    //deactivate reset of all peripherals
    SET(GPIO_DATA_REG, GPIO_SWRST_MASK);
    //enable access from host to exchange memory by removing reset of interface
    SET(GPIO_DATA_REG, GPIO_READY_MASK);

}

//reset the control module
//since this cancels any dma movement in flight and 
//clears the queues it's necessary to reset the dma tag and dma_tag_lookup
void encore_soft_reset(void){
    //1. activate reset pin  
    CLR(GPIO_DATA_REG, GPIO_SWRST_MASK);
    //2. re-initialize
    init();
}

//poll for a call from the host
static inline void wait_for_call(void) {
    // Poll the host cmd queue
    unsigned int invalid;
    do {
        invalid = 0;
        invalid += tngetd(CMD_CALL);
    } while (invalid);
}

//signal finish to the host and write ret value in exchange mem
void finalize_call(unsigned int retval) {
    Xil_Out32(RETVAL_OFFSET, retval);
    // Done: Set done and idle
    putd(STS_CALL, retval);
}

int run_accl() {
    unsigned int retval;
    unsigned int scenario, count, comm, root_src_dst, function, msg_tag;
    unsigned int datapath_cfg, compression_flags, stream_flags;
    unsigned int op0_addrl, op0_addrh, op1_addrl, op1_addrh, res_addrl, res_addrh;
    uint64_t op0_addr, op1_addr, res_addr;

    init();
    //register exception handler though setjmp. it will save stack status to unroll stack when the jmp is performed 
    //	ref: https://www.gnu.org/software/libc/manual/html_node/Longjmp-in-Handler.html
    // when first executed, setjmp returns 0. call to longjmp pass a value != 0 to indicate a jmp is occurred
    retval = setjmp(excp_handler);
    if(retval)
    {
        finalize_call(retval);
    }

    while (1) {
        wait_for_call();
        
        //read parameters from host command queue
        scenario            = getd(CMD_CALL);
        count               = getd(CMD_CALL);
        comm                = getd(CMD_CALL);
        root_src_dst        = getd(CMD_CALL);
        function            = getd(CMD_CALL);
        msg_tag             = getd(CMD_CALL);
        datapath_cfg        = getd(CMD_CALL);
        compression_flags   = getd(CMD_CALL);
        stream_flags        = getd(CMD_CALL);
        op0_addrl           = getd(CMD_CALL);
        op0_addrh           = getd(CMD_CALL);
        op1_addrl           = getd(CMD_CALL);
        op1_addrh           = getd(CMD_CALL);
        res_addrl           = getd(CMD_CALL);
        res_addrh           = getd(CMD_CALL);

        op0_addr = ((uint64_t) op0_addrh << 32) | op0_addrl;
        op1_addr = ((uint64_t) op1_addrh << 32) | op1_addrl;
        res_addr = ((uint64_t) res_addrh << 32) | res_addrl;

        //initialize arithmetic/compression config and communicator
        //NOTE: these are global because they're used in a lot of places but don't change during a call
        //TODO: determine if they can remain global in hierarchical collectives
        arcfg = *((datapath_arith_config *)(cfgmem+datapath_cfg/4));
        world = find_comm(comm);
        
        switch (scenario)
        {
            case XCCL_CONFIG:
                retval = 0;
                switch (function)
                {
                    case HOUSEKEEP_SWRST:
                        encore_soft_reset();
                        break;
                    case HOUSEKEEP_PKTEN:
                        start_depacketizer(NET_RXPKT_BASEADDR);
                        start_packetizer(NET_TXPKT_BASEADDR, MAX_PACKETSIZE);
                        break;
                    case HOUSEKEEP_TIMEOUT:
                        timeout = count;
                        break;
                    case OPEN_PORT:
                        if(use_tcp == 1){
                            retval = openPort();
                        } else{
                            retval = OPEN_PORT_NOT_SUCCEEDED;
                        }
                        break;
                    case OPEN_CON:
                        if(use_tcp == 1){
                            retval = openCon();
                        } else{
                            retval = OPEN_CON_NOT_SUCCEEDED;
                        }
                        break;
                    case SET_STACK_TYPE:
                        use_tcp = count;
                        break;
                    case SET_DMA_TRANSACTION_SIZE:
                        retval = DMA_NOT_EXPECTED_BTT_ERROR;
                        if(count < DMA_MAX_BTT){
                            dma_transaction_size = count;
                            retval = NO_ERROR;
                        }
                        break;
                    case SET_MAX_DMA_TRANSACTIONS:
                        retval = DMA_NOT_OKAY_ERROR;
                        if(count < DMA_MAX_TRANSACTIONS){
                            max_dma_in_flight = count;
                            retval = NO_ERROR;
                        }
                        break;
                    default:
                        break;
                }
                
                break;
            case XCCL_COPY:
                retval = copy(count, op0_addr, res_addr, datapath_cfg, compression_flags, stream_flags);
                break;
            case XCCL_COMBINE:
                retval = combine(count, function, op0_addr, op1_addr, res_addr, datapath_cfg, compression_flags, stream_flags);
                break;
            case XCCL_SEND:
                retval = send(root_src_dst, count, op0_addr, comm, datapath_cfg, msg_tag, compression_flags, stream_flags);
                break;
            case XCCL_RECV:
                retval = recv(root_src_dst, count, res_addr, datapath_cfg, msg_tag, compression_flags);
                break;
            // case XCCL_BCAST:
            //     retval = broadcast(count, root_src_dst, op0_addr, compression_flags);
            //     break;
            // case XCCL_SCATTER:
            //     retval = scatter(count, root_src_dst, op0_addr, res_addr, compression_flags);
            //     break;
            // case XCCL_GATHER:
            //     retval = gather(count, root_src_dst, op0_addr, res_addr, compression_flags);
            //     break;
            // case XCCL_REDUCE:
            //     retval = reduce(count, function, root_src_dst, op0_addr, res_addr, compression_flags);
            //     break;
            // case XCCL_ALLGATHER:
            //     retval = allgather(count, op0_addr, res_addr, compression_flags);
            //     break;
            // case XCCL_ALLREDUCE:
            //     retval = allreduce(count, function, op0_addr, res_addr, compression_flags);
            //     break;
            // case XCCL_REDUCE_SCATTER:
            //     retval = scatter_reduce(count, function, op0_addr, res_addr, compression_flags);
            //     break;
            default:
                retval = NO_ERROR;
                break;
        }

        finalize_call(retval);
    }
    return 0;
}

#ifdef ACCL_SYNTHESIS
int main(int argc, char **argv){
    return run_accl();
}
#endif