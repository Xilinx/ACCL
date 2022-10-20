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

#ifndef MB_FW_EMULATION
#include "xparameters.h"
#include "mb_interface.h"
#endif

static volatile int 		 use_tcp = 1;
static volatile unsigned int timeout = 1 << 28;
static volatile	unsigned int max_segment_size = DMA_MAX_BTT;

static datapath_arith_config arcfg;
static communicator world;
static bool comm_cached = false;
static bool comm_cache_adr;

#ifdef MB_FW_EMULATION
//uint32_t sim_cfgmem[END_OF_EXCHMEM/4];
uint32_t sim_cfgmem[(GPIO_BASEADDR+0x1000)/4];
uint32_t *cfgmem = sim_cfgmem;
hlslib::Stream<ap_axiu<32,0,0,0>, 512> cmd_fifos[4];
hlslib::Stream<ap_axiu<32,0,0,0>, 512> sts_fifos[4];
#else
uint32_t *cfgmem = (uint32_t *)(EXCHMEM_BASEADDR);
#endif

//utility functions
/*Function to find minimum of x and y*/
inline int min(int x, int y){
    return  y + ((x - y) & ((x - y) >> 31));
}
 
/*Function to find maximum of x and y*/
inline int max(int x, int y){
    return x - ((x - y) & ((x - y) >> 31));
}

//retrieves all the communicator
static inline communicator find_comm(unsigned int adr){
	communicator ret;
	ret.size 		= Xil_In32(adr);
	ret.local_rank 	= Xil_In32(adr+4);
	if(ret.size != 0 && ret.local_rank < ret.size){
		ret.ranks = (comm_rank*)(cfgmem+adr/4+2);
	} else {
		ret.size = 0;
		ret.local_rank = 0;
		ret.ranks = NULL;
	}
	return ret;
}

//Packetizer/Depacketizer
static inline void start_packetizer(unsigned int max_pktsize) {
    //get number of DATAPATH_WIDTH_BYTES transfers corresponding to max_pktsize (rounded down)
    unsigned int max_pkt_transfers = max_pktsize/DATAPATH_WIDTH_BYTES;
    Xil_Out32(NET_TXPKT_BASEADDR+0x10, max_pkt_transfers);
    SET(NET_TXPKT_BASEADDR, CONTROL_REPEAT_MASK | CONTROL_START_MASK);
}

static inline void start_depacketizer() {
    SET(NET_RXPKT_BASEADDR, CONTROL_REPEAT_MASK | CONTROL_START_MASK );
}

static inline void start_offload_engines(){
    //start rxbuf enqueue
    Xil_Out32(RX_ENQUEUE_BASEADDR+0x10, EXCHMEM_BASEADDR);
    SET(RX_ENQUEUE_BASEADDR, CONTROL_REPEAT_MASK | CONTROL_START_MASK);
    //start rxbuf dequeue
    Xil_Out32(RX_DEQUEUE_BASEADDR+0x10, EXCHMEM_BASEADDR);
    SET(RX_DEQUEUE_BASEADDR, CONTROL_REPEAT_MASK | CONTROL_START_MASK);
    //start rxbuf seek
    Xil_Out32(RX_SEEK_BASEADDR+0x10, EXCHMEM_BASEADDR);
    SET(RX_SEEK_BASEADDR, CONTROL_REPEAT_MASK | CONTROL_START_MASK);
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
    ret = NO_ERROR;
    for (int i = 0; i < size; i++)
    {
        if (i != local_rank)
        {
            cur_rank_ip 	= world.ranks[i].ip;
            cur_rank_port 	= world.ranks[i].port;
            //send open connection request to the packetizer
            putd(CMD_NET_CON, cur_rank_ip);
            putd(CMD_NET_CON, cur_rank_port);

            //wait until the connections status is returned
            session = getd(STS_NET_CON);
            dst_ip = getd(STS_NET_CON);
            dst_port = getd(STS_NET_CON);
            success = getd(STS_NET_CON);

            if(success){
                world.ranks[i].session = session;
            } else {
                ret = OPEN_CON_NOT_SUCCEEDED;
            }
        }
    }
    return ret;
}

//close connection with every other rank in the communicator
int closeCon()
{
    unsigned int cur_rank_sess_id;
    int ret = NO_ERROR;

    unsigned int size 		= world.size;
    unsigned int local_rank = world.local_rank;

    //close connection to all the ranks except for the local rank
    for (int i = 0; i < size; i++)
    {
        if (i != local_rank)
        {
            cur_rank_sess_id 	= world.ranks[i].session;
            //send close connection request to the packetizer
            putd(CMD_NET_CON, 0);
            putd(CMD_NET_CON, cur_rank_sess_id);
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

//configure datapath before calling this method
//instructs the data plane to move data
//use MOVE_IMMEDIATE
void start_move(
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
    int32_t  op0_stride,
    int32_t  op1_stride,
    int32_t  res_stride,
    uint32_t rx_src_rank,
    uint32_t rx_tag,
    uint32_t tx_dst_rank,
    uint32_t tx_tag
) {
    //TODO: mask everything to the correct bitwidth
    uint32_t opcode = 0;
    opcode |= op0_opcode;
    opcode |= op1_opcode << 3;
    opcode |= res_opcode << 6;
    opcode |= remote_flags << 9;
    bool res_is_remote = (remote_flags == RES_REMOTE);
    opcode |= (compression_flags & 0x7) << 10;//mask is to prevent ETH_COMPRESSED flag leaking into func_id
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
        putd(CMD_DMA_MOVE, (uint32_t)op0_stride);
    }
    //get addr for op1, or equivalents
    if(op1_opcode == MOVE_IMMEDIATE){
        putd(CMD_DMA_MOVE, (uint32_t)op1_addr);
        putd(CMD_DMA_MOVE, (uint32_t)(op1_addr>>32));
    } else if(op1_opcode == MOVE_ON_RECV){
        putd(CMD_DMA_MOVE, rx_src_rank);
        putd(CMD_DMA_MOVE, rx_tag);
    } else if(op1_opcode == MOVE_STRIDE){
        putd(CMD_DMA_MOVE, (uint32_t)op1_stride);
    }
    //get addr for res, or equivalents
    if(res_opcode == MOVE_IMMEDIATE){
        putd(CMD_DMA_MOVE, (uint32_t)res_addr);
        putd(CMD_DMA_MOVE, (uint32_t)(res_addr>>32));
    } else if(res_opcode == MOVE_STRIDE){
        putd(CMD_DMA_MOVE, (uint32_t)res_stride);
    }
    //get send related stuff, if result is remote or stream
    if(res_is_remote || res_opcode == MOVE_STREAM){
        putd(CMD_DMA_MOVE, tx_tag);
    }
    if(res_is_remote || op1_opcode == MOVE_ON_RECV){
        putd(CMD_DMA_MOVE, comm_offset/4);
    }
    if(res_is_remote){
        putd(CMD_DMA_MOVE, tx_dst_rank);
    }
}

inline int end_move(){
    return getd(STS_DMA_MOVE);
}

int move(
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
    int32_t  op0_stride,
    int32_t  op1_stride,
    int32_t  res_stride,
    uint32_t rx_src_rank,
    uint32_t rx_tag,
    uint32_t tx_dst_rank,
    uint32_t tx_tag
){
    start_move(
        op0_opcode, op1_opcode, res_opcode,
        compression_flags, remote_flags,
        func_id, count,
        comm_offset, arcfg_offset,
        op0_addr, op1_addr, res_addr,
        op0_stride, op1_stride, res_stride,
        rx_src_rank, rx_tag,
        tx_dst_rank, tx_tag
    );
    return end_move();
}

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
        compression, RES_LOCAL, function,
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
    //if ETH_COMPRESSED is set, also set RES_COMPRESSED
    compression |= (compression & ETH_COMPRESSED) >> 1;
    //TODO: when doing a one-sided send to a remote stream, check:
    //dst_tag is > 8 (for correct routing on remote side)
    //destination compression == Ethernet compression
    //(since data can't be decompressed on remote side)
    return move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        MOVE_NONE,
        (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
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
            unsigned int comm_offset,
            unsigned int arcfg_offset,
            unsigned int src_tag,
            unsigned int compression,
            unsigned int stream){
    //if ETH_COMPRESSED is set, also set OP1_COMPRESSED
    compression |= (compression & ETH_COMPRESSED) >> 2;
    return move(
        MOVE_NONE,
        MOVE_ON_RECV, 
        (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        compression, RES_LOCAL, 0,
        count, comm_offset, arcfg_offset, 0, 0, dst_addr, 0, 0, 0,
        src_rank, src_tag, 0, (stream & RES_STREAM) ? src_tag : 0
    );
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
int wait_on_rx(
    unsigned int src_rank,
    unsigned int count,
    unsigned int src_tag
){
    int idx, i;
    for(i = 0; timeout == 0 || i < timeout; i++){
        idx = seek_rx_buffer(src_rank, count, src_tag);
        if(idx >= 0) return idx;
    }
    return -1;
}

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
        unsigned int comm_offset,
        unsigned int arcfg_offset,
        unsigned int src_tag,
        unsigned int compression,
        unsigned int stream)
    {
    unsigned int err = NO_ERROR;

    //figure out compression
    //if ETH_COMPRESSED flag is set, then we need to set OP1_COMPRESSED (for the recv)
    //keep OP0_COMPRESSED and RES_COMPRESSED as-is
    compression = compression | ((compression & ETH_COMPRESSED) >> 2);

    err |= move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        MOVE_ON_RECV,
        (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        compression, RES_LOCAL, func,
        count,
        comm_offset, arcfg_offset, 
        op0_addr, 0, dst_addr, 0, 0, 0,
        src_rank, src_tag, 0, 0
    );

    return err;
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
        unsigned int comm_offset,
        unsigned int arcfg_offset,
        unsigned int mpi_tag,
        unsigned int compression,
        unsigned int stream)
    {

    unsigned int err = NO_ERROR;

    //figure out compression
    //if ETH_COMPRESSED flag is set, then we need to set OP1_COMPRESSED (for the recv)
    //and RES_COMPRESSED (for the send)
    //keep OP0_COMPRESSED as-is
    compression = (compression & OP0_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 1) | ((compression & ETH_COMPRESSED) >> 2);

    err |= move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE, 
        MOVE_ON_RECV,
        MOVE_IMMEDIATE, 
        compression, RES_REMOTE, func,
        count,
        comm_offset, arcfg_offset, 
        op0_addr, 0, 0, 0, 0, 0,
        src_rank, mpi_tag, dst_rank, mpi_tag
    );

    return err;
}

//COLLECTIVES

//root reads multiple times the same segment and send it to each rank before 
//moving to the next part of the buffer to be transmitted.
//use MOVE_IMMEDIATE and MOVE_INCREMENTING and MOVE_REPEATING in sequence 
int broadcast(  unsigned int count,
                unsigned int src_rank,
                uint64_t buf_addr,
                unsigned int comm_offset, 
                unsigned int arcfg_offset,
                unsigned int compression,
                unsigned int stream){

    int err = NO_ERROR;
    unsigned int max_seg_count; 
    int elems_remaining = count;
    //convert max segment size to max segment count
    //if pulling from a stream, segment size is irrelevant and we use the 
    //count directly because streams can't be read losslessly
    if(stream & OP0_STREAM){
        max_seg_count = count;
    } else{
        //compute count from uncompressed elem bytes in aright config
        //instead of Xil_In32 we could use:
        //(datapath_arith_config*)(arcfg_offset)->uncompressed_elem_bytes;
        max_seg_count = max_segment_size / Xil_In32(arcfg_offset);
    }

    while(elems_remaining > 0){
        //determine if we're sending or receiving
        if(src_rank == world.local_rank){
            //on the root we only care about ETH_COMPRESSED and OP0_COMPRESSED
            //so replace RES_COMPRESSED with ETH_COMPRESSED
            compression = compression | ((compression & ETH_COMPRESSED) >> 1);

            //send a segment to each of the ranks (excluding self)
            for(int i=0; i < world.size; i++){
                start_move(
                    (i==0) ? ((elems_remaining == count) ? MOVE_IMMEDIATE : MOVE_INCREMENT) : MOVE_REPEAT, 
                    MOVE_NONE, 
                    MOVE_IMMEDIATE, 
                    compression, RES_REMOTE, 0,
                    (i == src_rank) ? 0 : min(max_seg_count, elems_remaining), 
                    comm_offset, arcfg_offset, 
                    buf_addr, 0, 0, 0, 0, 0,
                    0, 0, i, TAG_ANY
                );
            }
            for(int i=0; i < world.size; i++){
                err |= end_move();
            }
        } else{
            //on non-root nodes we only care about ETH_COMPRESSED and RES_COMPRESSED
            //so replace OP1_COMPRESSED with the value of ETH_COMPRESSED
            compression = compression | ((compression & ETH_COMPRESSED) >> 2);
            err |= move(
                MOVE_NONE,
                MOVE_ON_RECV, 
                (elems_remaining == count) ? MOVE_IMMEDIATE : MOVE_INCREMENT, 
                compression, RES_LOCAL, 0,
                min(max_seg_count, elems_remaining),
                comm_offset, arcfg_offset, 
                0, 0, buf_addr, 0, 0, 0,
                src_rank, TAG_ANY, 0, 0
            );
        }
        elems_remaining -= max_seg_count;
    }
    return err;
}

//scatter segment at a time. root sends each rank a segment in a round robin fashion 
//use MOVE_IMMEDIATE and MOVE_INCREMENTING and MOVE_STRIDE in sequence 
int scatter(unsigned int count,
            unsigned int src_rank,
            uint64_t src_buf_addr,
            uint64_t dst_buf_addr,
            unsigned int comm_offset, 
            unsigned int arcfg_offset,
            unsigned int compression,
            unsigned int stream){

    //TODO: implement segmentation
    //TODO: scattering from/to stream

    int err = NO_ERROR;

    //determine if we're sending or receiving
    if(src_rank == world.local_rank){
        //on the root we only care about ETH_COMPRESSED and OP0_COMPRESSED
        //so replace RES_COMPRESSED with ETH_COMPRESSED
        compression = compression | ((compression & ETH_COMPRESSED) >> 1);

        for(int i=0; i < world.size; i++){
            start_move(
                (i==0) ? MOVE_IMMEDIATE : MOVE_INCREMENT, 
                MOVE_NONE,
                MOVE_IMMEDIATE, 
                compression, (i==src_rank) ? RES_LOCAL : RES_REMOTE, 0,
                count, 
                comm_offset, arcfg_offset, 
                src_buf_addr, 0, dst_buf_addr, 0, 0, 0,
                0, 0, i, TAG_ANY
            );
        }
        for(int i=0; i < world.size; i++){
            err |= end_move();
        }
    } else{
        //on non-root odes we only care about ETH_COMPRESSED and RES_COMPRESSED
        //so replace OP0_COMPRESSED with the value of ETH_COMPRESSED
        compression = compression | ((compression & ETH_COMPRESSED) >> 2);
        err |= move(
            MOVE_NONE,
            MOVE_ON_RECV, 
            MOVE_IMMEDIATE, 
            compression, RES_LOCAL, 0,
            count,
            comm_offset, arcfg_offset, 
            0, 0, dst_buf_addr, 0, 0, 0,
            src_rank, TAG_ANY, 0, 0
        );
    }

    return err;
}

//ring gather: non root relay data to the root. root copies segments in dst buffer as they come.
//on root, SEND_ON_RECV
//elsewhere MOVE_STRIDE then SEND_ON_RECV for relay
int gather( unsigned int count,
            unsigned int root_rank,
            uint64_t src_buf_addr,
            uint64_t dst_buf_addr,
            unsigned int comm_offset, 
            unsigned int arcfg_offset,
            unsigned int compression,
            unsigned int stream){
    uint64_t tmp_buf_addr;
    unsigned int i, curr_pos, next_in_ring, prev_in_ring, number_of_shift;
    int err = NO_ERROR;

    next_in_ring = (world.local_rank + 1) % world.size;
    prev_in_ring = (world.local_rank + world.size - 1) % world.size;
    
    //TODO: compute compression

    if(root_rank == world.local_rank){ //root ranks mainly receives

        //we need to compute correct compression schemes from the input compression flags
        //copy to self: keep all flags except ETH_COMPRESSED, which should be reset for safety
        //recv: keep RES_COMPRESSED and replace OP0_COMPRESSED with value of ETH_COMPRESSED
        unsigned int copy_compression = compression & ~ETH_COMPRESSED;
        unsigned int recv_compression = compression | ((compression & ETH_COMPRESSED) >> 2);

        //initialize destination address in offload core
        start_move(
            MOVE_NONE,
            MOVE_NONE, 
            MOVE_IMMEDIATE, 
            NO_COMPRESSION, RES_LOCAL, NO_STREAM,
            0,
            comm_offset, arcfg_offset,
            0, 0, dst_buf_addr, 0, 0, 0,
            0, 0, 0, 0
        );

        //receive from all members of the communicator
        curr_pos = world.local_rank;
        for(i=0; i<world.size; i++){
            start_move(
                (i==0) ? MOVE_IMMEDIATE : MOVE_NONE,
                (i==0) ? MOVE_NONE : MOVE_ON_RECV, 
                MOVE_STRIDE,
                (i==0) ? copy_compression : recv_compression, RES_LOCAL, NO_STREAM,
                count,
                comm_offset, arcfg_offset,
                src_buf_addr, 0, 0, 0, 0, count*((i==0) ? curr_pos : ((curr_pos==(world.size-1)) ? (world.size-1) : -1)),
                prev_in_ring, TAG_ANY, 0, 0
            );
            //update current position
            curr_pos = (curr_pos + world.size - 1) % world.size;
        }
        for(i=0; i<=world.size; i++){
            err |= end_move();
        }
    }else{
        //non root ranks sends their data + relay others data to the next rank in sequence
        // as a daisy chain

        //we need to compute correct compression schemes from the input compression flags
        //send: keep all flags except RES_COMPRESSED, which should be replaced by ETH_COMPRESSED
        //relay: keep ETH_COMPRESSED, reset everything else
        unsigned int send_compression = (compression & ~RES_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 1);
        unsigned int relay_compression = ((compression & ETH_COMPRESSED) >> 1) | ((compression & ETH_COMPRESSED) >> 2);

        //first send our own data
        start_move(
            MOVE_IMMEDIATE, 
            MOVE_NONE,
            MOVE_IMMEDIATE, 
            send_compression, RES_REMOTE, NO_STREAM,
            count, 
            comm_offset, arcfg_offset, 
            src_buf_addr, 0, 0, 0, 0, 0,
            0, 0, next_in_ring, TAG_ANY
        );
        //next relay a number of times depending on our position in the ring
        number_of_shift = ((world.size+world.local_rank-root_rank)%world.size) - 1 ; //distance to the root
        for (i=0; i<number_of_shift; i++){	
            start_move(
                MOVE_NONE, 
                MOVE_ON_RECV,
                MOVE_IMMEDIATE, 
                relay_compression, RES_REMOTE, NO_STREAM,
                count, 
                comm_offset, arcfg_offset, 
                0, 0, 0, 0, 0, 0,
                prev_in_ring, TAG_ANY, next_in_ring, TAG_ANY
            );
        }	
        for(i=0; i<=number_of_shift; i++){
            err |= end_move();
        }
    }
    return err;
}

//naive fused: 1) receive a segment 2) move in the dest buffer 3) relay to next rank 
int allgather(
    unsigned int count,
    uint64_t src_buf_addr,
    uint64_t dst_buf_addr,
    unsigned int comm_offset, 
    unsigned int arcfg_offset,
    unsigned int compression,
    unsigned int stream
){
    int i, curr_pos, rel_stride, abs_stride, next_in_ring, prev_in_ring;
    int err = NO_ERROR;

    //compression is tricky for the relay: we've already received into the destination buffer 
    //with associated flag RES_COMPRESSED; this buffer becomes the source for a send
    //so if RES_COMPRESSED is set, OP0_COMPRESSED must be set for the send, and RES_COMPRESSED reset
    unsigned int relay_compression = (compression & RES_COMPRESSED) ? (compression | OP0_COMPRESSED) : compression;
    relay_compression &= ~(RES_COMPRESSED);

    next_in_ring = (world.local_rank + 1) % world.size;
    prev_in_ring = (world.local_rank + world.size - 1) % world.size;

    //prime the address slot for the destination, so we can subsequently stride against it
    start_move(
        MOVE_NONE, MOVE_NONE, MOVE_IMMEDIATE, 
        NO_COMPRESSION, RES_LOCAL, 0,
        0,
        0, arcfg_offset, 
        src_buf_addr, 0, dst_buf_addr, 0, 0, 0,
        0, 0, 0, 0
    );

    //copy our local data into the appropriate destination slot
    start_move(
        MOVE_IMMEDIATE, MOVE_NONE, MOVE_STRIDE, 
        compression, RES_LOCAL, 0,
        count,
        0, arcfg_offset, 
        src_buf_addr, 0, 0, 0, 0, count*world.local_rank,
        0, 0, 0, 0
    );

    //send to next in ring
    //ETH_COMPRESSED flag overwrites RES_COMPRESSED
    start_move(
        MOVE_IMMEDIATE, MOVE_NONE, MOVE_IMMEDIATE, 
        compression | ((compression & ETH_COMPRESSED) >> 1), RES_REMOTE, 0,
        count, 
        comm_offset, arcfg_offset, 
        src_buf_addr, 0, 0, 0, 0, 0,
        0, 0, next_in_ring, TAG_ANY
    );

    err |= end_move();
    err |= end_move();
    err |= end_move();

    //receive and forward from all other members of the communicator
    curr_pos = world.local_rank;
    for(i=0; i<world.size-1; i++){
        rel_stride = count*((curr_pos == 0) ? (world.size-1) : -1);
        abs_stride = count*((curr_pos == 0) ? (world.size-1) : curr_pos-1);

        //we use a blocking move here, because we want to avoid a race condition with the relay below
        //TODO: avoid this; we either need to solve the RAW dependency in hardware (the generic approach),
        //or a way to reuse a rx buffer (solves this problem in particular but does not extend to e.g. reduces)
        //e.g. MOVE_ON_RECV_KEEP which would keep the RX buffer in the pending state
        // on compression: ETH_COMPRESSED flag overwrites OP1_COMPRESSED
        err |= move(
            MOVE_NONE, MOVE_ON_RECV, MOVE_STRIDE, 
            compression | ((compression & ETH_COMPRESSED) >> 2), RES_LOCAL, 0,
            count, 
            comm_offset, arcfg_offset, 
            0, 0, 0, 0, 0, rel_stride,
            prev_in_ring, TAG_ANY, 0, 0
        ); 

        if(i < world.size-2){ //if not the last data, relay to the next in ring
            //first prime the address 
            start_move(
                MOVE_NONE, MOVE_IMMEDIATE, MOVE_NONE, 
                NO_COMPRESSION, RES_REMOTE, 0,
                0, 
                comm_offset, arcfg_offset, 
                0, dst_buf_addr, 0, 0, 0, 0,
                0, 0, next_in_ring, TAG_ANY
            );
            //send
            //here we're copying from the result buffer back into the network, so
            //RES_COMPRESSED flag overwrites OP0_COMPRESSED, and 
            //ETH_COMPRESSED flag overwrites RES_COMPRESSED
            start_move(
                MOVE_NONE, MOVE_STRIDE, MOVE_IMMEDIATE, 
                ((compression & RES_COMPRESSED) >> 2) | ((compression & ETH_COMPRESSED) >> 1), RES_REMOTE, 0,
                count, 
                comm_offset, arcfg_offset, 
                0, 0, 0, 0, abs_stride, 0,
                0, 0, next_in_ring, TAG_ANY
            );

            err |= end_move();
            err |= end_move();
        }
        curr_pos = (curr_pos + world.size - 1) % world.size;
    }

    return err;
}


//every rank receives a buffer it reduces its own buffer and forwards to next rank in the ring
int reduce( unsigned int count,
            unsigned int func,
            unsigned int root_rank,
            uint64_t src_addr,
            uint64_t dst_addr,
            unsigned int comm_offset,
            unsigned int arcfg_offset,
            unsigned int compression,
            unsigned int stream){

    unsigned int next_in_ring = (world.local_rank + 1) % world.size;
    unsigned int prev_in_ring = (world.local_rank + world.size-1) % world.size;

    if(world.size == 1){
        //corner-case copy for when running a single-node reduction
        return copy(count, src_addr, dst_addr, arcfg_offset, compression, stream);
    }else if( prev_in_ring == root_rank){
        //non root ranks immediately after the root sends
        return send(next_in_ring, count, src_addr, comm_offset, arcfg_offset, TAG_ANY, compression, stream);
    }else if (world.local_rank != root_rank){
        //non root ranks sends their data + data received from previous rank to the next rank in sequence as a daisy chain
        return fused_recv_reduce_send(prev_in_ring, next_in_ring, count, func, src_addr, comm_offset, arcfg_offset, TAG_ANY, compression, stream);
    }else{	
        //root only receive from previous node in the ring, add its local buffer and save in destination buffer 
        return fused_recv_reduce(prev_in_ring, count, func, src_addr, dst_addr, comm_offset, arcfg_offset, TAG_ANY, compression,stream);
    }
}

//reduce_scatter: (a,b,c), (1,2,3), (X,Y,Z) -> (a+1+X,,) (,b+2+Y,) (,,c+3+Z)
//count == size of chunks
int reduce_scatter(
    unsigned int count,
    unsigned int func,
    uint64_t src_buf_addr,
    uint64_t dst_buf_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset,
    unsigned int compression,
    unsigned int stream
){
    int i, curr_pos, rel_stride, abs_stride, next_in_ring, prev_in_ring;
    int err = NO_ERROR;
    unsigned int tmp_compression = NO_COMPRESSION;

    if(world.size == 1){
        //corner-case copy for when running a single-node reduction
        return copy(count, src_buf_addr, dst_buf_addr, arcfg_offset, compression, stream);
    }

    next_in_ring = (world.local_rank + 1) % world.size;
    prev_in_ring = (world.local_rank + world.size - 1) % world.size;

    //preamble: send our data to next in ring
    //prime the address slot for the source, so we can subsequently stride against it
    start_move(
        MOVE_IMMEDIATE, MOVE_NONE, MOVE_NONE, 
        NO_COMPRESSION, RES_LOCAL, 0,
        0,
        0, arcfg_offset, 
        src_buf_addr, 0, 0, 0, 0, 0,
        0, 0, 0, 0
    );

    curr_pos = prev_in_ring;

    //send local chunk to next in ring
    //send: keep OP0_COMPRESSED, replace RES_COMPRESSED by ETH_COMPRESSED
    tmp_compression = (compression & OP0_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 1);
    start_move(
        MOVE_STRIDE, MOVE_NONE, MOVE_IMMEDIATE, 
        tmp_compression, RES_REMOTE, 0,
        count, 
        comm_offset, arcfg_offset, 
        0, 0, 0, count*curr_pos, 0, 0,
        0, 0, next_in_ring, TAG_ANY
    );

    //receive and reduce+forward from all other members of the communicator
    for(i=0; i<world.size-1; i++){
        rel_stride = count*((curr_pos == 0) ? (world.size-1) : -1);

        //simultaneous receive, reduce and send for the received chunk,
        //unless it is the last step, in which case we don't send, but save locally
        if(i < world.size-2){
            tmp_compression = (compression & OP0_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 2) | ((compression & ETH_COMPRESSED) >> 1);
            start_move(
                MOVE_STRIDE, MOVE_ON_RECV, MOVE_IMMEDIATE, 
                tmp_compression, RES_REMOTE, func,
                count, 
                comm_offset, arcfg_offset, 
                0, 0, 0, rel_stride, 0, 0,
                prev_in_ring, TAG_ANY, next_in_ring, TAG_ANY
            ); 
        } else{
            tmp_compression = compression | ((compression & ETH_COMPRESSED) >> 2);
            start_move(
                MOVE_STRIDE, MOVE_ON_RECV, MOVE_IMMEDIATE, 
                tmp_compression, RES_LOCAL, 0,
                count, 
                comm_offset, arcfg_offset, 
                0, 0, dst_buf_addr, rel_stride, 0, 0,
                prev_in_ring, TAG_ANY, 0, 0
            ); 
        }
        curr_pos = (curr_pos + world.size - 1) % world.size;
        //pop one result here to keep the result FIFO not full
        err |= end_move();
    }

    //pop final two results
    err |= end_move();
    err |= end_move();

    return err;
}

//2 stage allreduce: fused reduce_scatter+all_gather
int allreduce(
    unsigned int count,
    unsigned int func,
    uint64_t src_buf_addr,
    uint64_t dst_buf_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset,
    unsigned int compression,
    unsigned int stream
){
    int i, curr_pos, curr_count, rel_stride, abs_stride, next_in_ring, prev_in_ring;
    int err = NO_ERROR;
    unsigned int tmp_compression = NO_COMPRESSION;

    if(world.size == 1){
        //corner-case copy for when running a single-node reduction
        return copy(count, src_buf_addr, dst_buf_addr, arcfg_offset, compression, stream);
    }

    next_in_ring = (world.local_rank + 1) % world.size;
    prev_in_ring = (world.local_rank + world.size - 1) % world.size;

    //we need to break the input into world.size chunks of equal size
    //if count does not divide by world.size, the chunk with the largest index (tail) will be smaller
    unsigned int bulk_count = (count + world.size -1) / world.size;//equivalent to ceil(count/world.size)
    unsigned int tail_count = count - bulk_count*(world.size-1);

    //preamble: send our data to next in ring
    //prime the address slots for the source and destination, 
    //so we can subsequently stride against them
    start_move(
        MOVE_IMMEDIATE, MOVE_NONE, MOVE_IMMEDIATE, 
        NO_COMPRESSION, RES_LOCAL, 0,
        0,
        0, arcfg_offset, 
        src_buf_addr, 0, dst_buf_addr, 0, 0, 0,
        0, 0, 0, 0
    );

    //send local chunk to next in ring
    //send: keep OP0_COMPRESSED, replace RES_COMPRESSED by ETH_COMPRESSED
    tmp_compression = (compression & OP0_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 1);
    start_move(
        MOVE_STRIDE, MOVE_NONE, MOVE_IMMEDIATE, 
        tmp_compression, RES_REMOTE, 0,
        (world.local_rank == world.size-1) ? tail_count: bulk_count, 
        comm_offset, arcfg_offset, 
        0, 0, 0, bulk_count*world.local_rank, 0, 0,
        0, 0, next_in_ring, TAG_ANY
    );

    //receive and reduce+forward from all other members of the communicator
    curr_pos = world.local_rank;
    for(i=0; i<world.size-1; i++){
        rel_stride = bulk_count*((curr_pos == 0) ? (world.size-1) : -1);
        curr_count = (curr_pos == 0) ? tail_count : bulk_count;

        //simultaneous receive, reduce and send for the received chunk,
        //unless it is the last step, in which case we don't send, but save locally
        //unlike normal reduce-scatter, we don't save at offset 0 in the destination buffer
        //but at the appropriate offset for the data being saved
        if(i < world.size-2){
            //compression: if ETH_COMPRESSED is set then we receive compressed data over the network,
            //so we must set OP1_COMPRESSED here. Also keep OP0_COMPRESSED the same since we're reading from 
            //initial data. RES_COMPRESSED must be replaced by ETH_COMPRESSED since the result
            //goes out to network
            tmp_compression = (compression & OP0_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 2) | ((compression & ETH_COMPRESSED) >> 1);
            start_move(
                MOVE_STRIDE, MOVE_ON_RECV, MOVE_IMMEDIATE, 
                tmp_compression, RES_REMOTE, func,
                curr_count, 
                comm_offset, arcfg_offset, 
                0, 0, 0, rel_stride, 0, 0,
                prev_in_ring, TAG_ANY, next_in_ring, TAG_ANY
            ); 
        } else{
            //compression: if ETH_COMPRESSED is set then we receive compressed data over the network,
            //so we must set OP1_COMPRESSED here. Also keep RES_COMPRESSED and OP0_COMPRESSED the same.
            tmp_compression = compression | ((compression & ETH_COMPRESSED) >> 2);
            start_move(
                MOVE_STRIDE, MOVE_ON_RECV, MOVE_STRIDE, 
                tmp_compression, RES_LOCAL, 0,
                curr_count, 
                comm_offset, arcfg_offset, 
                0, 0, 0, rel_stride, 0, bulk_count*next_in_ring,
                prev_in_ring, TAG_ANY, 0, 0
            ); 
        }
        curr_pos = (curr_pos + world.size - 1) % world.size;
        //pop one result here to keep the result FIFO not full
        err |= end_move();
    }

    //pop final two results for reduce-scatter
    err |= end_move();
    err |= end_move();

    //next phase: allgather

    //send to next in ring, from dst_buf_addr where we stored the scattered reduction result
    start_move(
        MOVE_IMMEDIATE, MOVE_NONE, MOVE_NONE, 
        NO_COMPRESSION, RES_LOCAL, 0,
        0, 
        0, arcfg_offset, 
        dst_buf_addr, 0, 0, 0, 0, 0,
        0, 0, 0, 0
    );
    //send: keep all flags except RES_COMPRESSED, which should be replaced by ETH_COMPRESSED
    tmp_compression = (compression & ~RES_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 1);
    start_move(
        MOVE_STRIDE, MOVE_NONE, MOVE_IMMEDIATE, 
        tmp_compression, RES_REMOTE, 0,
        (next_in_ring == world.size-1) ? tail_count : bulk_count, 
        comm_offset, arcfg_offset, 
        0, 0, 0, bulk_count*next_in_ring, 0, 0,
        0, 0, next_in_ring, TAG_ANY
    );

    err |= end_move();
    err |= end_move();

    //receive and forward from all other members of the communicator
    curr_pos = next_in_ring;
    for(i=0; i<world.size-1; i++){
        rel_stride = bulk_count*((curr_pos == 0) ? (world.size-1) : -1);
        curr_count = (curr_pos == 0) ? tail_count : bulk_count;

        //we use a blocking move here, because we want to avoid a race condition with the relay below
        //TODO: avoid this; we either need to solve the RAW dependency in hardware (the generic approach),
        //or a way to reuse a rx buffer (solves this problem in particular but does not extend to e.g. reduces)
        //e.g. MOVE_ON_RECV_KEEP which would keep the RX buffer in the pending state
        tmp_compression = (compression & RES_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 2);
        err |= move(
            MOVE_NONE, MOVE_ON_RECV, MOVE_STRIDE, 
            tmp_compression, RES_LOCAL, 0,
            curr_count, 
            comm_offset, arcfg_offset, 
            0, 0, 0, 0, 0, rel_stride,
            prev_in_ring, TAG_ANY, 0, 0
        ); 
        curr_pos = (curr_pos + world.size - 1) % world.size;
        curr_count = (curr_pos == (world.size - 1)) ? tail_count : bulk_count;
        if(i < world.size-2){ //if not the last data, relay to the next in ring
            //first prime the address 
            start_move(
                MOVE_NONE, MOVE_IMMEDIATE, MOVE_NONE, 
                NO_COMPRESSION, RES_REMOTE, 0,
                0, 
                0, arcfg_offset, 
                0, dst_buf_addr, 0, 0, 0, 0,
                0, 0, 0, 0
            );
            //send
            //we're re-sending from the result, so copy RES_COMPRESSED over OP1_COMPRESSED,
            //and ETH_COMPRESSED over RES_COMPRESSED
            tmp_compression = ((compression & RES_COMPRESSED) >> 1) | ((compression & ETH_COMPRESSED) >> 1);
            start_move(
                MOVE_NONE, MOVE_STRIDE, MOVE_IMMEDIATE, 
                tmp_compression, RES_REMOTE, 0,
                curr_count, 
                comm_offset, arcfg_offset, 
                0, 0, 0, 0, bulk_count*curr_pos, 0,
                0, 0, next_in_ring, TAG_ANY
            );

            err |= end_move();
            err |= end_move();
        }
    }

    return err;
}

//barrier: swing a minimal packet around the ring then return
//Ideally count would be  computed such that the message is 64B; this is
//because using packets of size multiples of 64 keeps memory 
//accesses in the TCP stack aligned (so 1B would be slower than 64B)
//requires a buffer from which we pull the 64B
int barrier(
    uint64_t src_buf_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset
){
    unsigned int count = 1;
    int i, next_in_ring, prev_in_ring;
    int err = NO_ERROR;

    if(world.size == 1){
        //corner-case copy for when running a single-node barrier
        return NO_ERROR;
    }

    next_in_ring = (world.local_rank + 1) % world.size;
    prev_in_ring = (world.local_rank + world.size - 1) % world.size;

    //send local chunk to next in ring
    start_move(
        MOVE_IMMEDIATE, MOVE_NONE, MOVE_IMMEDIATE, 
        NO_COMPRESSION, RES_REMOTE, 0,
        count, 
        comm_offset, arcfg_offset, 
        src_buf_addr, 0, 0, 0, 0, 0,
        0, 0, next_in_ring, TAG_ANY
    );

    //receive and forward from all other members of the communicator
    for(i=0; i<world.size-1; i++){
        //simultaneous receive, and send for the received chunk,
        //unless it is the last step, in which case we copy back to the source buffer
        if(i < world.size-2){
            start_move(
                MOVE_NONE, MOVE_ON_RECV, MOVE_IMMEDIATE, 
                NO_COMPRESSION, RES_REMOTE, 0,
                count,
                comm_offset, arcfg_offset, 
                0, 0, 0, 0, 0, 0,
                prev_in_ring, TAG_ANY, next_in_ring, TAG_ANY
            ); 
        } else {
            start_move(
                MOVE_NONE, MOVE_ON_RECV, MOVE_IMMEDIATE, 
                NO_COMPRESSION, RES_LOCAL, 0,
                count,
                comm_offset, arcfg_offset, 
                0, 0, src_buf_addr, 0, 0, 0,
                prev_in_ring, TAG_ANY, 0, 0
            );
        }
        //pop one result here to keep the result FIFO not full
        err |= end_move();
    }

    //pop final result
    err |= end_move();

    return err;
}

//placeholder for all-to-all collective
int all_to_all(){
    return NO_ERROR;
}


//startup and main

void check_hwid(void){
    // read HWID from hardware and copy it to host-accessible memory
    // TODO: check the HWID against expected
    unsigned int hwid = Xil_In32(GPIO2_DATA_REG);
    Xil_Out32(HWID_OFFSET, hwid);
}

//initialize the system
void init(void) {
    // write a zero into CFGRDY SFR, indicating the config is not valid
    // indicating to the driver it needs to configure the CCLO
    Xil_Out32(CFGRDY_OFFSET, 0);
    // Wipe and re-fetch hardware ID
    Xil_Out32(HWID_OFFSET, 0);
    check_hwid();
    //deactivate reset of all peripherals
    SET(GPIO_DATA_REG, GPIO_SWRST_MASK);
    //mark init done
    SET(GPIO_DATA_REG, GPIO_READY_MASK);
}

//reset the control module
//since this cancels any dma movement in flight and 
//clears the queues it's necessary to reset the dma tag and dma_tag_lookup
void encore_soft_reset(void){
    //1. activate reset pin  
    CLR(GPIO_DATA_REG, GPIO_SWRST_MASK);
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

void run() {
    unsigned int retval;
    unsigned int scenario, count, comm, root_src_dst, function, msg_tag;
    unsigned int datapath_cfg, compression_flags, stream_flags;
    unsigned int op0_addrl, op0_addrh, op1_addrl, op1_addrh, res_addrl, res_addrh;
    uint64_t op0_addr, op1_addr, res_addr;

    init();

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
        if(!comm_cached || (comm != comm_cache_adr)){
            world = find_comm(comm);
            comm_cached = true;
            comm_cache_adr = comm;
        }
        
        switch (scenario)
        {
            case ACCL_COPY:
                retval = copy(count, op0_addr, res_addr, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_COMBINE:
                retval = combine(count, function, op0_addr, op1_addr, res_addr, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_SEND:
                retval = send(root_src_dst, count, op0_addr, comm, datapath_cfg, msg_tag, compression_flags, stream_flags);
                break;
            case ACCL_RECV:
                retval = recv(root_src_dst, count, res_addr, comm, datapath_cfg, msg_tag, compression_flags, stream_flags);
                break;
            case ACCL_BCAST:
                retval = broadcast(count, root_src_dst, op0_addr, comm, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_SCATTER:
                retval = scatter(count, root_src_dst, op0_addr, res_addr, comm, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_GATHER:
                retval = gather(count, root_src_dst, op0_addr, res_addr, comm, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_REDUCE:
                retval = reduce(count, function, root_src_dst, op0_addr, res_addr, comm, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_ALLGATHER:
                retval = allgather(count, op0_addr, res_addr, comm, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_REDUCE_SCATTER:
                retval = reduce_scatter(count, function, op0_addr, res_addr, comm, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_ALLREDUCE:
                retval = allreduce(count, function, op0_addr, res_addr, comm, datapath_cfg, compression_flags, stream_flags);
                break;
            case ACCL_BARRIER:
                retval = barrier(op0_addr, comm, datapath_cfg);
                break;
            case ACCL_ALLTOALL:
                retval = all_to_all();
                break;
            case ACCL_CONFIG:
                retval = 0;
                switch (function)
                {
                    case HOUSEKEEP_SWRST:
                        encore_soft_reset();
                        finalize_call(retval);
                        return;
                    case HOUSEKEEP_PKTEN:
                        start_depacketizer();
                        start_packetizer(MAX_PACKETSIZE);
                        start_offload_engines();
                        break;
                    case HOUSEKEEP_TIMEOUT:
                        timeout = count;
                        break;
                    case HOUSEKEEP_OPEN_PORT:
                        if(use_tcp == 1){
                            retval = openPort();
                        } else{
                            retval = OPEN_PORT_NOT_SUCCEEDED;
                        }
                        break;
                    case HOUSEKEEP_OPEN_CON:
                        if(use_tcp == 1){
                            retval = openCon();
                        } else{
                            retval = OPEN_CON_NOT_SUCCEEDED;
                        }
                        break;
                    case HOUSEKEEP_CLOSE_CON:
                        if(use_tcp == 1){
                            retval = closeCon();
                        } else{
                            retval = CLOSE_CON_NOT_SUCCEEDED;
                        }
                        break;
                    case HOUSEKEEP_SET_STACK_TYPE:
                        use_tcp = count;
                        break;
                    case HOUSEKEEP_SET_MAX_SEGMENT_SIZE:
                        retval = DMA_NOT_EXPECTED_BTT_ERROR;
                        if(count < DMA_MAX_BTT){
                            max_segment_size = count;
                            retval = NO_ERROR;
                        }
                        break;
                    default:
                        break;
                }
                break;
            default:
                retval = NO_ERROR;
                break;
        }
        finalize_call(retval);
    }
}

void run_accl() {
    while(true){
        run();
    }
}

#ifndef MB_FW_EMULATION
int main(int argc, char **argv){
    run_accl();
}
#endif
