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

static unsigned int timeout = 1 << 28;
static unsigned int max_eager_size = (1<<15);
static unsigned int max_rendezvous_size = (1<<15);
static unsigned int eager_rx_buf_size = DMA_MAX_BTT;
static unsigned int num_rndzv_pending = 0;
static unsigned int num_retry_pending = 0;
static bool new_call = true;
static bool flush_retries = false;
static unsigned int current_step;
static unsigned int datatype_nbytes;

static datapath_arith_config arcfg;
static communicator world;
static bool comm_cached = false;
static uint32_t comm_cache_adr;

#ifdef MB_FW_EMULATION
//uint32_t sim_cfgmem[END_OF_EXCHMEM/4];
uint64_t duration_us;
uint32_t sim_cfgmem[(GPIO_BASEADDR+0x1000)/4];
uint32_t *cfgmem = sim_cfgmem;
hlslib::Stream<ap_axiu<32,0,0,0>, 512> cmd_fifos[5];
hlslib::Stream<ap_axiu<32,0,0,0>, 512> sts_fifos[5];
#else
uint32_t volatile *cfgmem = (uint32_t volatile *)(EXCHMEM_BASEADDR);
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

/*Function to find log2 of uint32*/
inline unsigned int fast_log2(unsigned int val){
    unsigned int ret;
    unsigned int s;

    ret = (val > 0xFFFF) << 4;
    val >>= ret;
    s = (val > 0xFF) << 3;
    val >>= s;
    ret |= s;
    s = (val > 0xF) << 2;
    val >>= s;
    ret |= s;
    s = (val > 0x3) << 1;
    val >>= s;
    ret |= s;
    ret |= (val >> 1);

    return ret;
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

static inline unsigned int segment(unsigned int number_of_bytes,unsigned int segment_size){
    return  (number_of_bytes + segment_size - 1) / segment_size;
}

static inline uint32_t pack_flags(uint32_t compression_flags, uint32_t remote_flags, uint32_t host_flags){
    uint32_t ret;
    ret = ((host_flags & 0xff) << 16) | ((remote_flags & 0xff) << 8) | (compression_flags & 0xff);
    return ret;
}

static inline uint32_t pack_flags_rendezvous(uint32_t host_flags){
    uint32_t ret;
    ret = (1 << 24) | ((host_flags & 0xff) << 16) | ((RES_REMOTE & 0xff) << 8) | (NO_COMPRESSION & 0xff);
    return ret;
}

//sends a buffer address to a remote peer,
//enabling them to perform a RDMA WRITE to our buffer
void rendezvous_send_addr(uint32_t target_rank, uint64_t addr, bool host, uint32_t count, uint32_t tag){
    putd(CMD_RNDZV, world.ranks[target_rank].session);
    putd(CMD_RNDZV, (uint32_t)addr);
    putd(CMD_RNDZV, (uint32_t)(addr>>32));
    putd(CMD_RNDZV, host);
    putd(CMD_RNDZV, count);
    putd(CMD_RNDZV, tag);
    cputd(CMD_RNDZV, world.local_rank);
}

//receive a buffer address from a remote peer,
//enabling us to perform a RDMA WRITE to their buffer
int rendezvous_get_addr(uint32_t target_rank, uint64_t *target_addr, bool *target_host, uint32_t target_count, uint32_t target_tag){
    uint32_t type, rank, tag;
    uint64_t addr;
    bool match;
    // Check if there are pending notifications in STS_RNDZV_PENDING,
    // otherwise pull from STS_RNDZV
    unsigned int i;
    for(i = 0; i<num_rndzv_pending; i++){
        type = getd(STS_RNDZV_PENDING);
        rank = getd(STS_RNDZV_PENDING);
        tag = getd(STS_RNDZV_PENDING);
        match = (type == 2) && (rank == target_rank) && ((tag == target_tag) || (tag == TAG_ANY));
        if(!match){
            //copy from STS_RNDZV to STS_RNDZV_PENDING
            putd(CMD_RNDZV_PENDING, type);
            putd(CMD_RNDZV_PENDING, rank);
            putd(CMD_RNDZV_PENDING, tag);
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
        } else {
            addr = (uint64_t)getd(STS_RNDZV_PENDING);
            addr |= (uint64_t)getd(STS_RNDZV_PENDING) << 32;
            *target_addr = addr;
            *target_host = getd(STS_RNDZV_PENDING);
            getd(STS_RNDZV_PENDING);//count
            num_rndzv_pending--;
            return NO_ERROR;
        }
    }
    //if we're still executing, no match in RNDZV_PENDING, check STS_RNDZV
    while(!tngetd(STS_RNDZV)){
        type = getd(STS_RNDZV);
        rank = getd(STS_RNDZV);
        tag = getd(STS_RNDZV);
        match = (type == 2) && (rank == target_rank) && ((tag == target_tag) || (tag == TAG_ANY));
        if(!match){
            //copy from STS_RNDZV to STS_RNDZV_PENDING
            putd(CMD_RNDZV_PENDING, type);
            putd(CMD_RNDZV_PENDING, rank);
            putd(CMD_RNDZV_PENDING, tag);
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            num_rndzv_pending++;
        } else {
            addr = (uint64_t)getd(STS_RNDZV);
            addr |= (uint64_t)getd(STS_RNDZV) << 32;
            *target_addr = addr;
            *target_host = getd(STS_RNDZV);
            getd(STS_RNDZV);//count
            return NO_ERROR;
        }
    }
    //we found nothing, signal this up to caller
    return NOT_READY_ERROR;
}

//receive a buffer address from a remote peer,
//enabling us to perform a RDMA WRITE to their buffer
int rendezvous_get_any_addr(uint32_t *target_rank, uint64_t *target_addr, bool *target_host, uint32_t target_count, uint32_t target_tag){
    uint32_t type, rank, tag;
    uint64_t addr;
    bool match;
    // Check if there are pending notifications in STS_RNDZV_PENDING,
    // otherwise pull from STS_RNDZV
    unsigned int i;
    for(i = 0; i<num_rndzv_pending; i++){
        type = getd(STS_RNDZV_PENDING);
        rank = getd(STS_RNDZV_PENDING);
        tag = getd(STS_RNDZV_PENDING);
        match = (type == 2) && ((tag == target_tag) || (tag == TAG_ANY));
        if(!match){
            //copy from STS_RNDZV to STS_RNDZV_PENDING
            putd(CMD_RNDZV_PENDING, type);
            putd(CMD_RNDZV_PENDING, rank);
            putd(CMD_RNDZV_PENDING, tag);
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
        } else {
            *target_rank = rank;
            addr = (uint64_t)getd(STS_RNDZV_PENDING);
            addr |= (uint64_t)getd(STS_RNDZV_PENDING) << 32;
            *target_addr = addr;
            *target_host = getd(STS_RNDZV_PENDING);
            getd(STS_RNDZV_PENDING);//count
            num_rndzv_pending--;
            return NO_ERROR;
        }
    }
    //if we're still executing, no match in RNDZV_PENDING, check STS_RNDZV
    while(!tngetd(STS_RNDZV)){
        type = getd(STS_RNDZV);
        rank = getd(STS_RNDZV);
        tag = getd(STS_RNDZV);
        match = (type == 2) && ((tag == target_tag) || (tag == TAG_ANY));
        if(!match){
            //copy from STS_RNDZV to STS_RNDZV_PENDING
            putd(CMD_RNDZV_PENDING, type);
            putd(CMD_RNDZV_PENDING, rank);
            putd(CMD_RNDZV_PENDING, tag);
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            num_rndzv_pending++;
        } else {
            *target_rank = rank;
            addr = (uint64_t)getd(STS_RNDZV);
            addr |= (uint64_t)getd(STS_RNDZV) << 32;
            *target_addr = addr;
            *target_host = getd(STS_RNDZV);
            getd(STS_RNDZV);//count
            return NO_ERROR;
        }
    }
    //we found nothing, signal this up to caller
    return NOT_READY_ERROR;
}

//receives an acknowledgement that a RDMA WRITE to 
//our buffer has completed
int rendezvous_get_completion(unsigned int target_rank, uint64_t target_addr, bool target_host, uint32_t target_count, uint32_t target_tag){
    uint32_t type, rank, tag;
    uint64_t addr;
    uint64_t addrl, addrh;
    bool match;
    // Check if there are pending notifications in STS_RNDZV_PENDING,
    // otherwise pull from STS_RNDZV
    unsigned int i;
    for(i = 0; i<num_rndzv_pending; i++){
        type = getd(STS_RNDZV_PENDING);
        rank = getd(STS_RNDZV_PENDING);
        tag = getd(STS_RNDZV_PENDING);
        addrl = getd(STS_RNDZV_PENDING);
        addrh = getd(STS_RNDZV_PENDING);
        addr = ((uint64_t)addrh << 32) | (uint64_t)addrl;
        match = (type == 3) && (rank == target_rank) && ((tag == target_tag) || (tag == TAG_ANY)) && (addr == target_addr);
        if(!match){
            //copy from STS_RNDZV to STS_RNDZV_PENDING
            putd(CMD_RNDZV_PENDING, type);
            putd(CMD_RNDZV_PENDING, rank);
            putd(CMD_RNDZV_PENDING, tag);
            putd(CMD_RNDZV_PENDING, addrl);
            putd(CMD_RNDZV_PENDING, addrh);
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV_PENDING));
        } else {
            getd(STS_RNDZV_PENDING);//TODO check host
            getd(STS_RNDZV_PENDING);//count
            num_rndzv_pending--;
            return NO_ERROR;
        }
    }
    //if we're still executing, no match in RNDZV_PENDING, check STS_RNDZV
    while(!tngetd(STS_RNDZV)){
        type = getd(STS_RNDZV);
        rank = getd(STS_RNDZV);
        tag = getd(STS_RNDZV);
        addrl = getd(STS_RNDZV);
        addrh = getd(STS_RNDZV);
        addr = ((uint64_t)addrh << 32) | (uint64_t)addrl;
        match = (type == 3) && (rank == target_rank) && ((tag == target_tag) || (tag == TAG_ANY)) && (addr == target_addr);
        if(!match){
            //copy from STS_RNDZV to STS_RNDZV_PENDING
            putd(CMD_RNDZV_PENDING, type);
            putd(CMD_RNDZV_PENDING, rank);
            putd(CMD_RNDZV_PENDING, tag);
            putd(CMD_RNDZV_PENDING, addrl);
            putd(CMD_RNDZV_PENDING, addrh);
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            putd(CMD_RNDZV_PENDING, getd(STS_RNDZV));
            num_rndzv_pending++;
        } else {
            getd(STS_RNDZV);//TODO check host
            getd(STS_RNDZV);//count
            return NO_ERROR;
        }
    }
    //we found nothing, signal this up to caller
    return NOT_READY_ERROR;
}

//receives an acknowledgement that a RDMA WRITE to 
//our buffer has completed
int rendezvous_get_any_completion(unsigned int *target_rank, uint64_t *target_addr, bool *target_host, uint32_t target_count, uint32_t target_tag){
    uint32_t type, rank, tag, count, host;
    uint64_t addr;
    uint64_t addrl, addrh;
    bool match;
    // Check if there are pending notifications in STS_RNDZV_PENDING,
    // otherwise pull from STS_RNDZV
    unsigned int i;
    for(i = 0; i<num_rndzv_pending; i++){
        type = getd(STS_RNDZV_PENDING);
        rank = getd(STS_RNDZV_PENDING);
        tag = getd(STS_RNDZV_PENDING);
        addrl = getd(STS_RNDZV_PENDING);
        addrh = getd(STS_RNDZV_PENDING);
        addr = ((uint64_t)addrh << 32) | (uint64_t)addrl;
        host = getd(STS_RNDZV_PENDING);
        count = getd(STS_RNDZV_PENDING);
        match = (type == 3) && ((tag == target_tag) || (tag == TAG_ANY)) && (count == target_count);
        if(!match){
            //copy from STS_RNDZV to STS_RNDZV_PENDING
            putd(CMD_RNDZV_PENDING, type);
            putd(CMD_RNDZV_PENDING, rank);
            putd(CMD_RNDZV_PENDING, tag);
            putd(CMD_RNDZV_PENDING, addrl);
            putd(CMD_RNDZV_PENDING, addrh);
            putd(CMD_RNDZV_PENDING, host);
            putd(CMD_RNDZV_PENDING, count);
            putd(CMD_RNDZV_PENDING, host);
            putd(CMD_RNDZV_PENDING, count);
        } else {
            num_rndzv_pending--;
            *target_rank = rank;
            *target_addr = addr;
            *target_host = host;
            return NO_ERROR;
        }
    }
    //if we're still executing, no match in RNDZV_PENDING, check STS_RNDZV
    while(!tngetd(STS_RNDZV)){
        type = getd(STS_RNDZV);
        rank = getd(STS_RNDZV);
        tag = getd(STS_RNDZV);
        addrl = getd(STS_RNDZV);
        addrh = getd(STS_RNDZV);
        addr = ((uint64_t)addrh << 32) | (uint64_t)addrl;
        host = getd(STS_RNDZV);
        count = getd(STS_RNDZV);
        match = (type == 3) && ((tag == target_tag) || (tag == TAG_ANY)) && (count == target_count);
        if(!match){
            //copy from STS_RNDZV to STS_RNDZV_PENDING
            putd(CMD_RNDZV_PENDING, type);
            putd(CMD_RNDZV_PENDING, rank);
            putd(CMD_RNDZV_PENDING, tag);
            putd(CMD_RNDZV_PENDING, addrl);
            putd(CMD_RNDZV_PENDING, addrh);
            putd(CMD_RNDZV_PENDING, host);
            putd(CMD_RNDZV_PENDING, count);
            num_rndzv_pending++;
        } else {
            *target_rank = rank;
            *target_addr = addr;
            *target_host = host;
            return NO_ERROR;
        }
    }
    //we found nothing, signal this up to caller
    return NOT_READY_ERROR;
}

//configure datapath before calling this method
//instructs the data plane to move data
//use MOVE_IMMEDIATE
void start_move(
    uint32_t op0_opcode,
    uint32_t op1_opcode,
    uint32_t res_opcode,
    uint32_t flags,
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
    
    uint32_t compression_flags = flags & 0xff;
    uint32_t remote_flags = (flags>>8) & 0xff;
    uint32_t host_flags = (flags>>16) & 0xff;
    uint32_t rendezvous_flag = (flags>>24) & 0xff;

    opcode |= remote_flags << 9;
    bool res_is_remote = (remote_flags == RES_REMOTE);
    opcode |= (compression_flags & 0x7) << 10;//mask is to prevent ETH_COMPRESSED flag leaking into func_id
    opcode |= func_id << 13;
    opcode |= host_flags << 17;
    opcode |= rendezvous_flag << 20;
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
    uint32_t flags,
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
        flags,
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
                        unsigned int buftype) {
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    return move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE,
        MOVE_NONE,
        (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE,
        pack_flags(compression, RES_LOCAL, host), 0,
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
            unsigned int buftype) {
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    return move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE,
        MOVE_IMMEDIATE,
        (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE,
        pack_flags(compression, RES_LOCAL, host), function,
        count, 0, arcfg_offset, op0_addr, op1_addr, res_addr, 0, 0, 0,
        0, 0, 0, 0
    );
}

//transmits a buffer to a rank of the world communicator
//TODO: non-streaming send to self
int send(
    unsigned int dst_rank,
    unsigned int count,
    uint64_t src_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset,
    unsigned int dst_tag,
    unsigned int compression,
    unsigned int buftype
) {
    unsigned int host = (buftype >> 8) & 0xff;
    unsigned int stream = buftype & 0xff;
    //get count in bytes
    unsigned int bytes_count = datatype_nbytes*count;
    if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        //Rendezvous without segmentation
        //get remote address
        uint64_t dst_addr;
        bool dst_host;
        //no need to use current_step here, because there is only one place where we can retry
        int status = rendezvous_get_addr(dst_rank, &dst_addr, &dst_host, count, dst_tag);
        if(status == NOT_READY_ERROR){
            //we're not yet ready to serve this send, queue it for retry
            return NOT_READY_ERROR;
        }
        if(dst_host){
            host |= RES_HOST;
        }
        //do a RDMA write to the remote address 
        return move(
            MOVE_IMMEDIATE,
            MOVE_NONE,
            MOVE_IMMEDIATE,
            pack_flags_rendezvous(host),
            0, count, comm_offset, arcfg_offset, 
            src_addr, 0, dst_addr, 0, 0, 0,
            0, 0, dst_rank, dst_tag
        );
    } else {
        //Eager with segmentation
        //if ETH_COMPRESSED is set, also set RES_COMPRESSED
        compression |= (compression & ETH_COMPRESSED) >> 1;
        //TODO: when doing a one-sided send to a remote stream, check:
        //dst_tag is < 247 (for correct routing on remote side)
        //destination compression == Ethernet compression
        //(since data can't be decompressed on remote side)
        //calculate max segment size in elements, from element size 
        unsigned int max_seg_count = eager_rx_buf_size / datatype_nbytes;
        //calculate number of segments required for this send
        unsigned int nseg = (count+max_seg_count-1)/max_seg_count;
        int i;
        unsigned int expected_ack_count = 0;
        unsigned int ret = NO_ERROR;
        for(i=0; i<nseg; i++){
            start_move(
                (stream & OP0_STREAM) ? MOVE_STREAM : ((i==0) ? MOVE_IMMEDIATE : MOVE_STRIDE),
                MOVE_NONE,
                (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE,
                pack_flags(compression, (dst_rank == world.local_rank) ? RES_LOCAL : RES_REMOTE, host),
                0, (i!=(nseg-1)) ? max_seg_count : (count-i*max_seg_count), 
                comm_offset, arcfg_offset, 
                src_addr, 0, 0, 
                max_seg_count, 0, 0,
                0, 0, dst_rank, dst_tag
            );
            expected_ack_count++;
            if(expected_ack_count > 2){
                ret |= end_move();
                expected_ack_count--;
            }
        }
        for(i=0; i<expected_ack_count; i++){
            ret |= end_move();
        }
        return ret;
    }
}

//waits for a messages to come and move their contents in a buffer
//use MOVE_ON_RECV
int recv(
    unsigned int src_rank,
    unsigned int count,
    uint64_t dst_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset,
    unsigned int src_tag,
    unsigned int compression,
    unsigned int buftype
) {
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    //get count in bytes
    unsigned int bytes_count = datatype_nbytes*count;
    if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        //rendezvous without segmentation
        bool is_host = ((host & RES_HOST) != 0);
        //send address only if we haven't sent it before
        if(current_step == 0){
            rendezvous_send_addr(src_rank, dst_addr, is_host, count, src_tag);
            current_step++;
        }
        return rendezvous_get_completion(src_rank, dst_addr, is_host, count, src_tag);
    } else {
        //Eager with segmentation
        //if ETH_COMPRESSED is set, also set OP1_COMPRESSED
        compression |= (compression & ETH_COMPRESSED) >> 2;
        //calculate max segment size in elements, from element size 
        unsigned int max_seg_count = eager_rx_buf_size / datatype_nbytes;
        //calculate number of segments required for this send
        unsigned int nseg = (count+max_seg_count-1)/max_seg_count;
        int i;
        unsigned int expected_ack_count = 0;
        unsigned int ret = NO_ERROR;
        for(i=0; i<nseg; i++){
            start_move(
                MOVE_NONE,
                MOVE_ON_RECV,
                (stream & RES_STREAM) ? MOVE_STREAM : ((i==0) ? MOVE_IMMEDIATE : MOVE_STRIDE),
                pack_flags(compression, RES_LOCAL, host), 
                0, (i!=(nseg-1)) ? max_seg_count : (count-i*max_seg_count), 
                comm_offset, arcfg_offset, 
                0, 0, dst_addr, 
                0, 0, max_seg_count,
                src_rank, src_tag, 0, (stream & RES_STREAM) ? src_tag : 0
            );
            expected_ack_count++;
            if(expected_ack_count > 2){
                ret |= end_move();
                expected_ack_count--;
            }
        }
        for(i=0; i<expected_ack_count; i++){
            ret |= end_move();
        }
        return ret;
    }
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
        unsigned int buftype
){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    unsigned int err = NO_ERROR;

    //figure out compression
    //if ETH_COMPRESSED flag is set, then we need to set OP1_COMPRESSED (for the recv)
    //keep OP0_COMPRESSED and RES_COMPRESSED as-is
    compression = compression | ((compression & ETH_COMPRESSED) >> 2);

    err |= move(
        (stream & OP0_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE,
        MOVE_ON_RECV,
        (stream & RES_STREAM) ? MOVE_STREAM : MOVE_IMMEDIATE,
        pack_flags(compression, RES_LOCAL, host),
        func, count,
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
        unsigned int buftype
){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
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
        pack_flags(compression, RES_REMOTE, host),
        func, count,
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
                unsigned int root_rank,
                uint64_t buf_addr,
                unsigned int comm_offset,
                unsigned int arcfg_offset,
                unsigned int compression,
                unsigned int buftype){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    int err = NO_ERROR;

    unsigned int bytes_count = datatype_nbytes*count;
    if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        uint64_t dst_addr;
        uint32_t dst_rank;
        bool dst_host = ((host & RES_HOST) != 0);
        bool src_host = ((host & OP0_HOST) != 0);

        if(world.size > Xil_In32(BCAST_FLAT_TREE_MAX_RANKS_OFFSET)){
            //binary tree broadcast
            //we double the number of broadcasting nodes in each round until we've broadcast to every rank
            //starting broadcast is from root to the rank with the index equal to the largest power of 2
            //which is strictly smaller than worldsize: 2^floor(log2(worldsize-1))
            //D0(x) = 2^floor(log2(x-1))
            //examples: D0(3) = 2^floor(log2(2)) = 2; D0(4) = 2^floor(log2(3)) = 2; D0(5) = 2^floor(log2(4)) = 4
            //
            //from the sender's point of view, we send in rounds, whereby in each round
            //    we send to a rank at distance D from ourselves
            //    from ranks of index divisible by 2D
            //    we halve the distance
            //from the receiver's point of view:
            //    at each round if our local rank index is divisible by D, receive from distance -D (if that is >= 0)
            //    if we've received once, we convert to sender for remaining rounds
            //
            //an example for worldsize = 9
            //round 0: D = D0(9) = 8  where 16 divides local rank index: 0 -> 8
            //round 1: D = 4          where  8 divides local rank index: 0 -> 4, 8 -> x
            //round 2: D = 2          where  4 divides local rank index: 0 -> 2, 4 -> 6, 8 -> x
            //round 3: D = 1          where  2 divides local rank index: 0 -> 1, 2 -> 3, 4 -> 5, 6 -> 7, 8 -> x
            //round 4: D = 0 - DONE!
            
            unsigned int d = 1<<fast_log2(world.size-1);
            bool sender = (root_rank == world.local_rank);

            //calculate normalized rank l, where root is index 0
            unsigned int l = (world.local_rank + world.size - root_rank) % world.size;
            
            while(d > 0){
                if(sender && (l % (2*d) == 0) && (l+d < world.size)){
                    unsigned int receiver_rank = (l + d + root_rank) % world.size;
                    while(rendezvous_get_addr(receiver_rank, &dst_addr, &dst_host, count, TAG_ANY) == NOT_READY_ERROR);\
                    start_move(
                        MOVE_IMMEDIATE,
                        MOVE_NONE,
                        MOVE_IMMEDIATE,
                        pack_flags_rendezvous((src_host << 2) | dst_host),
                        0, count, comm_offset, arcfg_offset, 
                        buf_addr, 0, dst_addr, 0, 0, 0,
                        0, 0, receiver_rank, TAG_ANY
                    );
                    err |= end_move();
                } else if(!sender && ((l % d) == 0) && ((l-d) >= 0)){
                    unsigned int sender_rank = (l - d + root_rank + world.size) % world.size;
                    rendezvous_send_addr(sender_rank, buf_addr, dst_host, count, TAG_ANY);
                    while(rendezvous_get_completion(sender_rank, buf_addr, dst_host, count, TAG_ANY) == NOT_READY_ERROR);
                    //we now become a sender
                    sender = true;
                    src_host = dst_host;
                }
                d >>= 1; //halve d
            }
            return err;
        } else {
            //out-of-order high-fanout implementation
            //better tolerance for rank skews, preferable for small world sizes
            if(root_rank == world.local_rank){
                //get remote address
                unsigned int pending_moves = 0;
                //if root, wait for addresses then send
                while(current_step < (world.size-1)){
                    int status = rendezvous_get_any_addr(&dst_rank, &dst_addr, &dst_host, count, TAG_ANY);
                    if(status == NOT_READY_ERROR){
                        //we're not yet ready to serve this send, queue it for retry after flushing moves
                        while(pending_moves > 0){
                            err |= end_move();
                            pending_moves--;
                        }
                        //if err was NO_ERROR, we'll retry, otherwise give up
                        return (err | status);
                    }
                    if(dst_host){
                        host |= RES_HOST;
                    }
                    //do a RDMA write to the remote address 
                    start_move(
                        MOVE_IMMEDIATE,
                        MOVE_NONE,
                        MOVE_IMMEDIATE,
                        pack_flags_rendezvous(host),
                        0, count, comm_offset, arcfg_offset, 
                        buf_addr, 0, dst_addr, 0, 0, 0,
                        0, 0, dst_rank, TAG_ANY
                    );
                    pending_moves++;
                    current_step++;
                    if(pending_moves > 2){
                        err |= end_move();
                        pending_moves--;
                    }
                }
                while(pending_moves > 0){
                    err |= end_move();
                    pending_moves--;
                }
                return err;
            } else {
                //if not root, send address then wait completion
                bool res_is_host = ((host & RES_HOST) != 0);
                if(current_step == 0){
                    rendezvous_send_addr(root_rank, buf_addr, res_is_host, count, TAG_ANY);
                    current_step++;
                }
                return rendezvous_get_completion(root_rank, buf_addr, res_is_host, count, TAG_ANY);
            }
        }
    } else {
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
            max_seg_count = eager_rx_buf_size / datatype_nbytes;
        }

        int expected_ack_count = 0;
        while(elems_remaining > 0){
            //determine if we're sending or receiving
            if(root_rank == world.local_rank){
                //on the root we only care about ETH_COMPRESSED and OP0_COMPRESSED
                //so replace RES_COMPRESSED with ETH_COMPRESSED
                compression = compression | ((compression & ETH_COMPRESSED) >> 1);

                //send a segment to each of the ranks (excluding self)
                for(int i=0; i < world.size; i++){
                    start_move(
                        (i==0) ? ((elems_remaining == count) ? MOVE_IMMEDIATE : MOVE_INCREMENT) : MOVE_REPEAT,
                        MOVE_NONE,
                        MOVE_IMMEDIATE,
                        pack_flags(compression, RES_REMOTE, host & OP0_HOST),
                        0, 
                        (i == root_rank) ? 0 : min(max_seg_count, elems_remaining),
                        comm_offset, arcfg_offset,
                        buf_addr, 0, 0, 0, 0, 0,
                        0, 0, i, TAG_ANY
                    );
                    expected_ack_count++;
                    //start flushing out ACKs so our pipes don't fill up
                    if(expected_ack_count > 8){
                        err |= end_move();
                        expected_ack_count--;
                    }
                }
            } else{
                //on non-root nodes we only care about ETH_COMPRESSED and RES_COMPRESSED
                //so replace OP1_COMPRESSED with the value of ETH_COMPRESSED
                compression = compression | ((compression & ETH_COMPRESSED) >> 2);
                err |= move(
                    MOVE_NONE,
                    MOVE_ON_RECV,
                    (elems_remaining == count) ? MOVE_IMMEDIATE : MOVE_INCREMENT,
                    pack_flags(compression, RES_LOCAL, host & RES_HOST),
                    0,
                    min(max_seg_count, elems_remaining),
                    comm_offset, arcfg_offset,
                    0, 0, buf_addr, 0, 0, 0,
                    root_rank, TAG_ANY, 0, 0
                );
            }
            elems_remaining -= max_seg_count;
        }
        //flush remaining ACKs 
        for(int i=0; i < expected_ack_count; i++){
            err |= end_move();
        }
        return err;
    }
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
            unsigned int buftype){

    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    //TODO: implement segmentation
    //TODO: scattering from/to stream

    int err = NO_ERROR;

    unsigned int bytes_count = datatype_nbytes*count;
    if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        if(src_rank == world.local_rank){
            //get remote address
            uint64_t dst_addr;
            uint64_t buf_addr;
            uint32_t dst_rank;
            bool dst_host;
            unsigned int pending_moves = 0;
            //first copy to ourselves while we're waiting for addresses
            if(current_step == 0){
                buf_addr = src_buf_addr + src_rank*bytes_count;
                start_move(
                    MOVE_IMMEDIATE,
                    MOVE_NONE,
                    MOVE_IMMEDIATE,
                    pack_flags(compression, RES_LOCAL, host),
                    0, count, comm_offset, arcfg_offset, 
                    buf_addr, 0, dst_buf_addr, 0, 0, 0,
                    0, 0, 0, TAG_ANY
                );
                current_step++;
                pending_moves++;
            }
            //if root, wait for addresses then send
            while(current_step < world.size){
                int status = rendezvous_get_any_addr(&dst_rank, &dst_addr, &dst_host, count, TAG_ANY);
                if(status == NOT_READY_ERROR){
                    //we're not yet ready to serve this send, queue it for retry after flushing moves
                    while(pending_moves > 0){
                        err |= end_move();
                        pending_moves--;
                    }
                    //if err was NO_ERROR, we'll retry, otherwise give up
                    return (err | status);
                }
                if(dst_host){
                    host |= RES_HOST;
                }
                //compute address (striding doesnt work here because order of sends is random)
                buf_addr = src_buf_addr + dst_rank*bytes_count;
                //do a RDMA write to the remote address 
                start_move(
                    MOVE_IMMEDIATE,
                    MOVE_NONE,
                    MOVE_IMMEDIATE,
                    pack_flags_rendezvous(host),
                    0, count, comm_offset, arcfg_offset, 
                    buf_addr, 0, dst_addr, 0, 0, 0,
                    0, 0, dst_rank, TAG_ANY
                );
                pending_moves++;
                current_step++;
                if(pending_moves > 2){
                    err |= end_move();
                    pending_moves--;
                }
            }
            while(pending_moves > 0){
                err |= end_move();
                pending_moves--;
            }
            return err;
        } else {
            //if not root, send address then wait completion
            bool res_is_host = ((host & RES_HOST) != 0);
            if(current_step == 0){
                rendezvous_send_addr(src_rank, dst_buf_addr, res_is_host, count, TAG_ANY);
                current_step++;
            }
            return rendezvous_get_completion(src_rank, dst_buf_addr, res_is_host, count, TAG_ANY);
        }
    } else {
        //determine if we're sending or receiving
        if(src_rank == world.local_rank){
            //on the root we only care about ETH_COMPRESSED and OP0_COMPRESSED
            //so replace RES_COMPRESSED with ETH_COMPRESSED
            //unless we're copying to ourselves in case we're keeping flags as-is
            unsigned int compression_snd = (compression & OP0_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 1);

            for(int i=0; i < world.size; i++){
                start_move(
                    (i==0) ? MOVE_IMMEDIATE : MOVE_INCREMENT,
                    MOVE_NONE,
                    MOVE_IMMEDIATE,
                    pack_flags((i==src_rank) ? compression : compression_snd, (i==src_rank) ? RES_LOCAL : RES_REMOTE, host & (OP0_HOST | RES_HOST)),
                    0,
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
                pack_flags(compression, RES_LOCAL, host & RES_HOST),
                0,
                count,
                comm_offset, arcfg_offset,
                0, 0, dst_buf_addr, 0, 0, 0,
                src_rank, TAG_ANY, 0, 0
            );
        }

        return err;
    }
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
            unsigned int buftype){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    unsigned int i, curr_pos, next_in_ring, prev_in_ring, number_of_shift;
    int err = NO_ERROR;

    unsigned int bytes_count = datatype_nbytes*count;
    if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        bool dst_host;
        int status;
        uint64_t buf_addr;
        if(root_rank == world.local_rank){
            dst_host = (host & RES_HOST) != 0;
            //start copy to ourselves while we're sending addresses
            buf_addr = dst_buf_addr + root_rank*bytes_count;
            start_move(
                MOVE_IMMEDIATE,
                MOVE_NONE,
                MOVE_IMMEDIATE,
                pack_flags(compression, RES_LOCAL, host),
                0, count, comm_offset, arcfg_offset, 
                src_buf_addr, 0, buf_addr, 0, 0, 0,
                0, 0, 0, TAG_ANY
            );
            //send addresses
            unsigned int num_pending = 0, num_initiated = 0, num_completed = 0, src_rank=0;
            unsigned int max_fanin = (bytes_count > Xil_In32(GATHER_FLAT_TREE_MAX_COUNT_OFFSET)) ? Xil_In32(GATHER_FLAT_TREE_MAX_FANIN_OFFSET) : world.size-1;
            while(num_completed < world.size-1){
                if(num_initiated < world.size-1){
                    if(src_rank == world.local_rank){
                        src_rank++;
                    } else while(num_pending < max_fanin) {
                        buf_addr = dst_buf_addr + src_rank*bytes_count;
                        rendezvous_send_addr(src_rank, buf_addr, dst_host, count, TAG_ANY);
                        num_initiated++;
                        num_pending++;
                        src_rank++;      
                    }
                }
                if(num_pending > 0){
                    unsigned int dummy_rank;
                    bool dummy_host;
                    while(rendezvous_get_any_completion(&dummy_rank, &buf_addr, &dummy_host, bytes_count, TAG_ANY) == NOT_READY_ERROR);
                    num_pending--;
                    num_completed++;
                }
            }
            //finalize copy to ourselves
            err |= end_move();
            return err;
        } else {
            status = rendezvous_get_addr(root_rank, &buf_addr, &dst_host, count, TAG_ANY);
            if(status == NOT_READY_ERROR){
                //we're not yet ready to serve this send, queue it for retry
                return NOT_READY_ERROR;
            }
            if(dst_host){
                host |= RES_HOST;
            }
            //do a RDMA write to the remote address 
            return move(
                MOVE_IMMEDIATE,
                MOVE_NONE,
                MOVE_IMMEDIATE,
                pack_flags_rendezvous(host),
                0, count, comm_offset, arcfg_offset, 
                src_buf_addr, 0, buf_addr, 0, 0, 0,
                0, 0, root_rank, TAG_ANY
            );
        }
    } else {
        //ring-based eager gather 
        next_in_ring = (world.local_rank + 1) % world.size;
        prev_in_ring = (world.local_rank + world.size - 1) % world.size;

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
                pack_flags(NO_COMPRESSION, RES_LOCAL, NO_HOST),
                0,
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
                    pack_flags((i==0) ? copy_compression : recv_compression, RES_LOCAL, host),
                    0,
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
                pack_flags(send_compression, RES_REMOTE, host),
                0,
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
                    pack_flags(relay_compression, RES_REMOTE, NO_HOST),
                    0,
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
}

//naive fused: 1) receive a segment 2) move in the dest buffer 3) relay to next rank
int allgather(
    unsigned int count,
    uint64_t src_buf_addr,
    uint64_t dst_buf_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset,
    unsigned int compression,
    unsigned int buftype
){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    int i;
    int err = NO_ERROR;
    int next_in_ring = (world.local_rank + 1) % world.size;
    int prev_in_ring = (world.local_rank + world.size - 1) % world.size;

    unsigned int bytes_count = datatype_nbytes*count;
    if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        //rendezvous ring allgather with P segments
        //TODO: each rank pulls from next_in_ring into a segment, pushes a segment to prev_in_ring
        //we do this for P-1 steps (such that each rank gathers the P-1 segments it's missing)
        //at each step: send a addr notif to neighbour and wait for an address notif from neighbour + transfer
        bool dst_buf_host = (host & RES_HOST) != 0;
        bool downstream_host;
        uint64_t downstream_addr, upstream_addr;
        int status;
        //start copy to ourselves while we're sending/receiving addresses
        if(current_step == 0){
            start_move(
                MOVE_IMMEDIATE,
                MOVE_NONE,
                MOVE_IMMEDIATE,
                pack_flags(compression, RES_LOCAL, host),
                0, count, comm_offset, arcfg_offset, 
                src_buf_addr, 0, dst_buf_addr + world.local_rank*bytes_count,
                0, 0, 0,
                0, 0, 0, TAG_ANY
            );
            //we're sending the address to prev_in_ring, who will write to us 
            rendezvous_send_addr(prev_in_ring, dst_buf_addr, dst_buf_host, count, TAG_ANY);
            current_step++;
        }
        //get address of buffer in next_in_ring - where we'll write
        //this step has retries
        if(current_step == 1){
            status = rendezvous_get_addr(next_in_ring, &downstream_addr, &downstream_host, count, TAG_ANY);
            if(status == NOT_READY_ERROR){
                //if err was NO_ERROR, we'll retry, otherwise give up
                return status;
            } else {
                err |= end_move();
                current_step++;
            }
        }
        //wait-and-forward loop
        int pending_moves = 0;
        //forward own segment downstream first
        if(downstream_host){
            host |= RES_HOST;
        }
        if(dst_buf_host){
            host |= OP0_HOST;
        }
        start_move(
            MOVE_IMMEDIATE,
            MOVE_NONE,
            MOVE_IMMEDIATE,
            pack_flags_rendezvous(host),
            0, count, comm_offset, arcfg_offset, 
            dst_buf_addr + world.local_rank*bytes_count, 0, downstream_addr + world.local_rank*bytes_count,
            0, 0, 0,
            0, 0, next_in_ring, TAG_ANY
        );
        pending_moves++;
        i=prev_in_ring;
        while(i != world.local_rank){
            //get completion for a segment 
            upstream_addr = dst_buf_addr + i*bytes_count;
            while(rendezvous_get_completion(prev_in_ring, upstream_addr, dst_buf_host, count, TAG_ANY) == NOT_READY_ERROR);
            if(i != next_in_ring) {
                //send from the received segment
                start_move(
                    MOVE_IMMEDIATE,
                    MOVE_NONE,
                    MOVE_IMMEDIATE,
                    pack_flags_rendezvous(host),
                    0, count, comm_offset, arcfg_offset, 
                    upstream_addr, 0, downstream_addr + i*bytes_count,
                    0, 0, 0,
                    0, 0, next_in_ring, TAG_ANY
                );
                pending_moves++;
                if(pending_moves > 2){
                    err |= end_move();
                    pending_moves--;
                }
                current_step++;
            }
            i = (i + world.size - 1) % world.size;
        }
        //flush move statuses
        while(pending_moves > 0){
            err |= end_move();
            pending_moves--;
        }
    } else {
        int curr_pos, rel_stride, abs_stride;

        //compression is tricky for the relay: we've already received into the destination buffer
        //with associated flag RES_COMPRESSED; this buffer becomes the source for a send
        //so if RES_COMPRESSED is set, OP0_COMPRESSED must be set for the send, and RES_COMPRESSED reset
        unsigned int relay_compression = (compression & RES_COMPRESSED) ? (compression | OP0_COMPRESSED) : compression;
        relay_compression &= ~(RES_COMPRESSED);

        //prime the address slot for the destination, so we can subsequently stride against it
        start_move(
            MOVE_NONE, MOVE_NONE, MOVE_IMMEDIATE,
            pack_flags(NO_COMPRESSION, RES_LOCAL, NO_HOST),
            0,
            0,
            0, arcfg_offset,
            src_buf_addr, 0, dst_buf_addr, 0, 0, 0,
            0, 0, 0, 0
        );

        //copy our local data into the appropriate destination slot
        start_move(
            MOVE_IMMEDIATE, MOVE_NONE, MOVE_STRIDE,
            pack_flags(compression, RES_LOCAL, host),
            0,
            count,
            0, arcfg_offset,
            src_buf_addr, 0, 0, 0, 0, count*world.local_rank,
            0, 0, 0, 0
        );

        //send to next in ring
        //ETH_COMPRESSED flag overwrites RES_COMPRESSED
        start_move(
            MOVE_IMMEDIATE, MOVE_NONE, MOVE_IMMEDIATE,
            pack_flags(compression | ((compression & ETH_COMPRESSED) >> 1), RES_REMOTE, host),
            0,
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
                pack_flags(compression | ((compression & ETH_COMPRESSED) >> 2), RES_LOCAL, host & RES_HOST),
                0,
                count,
                comm_offset, arcfg_offset,
                0, 0, 0, 0, 0, rel_stride,
                prev_in_ring, TAG_ANY, 0, 0
            );

            if(i < world.size-2){ //if not the last data, relay to the next in ring
                //first prime the address
                start_move(
                    MOVE_NONE, MOVE_IMMEDIATE, MOVE_NONE,
                    pack_flags(NO_COMPRESSION, RES_REMOTE, NO_HOST),
                    0,
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
                    pack_flags(((compression & RES_COMPRESSED) >> 2) | ((compression & ETH_COMPRESSED) >> 1), RES_REMOTE, host),
                    0,
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
            unsigned int buftype){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    unsigned int bytes_count = datatype_nbytes*count;

    if(world.size == 1){
        //corner-case copy for when running a single-node reduction
        return copy(count, src_addr, dst_addr, arcfg_offset, compression, buftype);
    }else if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        //get scratchpad buffers from exchmem
        uint64_t tmp1_addr = ((uint64_t)Xil_In32(TMP1_OFFSET+4) << 32) | (uint64_t)Xil_In32(TMP1_OFFSET);
        uint64_t tmp2_addr = ((uint64_t)Xil_In32(TMP2_OFFSET+4) << 32) | (uint64_t)Xil_In32(TMP2_OFFSET);
        uint64_t tmp3_addr = ((uint64_t)Xil_In32(TMP3_OFFSET+4) << 32) | (uint64_t)Xil_In32(TMP3_OFFSET);
        //rendezvous tree reduce using tmp_addr as a scratchpad (preferably in device memory)
        unsigned int err = NO_ERROR;

        if(world.size <= Xil_In32(REDUCE_FLAT_TREE_MAX_RANKS_OFFSET) || bytes_count <= Xil_In32(REDUCE_FLAT_TREE_MAX_COUNT_OFFSET)){
            //flat tree reduce:
            //useful for small messages and small communicators, where the extra hops for binary tree are detrimental
            //we effectively flat-tree gather all data from non-root into one of the rendezvous spare buffers
            //we sum up the data as it comes in (assuming commutativity, user beware)
            bool dst_host;
            bool src_buf_host = (host & OP0_HOST) != 0;
            bool dst_buf_host = (host & RES_HOST) != 0;
            int status;
            uint64_t buf_addr;
            if(root_rank == world.local_rank){
                uint32_t src_rank;
                //send addresses
                for(src_rank=0; src_rank < world.size; src_rank++){
                    //skip sending address for ourselves
                    if(src_rank == world.local_rank){
                        continue;
                    }
                    //compute address into the first rendezvous spare buffer
                    buf_addr = tmp1_addr + src_rank*bytes_count;
                    rendezvous_send_addr(src_rank, buf_addr, false, count, TAG_ANY);
                }
                //wait for completions out-of-order
                int i, pending_moves=0;
                for(i=0; i < world.size-1; i++){
                    while(rendezvous_get_any_completion(&src_rank, &buf_addr, &dst_host, bytes_count, TAG_ANY) == NOT_READY_ERROR);
                    //start a reduction into either tmp2 or tmp3
                    uint64_t accum_addr = (i % 2) ? tmp2_addr : tmp3_addr;
                    uint64_t prev_accum_addr = (i % 2) ? tmp3_addr : tmp2_addr;
                    //manipulate host flags such that on first iteration we set OP0_HOST
                    //if src_addr is in host memory, and on last iteration we set RES_HOST
                    //if dst_addr is in host memory
                    host = ((i==0) && src_buf_host) | (((i==world.size-2) && dst_buf_host) << 2);
                    start_move(
                        MOVE_IMMEDIATE,
                        MOVE_IMMEDIATE,
                        MOVE_IMMEDIATE,
                        pack_flags(NO_COMPRESSION, RES_LOCAL, host),
                        func, count,
                        comm_offset, arcfg_offset,
                        (i==0) ? src_addr : prev_accum_addr, buf_addr, (i==world.size-2) ? dst_addr : accum_addr, 0, 0, 0,
                        0, 0, 0, 0
                    );
                    pending_moves++;
                    if(pending_moves > 2){
                        err |= end_move();
                        pending_moves--;
                    }
                }
                while(pending_moves > 0){
                    err |= end_move();
                    pending_moves--;
                }
                return err;
            } else {
                while(rendezvous_get_addr(root_rank, &buf_addr, &dst_host, count, TAG_ANY) == NOT_READY_ERROR);
                if(dst_host){
                    host |= RES_HOST;
                }
                //do a RDMA write to the remote address 
                return move(
                    MOVE_IMMEDIATE,
                    MOVE_NONE,
                    MOVE_IMMEDIATE,
                    pack_flags_rendezvous(host),
                    0, count, comm_offset, arcfg_offset, 
                    src_addr, 0, buf_addr, 0, 0, 0,
                    0, 0, root_rank, TAG_ANY
                );
            }
        } else {
            //binary tree reduce:
            //at each step, each rank can do one or more of 3 actions: send (S), receive (R), and sum (+)
            //for rendezvous, R involves an address send and completion check and does not involve the DMP
            //for rendezvous, S involves an address get and a DMP move
            //+ involves a DMP move from src_addr/dst_addr and tmp_addr to dst_addr
            //send and sum can be fused into a single action denoted +S
            //
            //we denote N as the step number
            //the action distance D = 2^N is the distance (in ranks) over which we communicate
            //L is the normalized local rank, obtained by subtracting root rank number from local rank, mod size
            //at each step, we calculate whether we perform any of the actions:
            //S(N,L) = (L%2D != 0) && (L != 0) = (L%(1<<(N+1)) != 0) && (L != 0)
            //R(N,L) = !S(N,L) && ((L+D)<size) = !S(N,L) && ((L+1<<N)<size)
            //+(N,L) = R(N-1,L)
            //
            //we have available two scratchpad buffers in device memory and the dst buffer in host or device memory
            //at each time step we optionally receive into one scratchpad, alternating
            //we accumulate in the dst buffer from previously received data
            //
            //we make sure that we operate from src_addr instead of dst_addr the very first time
            //
            //execution completed on root if !R or on non-root if S
            //
            //note: the implementation is more complex because we want to be stateless
            //everything should be computed from the current step such that we can swap the reduce in and out of execution
            unsigned int l = (world.local_rank + world.size - root_rank) % world.size;
            unsigned int d, n = 0;
            bool s = false;
            bool r = false;
            bool plus = false;
            bool is_root = (l==0);
            uint64_t remote_addr;
            bool remote_host;
            unsigned int receiving_rank, sending_rank;
            unsigned int scratchpad_sel;
            uint64_t current_accumulator_addr;
            uint64_t previous_accumulator_addr; //TODO: implement without saving from current
            uint64_t current_recv_addr;
            uint64_t previous_recv_addr; //TODO: implement without saving from current
            bool first_move = true; //TODO: implement without this state bit
            bool last_move = false; //TODO: implement without this state bit
            while(1){
                d = 1<<n;
                s = (l%(2*d) != 0) && !is_root;
                r = !s && ((l+d)<world.size);
                plus = !((l%d != 0) && !is_root) && ((l+d/2)<world.size) && (n>0);
                scratchpad_sel = (n % 3);
                current_recv_addr = (scratchpad_sel == 0) ? tmp1_addr : (scratchpad_sel == 1) ? tmp2_addr : tmp3_addr;
                current_accumulator_addr = (scratchpad_sel == 0) ? tmp2_addr : (scratchpad_sel == 1) ? tmp3_addr : tmp1_addr;
                receiving_rank = (l+world.size-d+root_rank)%world.size;
                sending_rank = (l+d+root_rank)%world.size;
                last_move = is_root && (d > world.size);//last move if the next-largest distance is larger than world size
                //TODO instead of just-in-time distributing addresses, we could do one run through this loop ahead
                //of time just for address distribution, then do address resolution just-in-time
                if(r){
                    rendezvous_send_addr(sending_rank, current_recv_addr, false, count, TAG_ANY);
                }
                //address resolution in case we need to send
                if(s){
                    while(rendezvous_get_addr(receiving_rank, &remote_addr, &remote_host, count, TAG_ANY) == NOT_READY_ERROR);
                    //adjust host flags and addresses in case we're sending from dst and scratchpad
                    if(n > 1){
                        if((host & RES_HOST) != 1){
                            host |= OP0_HOST;
                        }
                        if((n > 1) && remote_host){
                            host |= RES_HOST;
                        }
                    }
                }
                //DMP ops for sending/combining
                if(s && plus){
                    //fused combine+send
                    err |= move(
                        MOVE_IMMEDIATE,
                        MOVE_IMMEDIATE,
                        MOVE_IMMEDIATE,
                        pack_flags_rendezvous(host),
                        func, count,
                        comm_offset, arcfg_offset,
                        first_move ? src_addr : previous_accumulator_addr, previous_recv_addr, remote_addr, 0, 0, 0,
                        0, 0, receiving_rank, TAG_ANY
                    );
                } else if(s) {
                    //send
                    err |= move(
                        MOVE_IMMEDIATE,
                        MOVE_NONE,
                        MOVE_IMMEDIATE,
                        pack_flags_rendezvous(host),
                        func, count,
                        comm_offset, arcfg_offset,
                        first_move ? src_addr : previous_accumulator_addr, 0, remote_addr, 0, 0, 0,
                        0, 0, receiving_rank, TAG_ANY
                    );
                } else if(plus) {
                    //combine
                    err |= move(
                        MOVE_IMMEDIATE,
                        MOVE_IMMEDIATE,
                        MOVE_IMMEDIATE,
                        pack_flags(NO_COMPRESSION, RES_LOCAL, host),
                        func, count,
                        comm_offset, arcfg_offset,
                        first_move ? src_addr : previous_accumulator_addr, previous_recv_addr, last_move ? dst_addr : current_accumulator_addr, 0, 0, 0,
                        0, 0, 0, 0
                    );
                }
                //wait for completion on receives
                if(r){
                    while(rendezvous_get_completion(sending_rank, current_recv_addr, false, count, TAG_ANY) == NOT_READY_ERROR);
                }
                //end conditions
                if((!r && is_root) || (s && !is_root)) return err;
                //update first move
                if(s || plus){
                    first_move = false;
                }
                //increment step
                n++;
                //update scratchpads
                previous_accumulator_addr = current_accumulator_addr;
                previous_recv_addr = current_recv_addr;
            }
        }

    } else {
        unsigned int next_in_ring = (world.local_rank + 1) % world.size;
        unsigned int prev_in_ring = (world.local_rank + world.size-1) % world.size;
        if( prev_in_ring == root_rank){
            //non root ranks immediately after the root sends; only OP0_STREAM and OP0_HOST flags are relevant here
            return send(next_in_ring, count, src_addr, comm_offset, arcfg_offset, TAG_ANY, compression, ((host & OP0_HOST) << 8) | (stream & OP0_STREAM));
        }else if (world.local_rank != root_rank){
            //non root ranks sends their data + data received from previous rank to the next rank in sequence as a daisy chain; only OP0_STREAM flag is relevant here
            return fused_recv_reduce_send(prev_in_ring, next_in_ring, count, func, src_addr, comm_offset, arcfg_offset, TAG_ANY, compression, ((host & OP0_HOST) << 8) | (stream & OP0_STREAM));
        }else{
            //root only receive from previous node in the ring, add its local buffer and save in destination buffer
            return fused_recv_reduce(prev_in_ring, count, func, src_addr, dst_addr, comm_offset, arcfg_offset, TAG_ANY, compression, buftype);
        }
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
    unsigned int buftype
){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    int i, curr_pos, rel_stride, next_in_ring, prev_in_ring;
    int err = NO_ERROR;
    unsigned int tmp_compression = NO_COMPRESSION;
    unsigned int bytes_count = datatype_nbytes*count;

    if(world.size == 1){
        //corner-case copy for when running a single-node reduction
        return copy(count, src_buf_addr, dst_buf_addr, arcfg_offset, compression, buftype);
    } else if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        //reduce-scatter via reduction+scatter
        //copy the OP0_HOST flag over RES_HOST 
        //because we're broadcasting from the allreduce result buffer
        unsigned int r_host = (host & OP0_HOST) | ((host & OP0_HOST) << 2);
        unsigned int r_buftype = (buftype & 0xFFFFFF00) | (r_host & 0xFF);
        //reduce step - we reduce back into src_buf_addr
        while(reduce(count*world.size, func, 0, src_buf_addr, src_buf_addr, comm_offset, arcfg_offset, compression, r_buftype) == NOT_READY_ERROR);
        //copy the RES_HOST flag over OP0_HOST 
        //because we're broadcasting from the allreduce result buffer
        host = (host & RES_HOST) | ((host & RES_HOST) >> 2);
        buftype = (buftype & 0xFFFFFF00) | (host & 0xFF);
        //broadcast step
        while(scatter(count, 0, src_buf_addr, dst_buf_addr, comm_offset, arcfg_offset, compression, buftype) == NOT_READY_ERROR);
    } else {
        next_in_ring = (world.local_rank + 1) % world.size;
        prev_in_ring = (world.local_rank + world.size - 1) % world.size;

        //preamble: send our data to next in ring
        //prime the address slot for the source, so we can subsequently stride against it
        start_move(
            MOVE_IMMEDIATE, MOVE_NONE, MOVE_NONE,
            pack_flags(NO_COMPRESSION, RES_LOCAL, NO_HOST),
            0,
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
            pack_flags(tmp_compression, RES_REMOTE, host & OP0_HOST),
            0,
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
                    pack_flags(tmp_compression, RES_REMOTE, host & OP0_HOST),
                    func,
                    count,
                    comm_offset, arcfg_offset,
                    0, 0, 0, rel_stride, 0, 0,
                    prev_in_ring, TAG_ANY, next_in_ring, TAG_ANY
                );
            } else{
                tmp_compression = compression | ((compression & ETH_COMPRESSED) >> 2);
                start_move(
                    MOVE_STRIDE, MOVE_ON_RECV, MOVE_IMMEDIATE,
                    pack_flags(tmp_compression, RES_LOCAL, host),
                    func,
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
    }
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
    unsigned int buftype
){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    int i, curr_pos, curr_count, rel_stride, next_in_ring, prev_in_ring;
    unsigned int max_seg_count, elems_remaining, elems, bulk_count, tail_count, moved_bytes;
    int err = NO_ERROR;
    unsigned int tmp_compression = NO_COMPRESSION;
    uint64_t seg_src_buf_addr = src_buf_addr;
    uint64_t seg_dst_buf_addr = dst_buf_addr;
    unsigned int bytes_count = datatype_nbytes*count;

    if(world.size == 1){
        //corner-case copy for when running a single-node reduction
        return copy(count, src_buf_addr, dst_buf_addr, arcfg_offset, compression, stream);
    } else if((bytes_count > max_eager_size) && (compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        //allreduce via reduction+broadcast
        //reduce step
        while(reduce(count, func, 0, src_buf_addr, dst_buf_addr, comm_offset, arcfg_offset, compression, buftype) == NOT_READY_ERROR);
        //copy the RES_HOST flag over OP0_HOST 
        //because we're broadcasting from the allreduce result buffer
        host = (host & RES_HOST) | ((host & RES_HOST) >> 2);
        buftype = (buftype & 0xFFFFFF00) | (host & 0xFF);
        //broadcast step
        while(broadcast(count, 0, dst_buf_addr, comm_offset, arcfg_offset, compression, buftype) == NOT_READY_ERROR);
    } else {
        //convert max segment size to max segment count
        //if pulling from a stream, segment size is irrelevant and we use the
        //count directly because streams can't be read losslessly
        if (stream & OP0_STREAM) {
            max_seg_count = count;
        } else {
            //compute count from uncompressed elem bytes in aright config
            //instead of Xil_In32 we could use:
            //(datapath_arith_config*)(arcfg_offset)->uncompressed_elem_bytes;
            max_seg_count = eager_rx_buf_size / datatype_nbytes;
            // Round max segment size down to align with world size
            max_seg_count -= max_seg_count % world.size;
        }

        next_in_ring = (world.local_rank + 1) % world.size;
        prev_in_ring = (world.local_rank + world.size - 1) % world.size;

        for (elems_remaining = count; elems_remaining > 0; elems_remaining -= elems) {
            elems = min(max_seg_count, elems_remaining);

            //we need to break the input into world.size chunks of equal size
            //if count does not divide by world.size, the chunk with the largest index (tail) will be smaller
            bulk_count = (elems + world.size - 1) / world.size;//equivalent to ceil(elems/world.size)
            tail_count = elems - bulk_count * (world.size - 1);

            //preamble: send our data to next in ring
            //prime the address slots for the source and destination,
            //so we can subsequently stride against them
            start_move(
                MOVE_IMMEDIATE, MOVE_NONE, MOVE_IMMEDIATE,
                pack_flags(NO_COMPRESSION, RES_LOCAL, NO_HOST),
                0,
                0,
                0, arcfg_offset,
                seg_src_buf_addr, 0, seg_dst_buf_addr, 0, 0, 0,
                0, 0, 0, 0
            );

            //send local chunk to next in ring
            //send: keep OP0_COMPRESSED, replace RES_COMPRESSED by ETH_COMPRESSED
            tmp_compression = (compression & OP0_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 1);
            start_move(
                MOVE_STRIDE, MOVE_NONE, MOVE_IMMEDIATE,
                pack_flags(tmp_compression, RES_REMOTE, host & OP0_HOST),
                0,
                (world.local_rank == world.size - 1) ? tail_count: bulk_count,
                comm_offset, arcfg_offset,
                0, 0, 0, bulk_count * world.local_rank, 0, 0,
                0, 0, next_in_ring, TAG_ANY
            );

            //receive and reduce+forward from all other members of the communicator
            curr_pos = world.local_rank;
            for (i = 0; i < world.size - 1; i++) {
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
                        pack_flags(tmp_compression, RES_REMOTE, host & OP0_HOST),
                        func,
                        curr_count,
                        comm_offset, arcfg_offset,
                        0, 0, 0, rel_stride, 0, 0,
                        prev_in_ring, TAG_ANY, next_in_ring, TAG_ANY
                    );
                } else {
                    //compression: if ETH_COMPRESSED is set then we receive compressed data over the network,
                    //so we must set OP1_COMPRESSED here. Also keep RES_COMPRESSED and OP0_COMPRESSED the same.
                    tmp_compression = compression | ((compression & ETH_COMPRESSED) >> 2);
                    start_move(
                        MOVE_STRIDE, MOVE_ON_RECV, MOVE_STRIDE,
                        pack_flags(tmp_compression, RES_LOCAL, host & (OP0_HOST | RES_HOST)),
                        func,
                        curr_count,
                        comm_offset, arcfg_offset,
                        0, 0, 0, rel_stride, 0, bulk_count * next_in_ring,
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
            //send to next in ring, from seg_dst_buf_addr where we stored the scattered reduction result
            start_move(
                MOVE_IMMEDIATE, MOVE_NONE, MOVE_NONE,
                pack_flags(NO_COMPRESSION, RES_LOCAL, NO_HOST),
                0,
                0,
                0, arcfg_offset,
                seg_dst_buf_addr, 0, 0, 0, 0, 0,
                0, 0, 0, 0
            );
            //send: keep all flags except RES_COMPRESSED, which should be replaced by ETH_COMPRESSED
            //convert RES_HOST flag to OP0_HOST because we're sending from the destination buffer via Op0
            tmp_compression = (compression & ~RES_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 1);
            start_move(
                MOVE_STRIDE, MOVE_NONE, MOVE_IMMEDIATE,
                pack_flags(tmp_compression, RES_REMOTE, (host & RES_HOST)>>2),
                0,
                (next_in_ring == world.size - 1) ? tail_count : bulk_count,
                comm_offset, arcfg_offset,
                0, 0, 0, bulk_count * next_in_ring, 0, 0,
                0, 0, next_in_ring, TAG_ANY
            );

            err |= end_move();
            err |= end_move();

            //receive and forward from all other members of the communicator
            curr_pos = next_in_ring;
            for (i = 0; i < world.size - 1; i++) {
                rel_stride = bulk_count * ((curr_pos == 0) ? (world.size - 1) : -1);
                curr_count = (curr_pos == 0) ? tail_count : bulk_count;

                //we use a blocking move here, because we want to avoid a race condition with the relay below
                //TODO: avoid this; we either need to solve the RAW dependency in hardware (the generic approach),
                //or a way to reuse a rx buffer (solves this problem in particular but does not extend to e.g. reduces)
                //e.g. MOVE_ON_RECV_KEEP which would keep the RX buffer in the pending state
                tmp_compression = (compression & RES_COMPRESSED) | ((compression & ETH_COMPRESSED) >> 2);
                err |= move(
                    MOVE_NONE, MOVE_ON_RECV, MOVE_STRIDE,
                    pack_flags(tmp_compression, RES_LOCAL, host & RES_HOST),
                    0,
                    curr_count,
                    comm_offset, arcfg_offset,
                    0, 0, 0, 0, 0, rel_stride,
                    prev_in_ring, TAG_ANY, 0, 0
                );
                curr_pos = (curr_pos + world.size - 1) % world.size;
                curr_count = (curr_pos == (world.size - 1)) ? tail_count : bulk_count;
                if (i < world.size - 2) { //if not the last data, relay to the next in ring
                    //first prime the address
                    start_move(
                        MOVE_NONE, MOVE_IMMEDIATE, MOVE_NONE,
                        pack_flags(NO_COMPRESSION, RES_REMOTE, NO_HOST),
                        0,
                        0,
                        0, arcfg_offset,
                        0, seg_dst_buf_addr, 0, 0, 0, 0,
                        0, 0, 0, 0
                    );
                    //send
                    //we're re-sending from the result, so copy RES_COMPRESSED over OP1_COMPRESSED,
                    //and ETH_COMPRESSED over RES_COMPRESSED
                    //we convert RES_HOST flag to OP1_HOST since we're sending from the result buffer via Op1
                    tmp_compression = ((compression & RES_COMPRESSED) >> 1) | ((compression & ETH_COMPRESSED) >> 1);
                    start_move(
                        MOVE_NONE, MOVE_STRIDE, MOVE_IMMEDIATE,
                        pack_flags(tmp_compression, RES_REMOTE, (host & RES_HOST) >> 1),
                        0,
                        curr_count,
                        comm_offset, arcfg_offset,
                        0, 0, 0, 0, bulk_count * curr_pos, 0,
                        0, 0, next_in_ring, TAG_ANY
                    );

                    err |= end_move();
                    err |= end_move();
                }
            }

            moved_bytes = elems * datatype_nbytes;
            seg_src_buf_addr += moved_bytes;
            seg_dst_buf_addr += moved_bytes;
        }
    }

    return err;
}

//barrier using the rendezvous notification mechanism
int barrier(
    uint64_t src_buf_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset
){
    uint64_t addr;
    bool host;
    unsigned int i;
    int err = NO_ERROR;

    if(world.size == 1){
        //corner-case single-node barrier
        return NO_ERROR;
    }

    //flush retry queue
    flush_retries = true;
    if(num_retry_pending > 0){
        return NOT_READY_ERROR;
    }

    //gather notifications to rank 0
    if(world.local_rank == 0){
        for(i=1; i<world.size; i++){
            while(rendezvous_get_addr(i, &addr, &host, 0, TAG_ANY) == NOT_READY_ERROR);
        }
    } else {
        rendezvous_send_addr(0, 0, false, 0, TAG_ANY);
    }

    //scatter notifications from rank 0
    if(world.local_rank == 0){
        for(i=1; i<world.size; i++){
            rendezvous_send_addr(i, 0, false, 0, TAG_ANY);
        }
    } else {
        while(rendezvous_get_addr(0, &addr, &host, 0, TAG_ANY)== NOT_READY_ERROR);
    }

    flush_retries = false;

    return err;
}

//placeholder for all-to-all collective
int all_to_all(
    unsigned int count,
    uint64_t src_buf_addr,
    uint64_t dst_buf_addr,
    unsigned int comm_offset,
    unsigned int arcfg_offset,
    unsigned int compression,
    unsigned int buftype
){
    unsigned int stream = buftype & 0xff;
    unsigned int host = (buftype >> 8) & 0xff;
    int err = NO_ERROR;
    unsigned int bytes_count = datatype_nbytes*count;

    if(world.size == 1){
        //corner-case single-node alltoall
        return copy(count, src_buf_addr, dst_buf_addr, arcfg_offset, compression, stream);
    } else if(/*(bytes_count > max_eager_size) && */(compression == NO_COMPRESSION) && (stream == NO_STREAM)){
        //alltoall via simultaneous broadcast
        //since in alltoall each endpoint must receive P-1 messages (where P is the world size)
        //then the minimum time required to complete the alltoall is (P-1)(latency+M/bandwidth)
        //therefore alltoall is linear in P and M, regardless of our broadcast implementation
        //therefore, we use the out-of-order flat tree implementation for broadcast
        //but for performance we fuse one locally-rooted broadcast with P-1 locally receiving broadcasts
        //NOTE: while this flat tree implementation has better tolerance for rank skews, it
        //also has less control over congestion, as the logical topology allows fanin > 1

        //get remote address
        //if root, wait for addresses then send
        unsigned int pending_moves = 0;
        uint64_t dst_addr;
        bool dst_host;
        unsigned int dst_rank;
        int i;
        bool res_is_host = ((host & RES_HOST) != 0);

        //start local copy of our source data to the destination buffer
        start_move(
            MOVE_IMMEDIATE,
            MOVE_NONE,
            MOVE_IMMEDIATE,
            pack_flags(NO_COMPRESSION, RES_LOCAL, host),
            0, count, comm_offset, arcfg_offset, 
            src_buf_addr + world.local_rank*bytes_count, 0, dst_buf_addr + world.local_rank*bytes_count, 0, 0, 0,
            0, 0, 0, TAG_ANY
        );
        pending_moves++;

        //send our addresses to peers while copy is happening
        for(i=0; i<world.size; i++){
            if(i != world.local_rank){
                rendezvous_send_addr(i, dst_buf_addr+i*bytes_count, res_is_host, count, TAG_ANY);
            }
        }

        //send to peers as we get their addresses
        for(i=0; i<(world.size-1); i++){
            while(rendezvous_get_any_addr(&dst_rank, &dst_addr, &dst_host, count, TAG_ANY) == NOT_READY_ERROR);
            if(dst_host){
                host |= RES_HOST;
            }
            //do a RDMA write to the remote address 
            start_move(
                MOVE_IMMEDIATE,
                MOVE_NONE,
                MOVE_IMMEDIATE,
                pack_flags_rendezvous(host),
                0, count, comm_offset, arcfg_offset, 
                src_buf_addr + dst_rank*bytes_count, 0, dst_addr, 0, 0, 0,
                0, 0, dst_rank, TAG_ANY
            );
            pending_moves++;
            if(pending_moves > 2){
                err |= end_move();
                pending_moves--;
            }
        }

        //get completions as they come
        for(i=0; i<(world.size-1); i++){
            while(rendezvous_get_any_completion(&dst_rank, &dst_addr, &res_is_host, bytes_count, TAG_ANY) == NOT_READY_ERROR);
        }

        //make sure all RDMA writes have finished and return
        while(pending_moves > 0){
            err |= end_move();
            pending_moves--;
        }
        return err;

    } else {
        //TODO
        return COLLECTIVE_NOT_IMPLEMENTED;
    }

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
    //mark communicator not cached
    comm_cached = false;
    new_call = true;
    flush_retries = false;
}

//reset the control module and flush any pending operations
void encore_soft_reset(void){
    //1. drop all calls in retry queue
    while(tngetd(STS_CALL_RETRY) == 0){
        int i;
        for(i=0; i<16; i++){
            getd(STS_CALL_RETRY);
        }
        Xil_Out32(RETVAL_OFFSET, NOT_READY_ERROR);
        putd(STS_CALL, NOT_READY_ERROR);
    }
    //2. activate reset pin
    CLR(GPIO_DATA_REG, GPIO_SWRST_MASK);
}

//poll for a call from the host
static inline void wait_for_call(void) {
    // Poll the cmd queues
    unsigned int invalid;
    bool rr_sel = !new_call;
    do {
        invalid = 0;
        if(rr_sel || flush_retries){
            invalid += tngetd(STS_CALL_RETRY);
            rr_sel = false;
        } else {
            invalid += tngetd(CMD_CALL);
            rr_sel = true;
        }
    } while (invalid);
    new_call = rr_sel;
    //clear performance counter then start it
#ifndef MB_FW_EMULATION
    SET(PERFCTR_CONTROL_REG, PERFCTR_CE_MASK);
    CLR(PERFCTR_CONTROL_REG, PERFCTR_SCLR_MASK);
#else
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
#endif
}

//signal finish to the host and write ret value in exchange mem
void finalize_call(unsigned int retval) {
    Xil_Out32(RETVAL_OFFSET, retval);
    //stop performance counter, copy its value to exchmem
#ifndef MB_FW_EMULATION
    CLR(PERFCTR_CONTROL_REG, PERFCTR_CE_MASK);
    Xil_Out32(PERFCTR_OFFSET, Xil_In32(PERFCTR_DATA_REG));
    SET(PERFCTR_CONTROL_REG, PERFCTR_SCLR_MASK);
#else
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count() - duration_us;
    Xil_Out32(PERFCTR_OFFSET, (uint32_t)((duration_us*250)&0xffffffff));
#endif
    // Done: Set done and idle
    putd(STS_CALL, retval);
}

void run() {
    unsigned int retval;
    unsigned int scenario, count, comm, root_src_dst, function, msg_tag;
    unsigned int datapath_cfg, compression_flags, buftype_flags;
    unsigned int op0_addrl, op0_addrh, op1_addrl, op1_addrh, res_addrl, res_addrh;
    uint64_t op0_addr, op1_addr, res_addr;

    init();

    while (1) {
        wait_for_call();
        if(new_call){
            //read parameters from host command queue
            scenario            = getd(CMD_CALL);
            count               = getd(CMD_CALL);
            comm                = getd(CMD_CALL);
            root_src_dst        = getd(CMD_CALL);
            function            = getd(CMD_CALL);
            msg_tag             = getd(CMD_CALL);
            datapath_cfg        = getd(CMD_CALL);
            compression_flags   = getd(CMD_CALL);
            buftype_flags       = getd(CMD_CALL);
            op0_addrl           = getd(CMD_CALL);
            op0_addrh           = getd(CMD_CALL);
            op1_addrl           = getd(CMD_CALL);
            op1_addrh           = getd(CMD_CALL);
            res_addrl           = getd(CMD_CALL);
            res_addrh           = getd(CMD_CALL);
            current_step        = 0;
        } else {
            //read parameters from command retry queue
            scenario            = getd(STS_CALL_RETRY);
            count               = getd(STS_CALL_RETRY);
            comm                = getd(STS_CALL_RETRY);
            root_src_dst        = getd(STS_CALL_RETRY);
            function            = getd(STS_CALL_RETRY);
            msg_tag             = getd(STS_CALL_RETRY);
            datapath_cfg        = getd(STS_CALL_RETRY);
            compression_flags   = getd(STS_CALL_RETRY);
            buftype_flags       = getd(STS_CALL_RETRY);
            op0_addrl           = getd(STS_CALL_RETRY);
            op0_addrh           = getd(STS_CALL_RETRY);
            op1_addrl           = getd(STS_CALL_RETRY);
            op1_addrh           = getd(STS_CALL_RETRY);
            res_addrl           = getd(STS_CALL_RETRY);
            res_addrh           = getd(STS_CALL_RETRY);
            current_step        = getd(STS_CALL_RETRY);
            num_retry_pending--;
        }

        op0_addr = ((uint64_t) op0_addrh << 32) | op0_addrl;
        op1_addr = ((uint64_t) op1_addrh << 32) | op1_addrl;
        res_addr = ((uint64_t) res_addrh << 32) | res_addrl;

        //initialize arithmetic/compression config and communicator
        //NOTE: these are global because they're used in a lot of places but don't change during a call
        //TODO: determine if they can remain global in hierarchical collectives
        if(scenario != ACCL_CONFIG && scenario != ACCL_NOP){
            if(!comm_cached || (comm != comm_cache_adr)){
                world = find_comm(comm);
                comm_cached = true;
                comm_cache_adr = comm;
            }
            //get number of bytes of selected datatype
            datatype_nbytes = Xil_In32(datapath_cfg);
        }

        switch (scenario)
        {
            case ACCL_COPY:
                retval = copy(count, op0_addr, res_addr, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_COMBINE:
                retval = combine(count, function, op0_addr, op1_addr, res_addr, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_SEND:
                retval = send(root_src_dst, count, op0_addr, comm, datapath_cfg, msg_tag, compression_flags, buftype_flags);
                break;
            case ACCL_RECV:
                retval = recv(root_src_dst, count, res_addr, comm, datapath_cfg, msg_tag, compression_flags, buftype_flags);
                break;
            case ACCL_BCAST:
                retval = broadcast(count, root_src_dst, op0_addr, comm, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_SCATTER:
                retval = scatter(count, root_src_dst, op0_addr, res_addr, comm, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_GATHER:
                retval = gather(count, root_src_dst, op0_addr, res_addr, comm, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_REDUCE:
                retval = reduce(count, function, root_src_dst, op0_addr, res_addr, comm, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_ALLGATHER:
                retval = allgather(count, op0_addr, res_addr, comm, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_REDUCE_SCATTER:
                retval = reduce_scatter(count, function, op0_addr, res_addr, comm, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_ALLREDUCE:
                retval = allreduce(count, function, op0_addr, res_addr, comm, datapath_cfg, compression_flags, buftype_flags);
                break;
            case ACCL_BARRIER:
                retval = barrier(op0_addr, comm, datapath_cfg);
                break;
            case ACCL_ALLTOALL:
                retval = all_to_all(count, op0_addr, res_addr, comm, datapath_cfg, compression_flags, buftype_flags);
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
                    case HOUSEKEEP_EAGER_MAX_SIZE:
                        //get size of RX buffers from the RX buffer spec in memory
                        eager_rx_buf_size = Xil_In32(RX_BUFFER_MAX_LEN_OFFSET);
                        if(count < eager_rx_buf_size){
                            finalize_call(EAGER_THRESHOLD_INVALID);
                            continue;
                        }
                        max_eager_size = count;

                        break;
                    case HOUSEKEEP_RENDEZVOUS_MAX_SIZE:
                        if(count <= max_eager_size){
                            finalize_call(RENDEZVOUS_THRESHOLD_INVALID);
                            continue;
                        }
                        max_rendezvous_size = count;
                        break;
                    default:
                        break;
                }
                break;
            case ACCL_NOP:
                retval = NO_ERROR;
                break;
            default:
                retval = COLLECTIVE_NOT_IMPLEMENTED;
                break;
        }
        if(retval == NOT_READY_ERROR){
            //put the current call in the retry queue
            putd(CMD_CALL_RETRY,scenario);
            putd(CMD_CALL_RETRY,count);
            putd(CMD_CALL_RETRY,comm);
            putd(CMD_CALL_RETRY,root_src_dst);
            putd(CMD_CALL_RETRY,function);
            putd(CMD_CALL_RETRY,msg_tag);
            putd(CMD_CALL_RETRY,datapath_cfg);
            putd(CMD_CALL_RETRY,compression_flags);
            putd(CMD_CALL_RETRY,buftype_flags);
            putd(CMD_CALL_RETRY,op0_addrl);
            putd(CMD_CALL_RETRY,op0_addrh);
            putd(CMD_CALL_RETRY,op1_addrl);
            putd(CMD_CALL_RETRY,op1_addrh);
            putd(CMD_CALL_RETRY,res_addrl);
            putd(CMD_CALL_RETRY,res_addrh);
            cputd(CMD_CALL_RETRY,current_step);
            num_retry_pending++;
        } else {
            finalize_call(retval);
        }
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
