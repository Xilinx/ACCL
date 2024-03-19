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

#include <stdint.h>
#ifdef MB_FW_EMULATION
#include <stdio.h>
#include <semaphore.h>
#include "Stream.h"
#include "ap_axi_sdata.h"
#endif

#ifndef _CCL_OFFLOAD_CONTROL_H_
#define _CCL_OFFLOAD_CONTROL_H_

#define KERNEL_NAME    "ccl_offload"
#define KERNEL_VENDOR  "Xilinx"
#define KERNEL_LIBRARY "ACCL"

//Datapath width (in bytes) for all streams everywhere in the design
#define DATAPATH_WIDTH_BYTES 64

//AXIS interfaces to/from MB 

#define CMD_CALL     0
#define CMD_DMA_MOVE 1
#define CMD_RNDZV    2
#define CMD_RNDZV_PENDING 3
#define CMD_CALL_RETRY    4

#define STS_CALL     0
#define STS_DMA_MOVE 1
#define STS_RNDZV    2
#define STS_RNDZV_PENDING 3
#define STS_CALL_RETRY    4

//PACKT CONST
#define MAX_PACKETSIZE 4096

//DMA CONST 
#define DMA_MAX_BTT              (((1<<23)-1)/64*64)

//******************************
//**  XCC Operations          **
//******************************
//Housekeeping
#define ACCL_CONFIG         0
//Primitives
#define ACCL_COPY           1
#define ACCL_COMBINE        2
#define ACCL_SEND           3 
#define ACCL_RECV           4
//Collectives
#define ACCL_BCAST          5
#define ACCL_SCATTER        6
#define ACCL_GATHER         7
#define ACCL_REDUCE         8
#define ACCL_ALLGATHER      9
#define ACCL_ALLREDUCE      10
#define ACCL_REDUCE_SCATTER 11
#define ACCL_BARRIER        12
#define ACCL_ALLTOALL       13
#define ACCL_NOP            255

//ACCL_CONFIG SUBFUNCTIONS
#define HOUSEKEEP_SWRST                0
#define HOUSEKEEP_PKTEN                1
#define HOUSEKEEP_TIMEOUT              2
#define HOUSEKEEP_EAGER_MAX_SIZE       3
#define HOUSEKEEP_RENDEZVOUS_MAX_SIZE  4

//AXI MMAP address
#define GATHER_FLAT_TREE_MAX_FANIN_OFFSET  0x1FC4
#define GATHER_FLAT_TREE_MAX_COUNT_OFFSET  0x1FC8
#define BCAST_FLAT_TREE_MAX_RANKS_OFFSET  0x1FCC
#define REDUCE_FLAT_TREE_MAX_RANKS_OFFSET 0x1FD0
#define REDUCE_FLAT_TREE_MAX_COUNT_OFFSET 0x1FD4
#define TMP1_OFFSET       0x1FD8    //address of first spare buffer in device memory
#define TMP2_OFFSET       0x1FE0    //address of second spare buffer in device memory
#define TMP3_OFFSET       0x1FE8    //address of third spare buffer in device memory
#define PERFCTR_OFFSET    0x1FF0
#define CFGRDY_OFFSET     0x1FF4
#define HWID_OFFSET       0x1FF8
#define RETVAL_OFFSET     0x1FFC
#define END_OF_EXCHMEM    0x2000

#ifndef MB_FW_EMULATION  
#define EXCHMEM_BASEADDR      0x0
#define NET_RXPKT_BASEADDR    0x30000
#define NET_TXPKT_BASEADDR    0x40000
#define RX_DEQUEUE_BASEADDR   0x50000
#define RX_ENQUEUE_BASEADDR   0x60000
#define RX_SEEK_BASEADDR      0x70000
#define GPIO_BASEADDR         0x40000000
#define PERFCTR_BASEADDR      0x40001000
#else       
#define EXCHMEM_BASEADDR      0x0000
#define NET_RXPKT_BASEADDR    0x3000
#define NET_TXPKT_BASEADDR    0x4000
#define RX_DEQUEUE_BASEADDR   0x5000
#define RX_ENQUEUE_BASEADDR   0x6000
#define RX_SEEK_BASEADDR      0x7000
#define GPIO_BASEADDR         0x8000
#define PERFCTR_BASEADDR      0x9000
#endif

//https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/managing_interface_synthesis.html#tzw1539734223235
#define CONTROL_START_MASK      0x00000001
#define CONTROL_DONE_MASK       0x00000001 << 1
#define CONTROL_IDLE_MASK       0x00000001 << 2 
#define CONTROL_READY_MASK      0x00000001 << 3
#define CONTROL_REPEAT_MASK     0x00000001 << 7

#define GPIO_DATA_REG           GPIO_BASEADDR + 0x0000
#define GPIO_TRI_REG            GPIO_BASEADDR + 0x0004
#define GPIO2_DATA_REG          GPIO_BASEADDR + 0x0008
#define GPIO2_TRI_REG           GPIO_BASEADDR + 0x000C
#define PERFCTR_CONTROL_REG     PERFCTR_BASEADDR +0x0000
#define PERFCTR_DATA_REG        PERFCTR_BASEADDR +0x0008
#define GPIO_READY_MASK         0x00000001
#define GPIO_SWRST_MASK         0x00000002
#define PERFCTR_SCLR_MASK       0x00000001
#define PERFCTR_CE_MASK         0x00000002

//EXCEPTIONS
#define NO_ERROR                                      0   
#define DMA_MISMATCH_ERROR                            (1<< 0)    
#define DMA_INTERNAL_ERROR                            (1<< 1)    
#define DMA_DECODE_ERROR                              (1<< 2) 
#define DMA_SLAVE_ERROR                               (1<< 3)
#define DMA_NOT_OKAY_ERROR                            (1<< 4)    
#define DMA_NOT_END_OF_PACKET_ERROR                   (1<< 5)            
#define DMA_NOT_EXPECTED_BTT_ERROR                    (1<< 6)
#define DMA_TIMEOUT_ERROR                             (1<< 7)            
#define CONFIG_SWITCH_ERROR                           (1<< 8)
#define DEQUEUE_BUFFER_TIMEOUT_ERROR                  (1<< 9)
#define DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR      (1<<10)
#define RECEIVE_TIMEOUT_ERROR                         (1<<11)
#define DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH   (1<<12)
#define DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR       (1<<13)
#define COLLECTIVE_NOT_IMPLEMENTED                    (1<<14)
#define RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID       (1<<15)
#define EAGER_THRESHOLD_INVALID                       (1<<16)
#define RENDEZVOUS_THRESHOLD_INVALID                  (1<<17)
#define DMA_SIZE_ERROR                                (1<<18)
#define ARITH_ERROR                                   (1<<19) 
#define PACK_TIMEOUT_STS_ERROR                        (1<<20)
#define PACK_SEQ_NUMBER_ERROR                         (1<<21)
#define COMPRESSION_ERROR                             (1<<22)
#define KRNL_TIMEOUT_STS_ERROR                        (1<<23)
#define KRNL_STS_COUNT_ERROR                          (1<<24)
#define SEGMENTER_EXPECTED_BTT_ERROR                  (1<<25)
#define DMA_TAG_MISMATCH_ERROR                        (1<<26)
#define NOT_READY_ERROR                               (1<<31)

//define opcodes for move offload
//each address parameter (op0, op1, res) should carry one of these opcodes
#define MOVE_NONE      0 //this parameter is not in use
#define MOVE_STREAM    1 //will move using a stream (no address) - only available on op0 and res
#define MOVE_IMMEDIATE 2 //will move using an immediate address - available on both ops and res (res will currently ignore address if RES_REMOTE is set)
#define MOVE_ON_RECV   3 //will resolve address from RX dequeue using immediate src, tag (replacing address), and len - only available on op1
#define MOVE_INCREMENT 4 //resolve new address by adding together the count and address of the previous move - available on both ops and res
#define MOVE_REPEAT    5 //use address of previous move - available on both ops and res
#define MOVE_STRIDE    6 //resolve address by adding an immediate stride count (replacing address) to address of previous move - available on both ops and res

//define compression flags; these are one-hot, one bit per parameter
//ETH_COMPRESSED is a meta-flag, it's passed in the call to the CCLO,
//but it does not go down into the move operation, but instead 
//influences the value of the actual compression flags for each move step
//e.g. a broadcast with just ETH_COMPRESSED flag turns into:
//(on root) multiple moves (MOVE_IMMEDIATE then MOVE_STRIDE or MOVE_INCREMENT) with RES_COMPRESSED set
//(elsewhere) one MOVE_IMMEDIATE+MOVE_ON_RECV with OP1_COMPRESSED
#define NO_COMPRESSION 0
#define OP0_COMPRESSED (1<<0)
#define OP1_COMPRESSED (1<<1)
#define RES_COMPRESSED (1<<2)
#define ETH_COMPRESSED (1<<3)

//define if result is stored locally or sent to remote node
#define RES_LOCAL  0
#define RES_REMOTE 1

//flags as sent by the host indicating streaming operands
#define NO_STREAM  0
#define OP0_STREAM 1
#define RES_STREAM 2

//flags as sent by the host indicating host-side operands
#define NO_HOST  0
#define OP0_HOST (1<<0)
#define OP1_HOST (1<<1)
#define RES_HOST (1<<2)

typedef struct{
    uint64_t ptr; //actual byte pointer to the data
    unsigned int elem_ratio; //scaling factor
    unsigned int elem_bytes; //bytes per element
} dm_addr;

typedef struct{
    unsigned int compression; //compression options
    unsigned int stream; //stream option
    unsigned int remote; //indicates whether any comm is to remote nodes
    unsigned int which_dm; //indicate which datamover hardware to utilize
    unsigned int elems_remaining; //number of uncompressed elements remaining to transfer
    unsigned int elems_per_transfer; //number of uncompressed elements to transfer in each chunk
    unsigned int op0_len; //elems_per_transfer * elem_width (which depends on compression)
    unsigned int op1_len; //elems_per_transfer * elem_width (which depends on compression)
    unsigned int res_len; //elems_per_transfer * elem_width (which depends on compression)
    dm_addr op0_addr;
    dm_addr op1_addr;
    dm_addr res_addr;
} dm_config;

//utility functions for register mapped accesses
#ifdef MB_FW_EMULATION
extern uint32_t *cfgmem;
#else
extern uint32_t volatile *cfgmem;
#endif
#define Xil_Out32(offset, value) {cfgmem[(offset)/4] = (value);}
#define Xil_In32(offset) ({uint32_t value = cfgmem[(offset)/4]; value; })

#define SET(offset, mask) Xil_Out32(offset, Xil_In32(offset) | (mask))
#define CLR(offset, mask) Xil_Out32(offset, Xil_In32(offset) & ~(mask))

//stream handling
#ifndef MB_FW_EMULATION
//push data to stream
#define putd(channel, value) asm volatile ("putd\t%0,%1" :: "d" (value), "d" (channel))
//push data to stream and set control bit
#define cputd(channel, value) asm volatile ("cputd\t%0,%1" :: "d" (value), "d" (channel))
//test-only non-blocking get from stream channel
#define tngetd(channel) ({unsigned int value; asm volatile ("tngetd\tr0,%1\n\taddic\t%0,r0,0" : "=d" (value) : "d" (channel)); value; })
//blocking get from stream channel
#define getd(channel) ({unsigned int value; asm volatile ("getd\t%0,%1" : "=d" (value) : "d" (channel)); value; })

#else

extern hlslib::Stream<ap_axiu<32,0,0,0>, 512> cmd_fifos[5];
extern hlslib::Stream<ap_axiu<32,0,0,0>, 512> sts_fifos[5];

//push data to stream
#define putd(channel, value) cmd_fifos[channel].Push((ap_axiu<32,0,0,0>){.data=value, .last=0})
//push data to stream and set control bit
#define cputd(channel, value) cmd_fifos[channel].Push((ap_axiu<32,0,0,0>){.data=value, .last=1})
//test-only non-blocking get from stream channel
#define tngetd(channel) sts_fifos[channel].IsEmpty()
//blocking get from stream channel
#define getd(channel) ((sts_fifos[channel].Pop()).data)

#endif

typedef struct {
    unsigned int status;
    unsigned int addrl;
    unsigned int addrh;
    unsigned int max_len;
    unsigned int rx_tag;
    unsigned int rx_len;
    unsigned int rx_src;
    unsigned int sequence_number;
} rx_buffer;

#define STATUS_OFFSET           0
#define ADDRL_OFFSET            1
#define ADDRH_OFFSET            2
#define RX_TAG_OFFSET           3
#define RX_LEN_OFFSET           4
#define RX_SRC_OFFSET           5
#define SEQUENCE_NUMBER_OFFSET  6   
#define SPARE_BUFFER_FIELDS     7       

#define STATUS_IDLE     0x00
#define STATUS_ENQUEUED 0x01
#define STATUS_RESERVED 0x02
#define STATUS_ERROR    0x04

#define RX_BUFFER_COUNT_OFFSET 0x0
#define RX_BUFFER_MAX_LEN_OFFSET 0x4
#define RX_BUFFER_METADATA_OFFSET 0x8
#define COMM_OFFSET (RX_BUFFER_METADATA_OFFSET+4*Xil_In32(RX_BUFFER_COUNT_OFFSET)*SPARE_BUFFER_FIELDS)

typedef struct {
    unsigned int ip;
    unsigned int port;
    unsigned int inbound_seq;
    unsigned int outbound_seq;
    unsigned int session;
    unsigned int max_seg_size;
} comm_rank;

typedef struct {
    unsigned int size;
    unsigned int local_rank;
    comm_rank* ranks;
} communicator;

//COMMUNICATOR OFFSETS
#define COMM_SIZE_OFFSET                 0
#define COMM_LOCAL_RANK_OFFSET           1
#define COMM_RANKS_OFFSET                2
//RANK OFFSET
#define RANK_IP_OFFSET                   0
#define RANK_PORT_OFFSET                 1
#define RANK_INBOUND_SEQ_OFFSET          2
#define RANK_OUTBOUND_SEQ_OFFSET         3
#define RANK_SESSION_OFFSET              4
#define RANK_SEGLEN_OFFSET               5
#define RANK_SIZE                        6


//structure defining arithmetic config parameters
//TODO: make unsigned char to save on space
#define MAX_REDUCE_FUNCTIONS 10

typedef struct {
    unsigned int uncompressed_elem_bytes;//bitwidth of one element of uncompressed data
    unsigned int compressed_elem_bytes;  //bitwidth of one element of compressed data
    unsigned int elem_ratio_log;         //how many uncompressed elements per compressed element
    unsigned int compressor_tdest;      //clane TDEST for targeting the compressor
    unsigned int decompressor_tdest;    //clane TDEST for targeting the compressor
    unsigned int arith_nfunctions;      //number of supported functions (<= MAX_REDUCE_FUNCTIONS)
    unsigned int arith_is_compressed;   //perform arithmetic on compressed (1) or uncompressed (0) values
    unsigned int arith_tdest[MAX_REDUCE_FUNCTIONS]; //arithmetic TDEST
} datapath_arith_config;

//Tag definitions
#define TAG_ANY 0xFFFFFFFF

//define exception handling for simulation
#ifdef MB_FW_EMULATION
#define setjmp(x) 0
void finalize_call(unsigned int);
#define longjmp(x, y) finalize_call(y)
#endif

#ifdef __cplusplus
extern "C" void run_accl();
extern "C" void stream_isr();
#endif

#endif // _CCL_OFFLOAD_CONTROL_H_
