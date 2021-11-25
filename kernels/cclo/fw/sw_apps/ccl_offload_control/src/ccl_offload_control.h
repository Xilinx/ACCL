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
#ifdef ACCL_BD_SIM
#include <stdio.h>
#include <semaphore.h>
#endif

#ifndef _CCL_OFFLOAD_CONTROL_H_
#define _CCL_OFFLOAD_CONTROL_H_

#define KERNEL_NAME    "ccl_offload"
#define KERNEL_VENDOR  "Xilinx"
#define KERNEL_LIBRARY "XCCL"

//Datapath width (in bytes) for all streams everywhere in the design
#define DATAPATH_WIDTH_BYTES 64

//AXIS interfaces to/from MB 

#define CMD_DMA0_TX  0
#define CMD_DMA0_RX  1
#define CMD_DMA1_TX  2
#define CMD_DMA1_RX  3
#define CMD_NET_TX   4
#define CMD_NET_PORT 5
#define CMD_NET_CON  6
#define CMD_HOST     7
#define CMD_KRNL_PKT 8

#define STS_DMA0_RX  0
#define STS_DMA0_TX  1
#define STS_DMA1_RX  2
#define STS_DMA1_TX  3
#define STS_NET_RX   4
#define STS_NET_PORT 5
#define STS_NET_CON  6
#define STS_HOST     7 
#define STS_KRNL_PKT 8
#define STS_NET_PKT  9

//MAIN SWITCH

#define DATAPATH_DMA_LOOPBACK      1
#define DATAPATH_DMA_REDUCTION     2
#define DATAPATH_OFFCHIP_TX        3
#define DATAPATH_OFFCHIP_REDUCTION 4

#define SWITCH_M_NET_TX    0
#define SWITCH_M_DMA1_TX   1
#define SWITCH_M_EXT_KRNL  2
#define SWITCH_M_ARITH_OP0 3
#define SWITCH_M_ARITH_OP1 4
#define SWITCH_M_COMPRESS0 5
#define SWITCH_M_COMPRESS1 6
#define SWITCH_M_COMPRESS2 7

#define SWITCH_S_DMA0_RX   0
#define SWITCH_S_DMA1_RX   1
#define SWITCH_S_EXT_KRNL  2
#define SWITCH_S_ARITH_RES 3
#define SWITCH_S_COMPRESS0 4
#define SWITCH_S_COMPRESS1 5
#define SWITCH_S_COMPRESS2 6

//PACKT CONST
#define MAX_PACKETSIZE 1536
#define MAX_SEG_SIZE 1048576
//DMA CONST 
#define DMA_MAX_BTT              0x7FFFFF
#define DMA_MAX_TRANSACTIONS     20
#define DMA_TRANSACTION_SIZE     4194304 //info: can correspond to MAX_BTT
#define MAX_DMA_TAGS 16

//******************************
//**  XCC Operations          **
//******************************
//Housekeeping
#define XCCL_CONFIG         0
//Primitives
#define XCCL_COPY           1
#define XCCL_COMBINE        2
#define XCCL_SEND           3 
#define XCCL_RECV           4
//Collectives
#define XCCL_BCAST          5
#define XCCL_SCATTER        6
#define XCCL_GATHER         7
#define XCCL_REDUCE         8
#define XCCL_ALLGATHER      9
#define XCCL_ALLREDUCE      10
#define XCCL_REDUCE_SCATTER 11

//XCCL_CONFIG SUBFUNCTIONS
#define HOUSEKEEP_IRQEN           0
#define HOUSEKEEP_IRQDIS          1
#define HOUSEKEEP_SWRST           2
#define HOUSEKEEP_PKTEN           3
#define HOUSEKEEP_TIMEOUT         4
#define OPEN_PORT                 5
#define OPEN_CON                  6
#define SET_STACK_TYPE            7
#define START_PROFILING           8
#define END_PROFILING             9
#define SET_DMA_TRANSACTION_SIZE  10
#define SET_MAX_DMA_TRANSACTIONS  11

//AXI MMAP address
#define CONTROL_OFFSET          0x0000
#define ARG00_OFFSET            0x0010
#define AXI00_PTR0_OFFSET       0x0018
#define AXI01_PTR0_OFFSET       0x0024
#define END_OF_REG_OFFSET       0x0030
#define TIME_TO_ACCESS_EXCH_MEM 0x1FF4
#define HWID_OFFSET       0x1FF8
#define RETVAL_OFFSET     0x1FFC
#define END_OF_EXCHMEM    0x2000

#ifndef ACCL_BD_SIM
#define HOSTCTRL_BASEADDR     0x0        
#define EXCHMEM_BASEADDR      0x1000
#define NET_RXPKT_BASEADDR    0x30000
#define NET_TXPKT_BASEADDR    0x40000
#define GPIO_BASEADDR         0x40000000
#define GPIO_TDEST_BASEADDR   0x40010000
#define SWITCH_BASEADDR       0x44A00000
#define IRQCTRL_BASEADDR      0x44A10000
#define TIMER_BASEADDR        0x44A20000
#else
#define HOSTCTRL_BASEADDR     0x0        
#define EXCHMEM_BASEADDR      0x1000
#define NET_RXPKT_BASEADDR    0x3000
#define NET_TXPKT_BASEADDR    0x4000
#define GPIO_BASEADDR         0x7000
#define GPIO_TDEST_BASEADDR   0x8000
#define SWITCH_BASEADDR       0x9000
#define IRQCTRL_BASEADDR      0xA000
#define TIMER_BASEADDR        0xB000
#endif

//https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/managing_interface_synthesis.html#tzw1539734223235
#define CONTROL_START_MASK      0x00000001
#define CONTROL_DONE_MASK       0x00000001 << 1
#define CONTROL_IDLE_MASK       0x00000001 << 2 
#define CONTROL_READY_MASK      0x00000001 << 3
#define CONTROL_REPEAT_MASK     0x00000001 << 7

#define GPIO_DATA_REG      GPIO_BASEADDR + 0x0000
#define GPIO2_DATA_REG     GPIO_BASEADDR + 0x0008
#define GPIO_READY_MASK       0x00000001
#define GPIO_SWRST_MASK       0x00000002

//from AXI interrupt controller v4.1 LogiCORE IP product guide
//https://www.xilinx.com/support/documentation/ip_documentation/axi_intc/v4_1/pg099-axi-intc.pdf
#define IRQCTRL_ISR_OFFSET   0x000
#define IRQCTRL_IPR_OFFSET   0x004
#define IRQCTRL_IER_OFFSET   0x008
#define IRQCTRL_IAR_OFFSET   0x00C
#define IRQCTRL_SIE_OFFSET   0x010
#define IRQCTRL_CIE_OFFSET   0x014
#define IRQCTRL_IVR_OFFSET   0x018
#define IRQCTRL_MER_OFFSET   0x01C
#define IRQCTRL_IMR_OFFSET   0x020
#define IRQCTRL_ILR_OFFSET   0x024
#define IRQCTRL_IVAR_OFFSET  0x100
#define IRQCTRL_IVEAR_OFFSET 0x200
#define IRQCTRL_SIZE         0x300

#define IRQCTRL_TIMER_ENABLE             0x01
#define IRQCTRL_DMA0_CMD_QUEUE_EMPTY     0x02
#define IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY 0x04
#define IRQCTRL_DMA2_CMD_QUEUE_EMPTY     0x08
#define IRQCTRL_DMA2_STS_QUEUE_NON_EMPTY 0x10

#define IRQCTRL_MER_HARDWARE_INTERRUPT_ENABLE 0x2
#define IRQCTRL_MER_MASTER_ENABLE             0x1

//from AXI Timer v2.0 LogiCORE IP Product Guide
//https://www.xilinx.com/support/documentation/ip_documentation/axi_timer/v2_0/pg079-axi-timer.pdf
#define TIMER_CSR0_OFFSET 0x00
#define TIMER_LR0_OFFSET  0x04
#define TIMER_CR0_OFFSET  0x08
#define TIMER_CSR1_OFFSET 0x10
#define TIMER_LR1_OFFSET  0x14
#define TIMER_CR1_OFFSET  0x18
#define TIMER_SIZE        0x20
//TIMERX_CONTROL_AND_STATUS_REGISTER
#define TIMER_CSR_CASC_MASK              1 << 11
#define TIMER_CSR_ENABLE_ALL_MASK        1 << 10 
#define TIMER_CSR_PWMX_ENABLE_MASK       1 << 9 
#define TIMER_CSR_INTERRUPT_MASK         1 << 8 
#define TIMER_CSR_ENABLE_MASK            1 << 7 
#define TIMER_CSR_INTERRUPT_ENABLE_MASK  1 << 6
#define TIMER_CSR_LOAD_TIMER_MASK        1 << 5
#define TIMER_CSR_AUTO_RELOAD_TIMER_MASK 1 << 4
#define TIMER_CSR_CAPTURE_MASK           1 << 3
#define TIMER_CSR_GENERATE_MASK          1 << 2
#define TIMER_CSR_UP_DOWN_MASK           1 << 1
#define TIMER_CSR_MODE_TIMER_MASK        1 << 0

//EXCEPTIONS
#define COLLECTIVE_OP_SUCCESS                         0    
#define DMA_MISMATCH_ERROR                            1     
#define DMA_INTERNAL_ERROR                            2     
#define DMA_DECODE_ERROR                              3  
#define DMA_SLAVE_ERROR                               4 
#define DMA_NOT_OKAY_ERROR                            5     
#define DMA_NOT_END_OF_PACKET_ERROR                   6             
#define DMA_NOT_EXPECTED_BTT_ERROR                    7
#define DMA_TIMEOUT_ERROR                             8             
#define CONFIG_SWITCH_ERROR                           9
#define DEQUEUE_BUFFER_TIMEOUT_ERROR                  10
#define DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR      11
#define RECEIVE_TIMEOUT_ERROR                         12
#define DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH   13
#define DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR       14
#define COLLECTIVE_NOT_IMPLEMENTED                    15
#define RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID       16
#define OPEN_PORT_NOT_SUCCEEDED                       17
#define OPEN_CON_NOT_SUCCEEDED                        18
#define DMA_SIZE_ERROR                                19
#define ARITH_ERROR                                   20 
#define PACK_TIMEOUT_STS_ERROR                        21
#define PACK_SEQ_NUMBER_ERROR                         22
#define COMPRESSION_ERROR                             23
#define KRNL_TIMEOUT_STS_ERROR                        24
#define KRNL_STS_COUNT_ERROR                          25

//data movement structures and defines
#define USE_NONE   0
#define USE_OP0_DM 1
#define USE_OP1_DM 2
#define USE_RES_DM 4

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
extern uint32_t *cfgmem;
#define Xil_Out32(offset, value) {cfgmem[(offset)/4] = (value);}
#define Xil_In32(offset) ({uint32_t value = cfgmem[(offset)/4]; value; })

#define SET(offset, mask) Xil_Out32(offset, Xil_In32(offset) | (mask))
#define CLR(offset, mask) Xil_Out32(offset, Xil_In32(offset) & ~(mask))

//stream handling
#ifndef ACCL_BD_SIM
//push data to stream
#define putd(channel, value) asm volatile ("putd\t%0,%1" :: "d" (value), "d" (channel))
//push data to stream and set control bit
#define cputd(channel, value) asm volatile ("cputd\t%0,%1" :: "d" (value), "d" (channel))
//test-only non-blocking get from stream channel
#define tngetd(channel) ({unsigned int value; asm volatile ("tngetd\tr0,%1\n\taddic\t%0,r0,0" : "=d" (value) : "d" (channel)); value; })
//blocking get from stream channel
#define getd(channel) ({unsigned int value; asm volatile ("getd\t%0,%1" : "=d" (value) : "d" (channel)); value; })

#else

#include "Stream.h"
#include "Axi.h"
extern hlslib::Stream<hlslib::axi::Stream<ap_uint<32> >, 512> cmd_fifos[9];
extern hlslib::Stream<hlslib::axi::Stream<ap_uint<32> >, 512> sts_fifos[10];
extern sem_t mb_irq_mutex;

//push data to stream
#define putd(channel, value) cmd_fifos[channel].Push(hlslib::axi::Stream<ap_uint<32> >(value, 0))
//push data to stream and set control bit
#define cputd(channel, value) cmd_fifos[channel].Push(hlslib::axi::Stream<ap_uint<32> >(value, 1))
//test-only non-blocking get from stream channel
#define tngetd(channel) sts_fifos[channel].IsEmpty()
//blocking get from stream channel
#define getd(channel) sts_fifos[channel].Pop().data

#endif

typedef struct {
    unsigned int addrl;
    unsigned int addrh;
    unsigned int max_len;
    unsigned int dma_tag;
    unsigned int status;
    unsigned int rx_tag;
    unsigned int rx_len;
    unsigned int rx_src;
    unsigned int sequence_number;
} rx_buffer;
#define STATUS_IDLE     0x00
#define STATUS_ENQUEUED 0x01
#define STATUS_RESERVED 0x02

#define RX_BUFFER_COUNT_OFFSET 0x1000

#define COMM_OFFSET (RX_BUFFER_COUNT_OFFSET+4*(1 + Xil_In32(RX_BUFFER_COUNT_OFFSET)*9))

typedef struct {
    unsigned int ip;
    unsigned int port;
    unsigned int inbound_seq;
    unsigned int outbound_seq;
    unsigned int session;
} comm_rank;

typedef struct {
    unsigned int size;
    unsigned int local_rank;
    comm_rank* ranks;
} communicator;

//structure defining arithmetic config parameters
//TODO: make unsigned char to save on space
#define MAX_REDUCE_FUNCTIONS 10

typedef struct {
    unsigned int uncompressed_elem_bits;//bitwidth of one element of uncompressed data
    unsigned int compressed_elem_bits;  //bitwidth of one element of compressed data
    unsigned int elem_ratio;            //how many uncompressed elements per compressed element
    unsigned int compressor_tdest;      //clane TDEST for targeting the compressor
    unsigned int decompressor_tdest;    //clane TDEST for targeting the compressor
    unsigned int arith_nfunctions;      //number of supported functions (<= MAX_REDUCE_FUNCTIONS)
    unsigned int arith_is_compressed;   //perform arithmetic on compressed (1) or uncompressed (0) values
    unsigned int arith_tdest[MAX_REDUCE_FUNCTIONS]; //arithmetic TDEST
} datapath_arith_config;

//define compression flags
#define NO_COMPRESSION 0
#define OP0_COMPRESSED 1
#define OP1_COMPRESSED 2
#define RES_COMPRESSED 4
#define ETH_COMPRESSED 8

//define stream flags
#define NO_STREAM  0
#define OP0_STREAM 1
#define RES_STREAM 2

//define remote flags
#define NO_REMOTE 0
#define RES_REMOTE 1

//Tag definitions
#define TAG_ANY 0xFFFFFFFF

//circular buffer
#define MAX_CIRCULAR_BUFFER_SIZE 20
typedef struct circular_buffer
{
    unsigned int buffer[MAX_CIRCULAR_BUFFER_SIZE];     // data buffer
    unsigned int capacity;      //real desired capacity of buffer
    unsigned int occupancy;     //current occupancy
    unsigned int write_idx;          // where we put data
    unsigned int read_idx;          // where we get data from
} circular_buffer;

//define exception handling for simulation
#ifdef ACCL_BD_SIM
#define setjmp(x) 0
void finalize_call(unsigned int);
#define longjmp(x, y) finalize_call(y)
#endif

#ifdef __cplusplus
extern "C" int run_accl();
extern "C" void stream_isr();
#endif

#endif // _CCL_OFFLOAD_CONTROL_H_
