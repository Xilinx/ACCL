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

#ifndef _CCL_OFFLOAD_CONTROL_H_
#define _CCL_OFFLOAD_CONTROL_H_

#ifdef __cplusplus
extern "C" {
#endif

#define KERNEL_NAME    "ccl_offload"
#define KERNEL_VENDOR  "Xilinx"
#define KERNEL_LIBRARY "ACCL"

// #define NUM_M_AXI      3
// #define NUM_CLOCKS     1
// #define NUM_INPUT_ARGS 1
// #define NUM_CHAN       2
// #define NUM_STREAMS    5

//AXIS interfaces to/from MB 

#define CMD_DMA0_TX  0
#define CMD_DMA0_RX  1
#define CMD_DMA1_TX  2
#define CMD_DMA1_RX  3
#define CMD_DMA2_TX  4
#define CMD_DMA2_RX  5
#define CMD_UDP_TX   6
#define CMD_TCP_PORT 7
#define CMD_TCP_CON  8
#define CMD_TCP_TX   9
#define CMD_HOST    10

#define STS_DMA0_RX  0
#define STS_DMA0_TX  1
#define STS_DMA1_RX  2
#define STS_DMA1_TX  3
#define STS_DMA2_RX  4
#define STS_DMA2_TX  5
#define STS_UDP_RX   6
#define STS_TCP_PORT 7
#define STS_TCP_CON  8
#define STS_TCP_RX   9
#define STS_HOST     10 

//MAIN SWITCH

#define DATAPATH_DMA_LOOPBACK      1
#define DATAPATH_DMA_REDUCTION     2
#define DATAPATH_OFFCHIP_TX_UDP    3
#define DATAPATH_OFFCHIP_TX_TCP    4
#define DATAPATH_OFFCHIP_UDP_REDUCTION 5
#define DATAPATH_OFFCHIP_TCP_REDUCTION 6
#define DATAPATH_DMA_EXT_LOOPBACK  7
//#define DATAPATH_OFFCHIP_RX_UDP    8 not used up to now since DMA are physically linked to depacketizer
//#define DATAPATH_OFFCHIP_RX_TCP    9 not used up to now since DMA are physically linked to depacketizer

#define MAIN_SWITCH_M_UDP_TX    0
#define MAIN_SWITCH_M_TCP_TX    1
#define MAIN_SWITCH_M_DMA1_TX   2
#define MAIN_SWITCH_M_ARITH_OP0 3
#define MAIN_SWITCH_M_ARITH_OP1 4
#define MAIN_SWITCH_M_EXT_KRNL  5

#define MAIN_SWITCH_S_DMA0_RX   0
#define MAIN_SWITCH_S_DMA1_RX   1
#define MAIN_SWITCH_S_DMA2_RX   2
#define MAIN_SWITCH_S_ARITH_RES 3
#define MAIN_SWITCH_S_EXT_KRNL  4

//ARITH SWITCH
#define ARITH_INTERNAL 0
#define ARITH_EXTERNAL 1

#define ARITH_SWITCH_M_INT_OP0 0
#define ARITH_SWITCH_M_INT_OP1 1
#define ARITH_SWITCH_M_EXT_OP0 2
#define ARITH_SWITCH_M_EXT_OP1 3
#define ARITH_SWITCH_M_RES     4

#define ARITH_SWITCH_S_OP0     0
#define ARITH_SWITCH_S_OP1     1
#define ARITH_SWITCH_S_INT_RES 2
#define ARITH_SWITCH_S_EXT_RES 3

//PACKT CONST
#define MAX_PACKETSIZE 2048
//DMA CONST 
#define MAX_BTT        0x7FFFFF

//******************************
//**  ACCL Primitives         **
//******************************
#define ACCL_CONFIG         0
#define ACCL_SEND           1 
#define ACCL_RECV           2
#define ACCL_BCAST          3
#define ACCL_SCATTER        4
#define ACCL_GATHER         5
#define ACCL_REDUCE         6
#define ACCL_ALLGATHER      7
#define ACCL_ALLREDUCE      8
#define ACCL_ACC            9
#define ACCL_COPY           10
#define ACCL_REDUCE_RING    11
#define ACCL_ALLREDUCE_FUSED_RING 12
#define ACCL_GATHER_RING    13
#define ACCL_ALLGATHER_RING 14
#define ACCL_EXT_STREAM_KRNL 15
#define ACCL_EXT_REDUCE     16

//ACCL_CONFIG SUBFUNCTIONS
#define HOUSEKEEP_IRQEN   0
#define HOUSEKEEP_IRQDIS  1
#define HOUSEKEEP_SWRST   2
#define HOUSEKEEP_PKTEN   3
#define HOUSEKEEP_TIMEOUT 4
#define INIT_CONNECTION   5 
#define OPEN_PORT         6
#define OPEN_CON          7
#define USE_TCP_STACK     8
#define USE_UDP_STACK     9
#define START_PROFILING   10
#define END_PROFILING     11

//AXI MMAP address
#define CONTROL_OFFSET    0x0000
#define HWID_OFFSET       0x0FF8
#define RETVAL_OFFSET     0x0FFC
#define ARG00_OFFSET      0x0010
#define AXI00_PTR0_OFFSET 0x0018
#define AXI01_PTR0_OFFSET 0x0024
#define END_OF_REG_OFFSET 0x0030
#define END_OF_EXCHMEM    0x1000

#define HOSTCTRL_BASEADDR     0x0        
#define EXCHMEM_BASEADDR      0x0800
#define ARITH_BASEADDR        0x20000
#define UDP_RXPKT_BASEADDR    0x30000
#define UDP_TXPKT_BASEADDR    0x40000
#define TCP_RXPKT_BASEADDR    0x50000
#define TCP_TXPKT_BASEADDR    0x60000
#define GPIO_BASEADDR         0x40000000
#define MAIN_SWITCH_BASEADDR  0x44A00000
#define IRQCTRL_BASEADDR      0x44A10000
#define TIMER_BASEADDR        0x44A20000
#define ARITH_SWITCH_BASEADDR 0x44B00000

#define CONTROL_START_MASK  0x00000001
#define CONTROL_DONE_MASK   0x00000002
#define CONTROL_IDLE_MASK   0x00000004
#define CONTROL_REPEAT_MASK 0x00000080

#define GPIO_DATA_REG      XPAR_GPIO_0_BASEADDR + 0x0000
#define GPIO_READY_MASK    0x00000001
#define GPIO_SWRST_MASK    0x00000002

#define GPIO_TRI_REG       XPAR_GPIO_0_BASEADDR + 0x0004

#define GPIO2_DATA_REG     XPAR_GPIO_0_BASEADDR + 0x0008
#define STREAM_INTR_MASK   0x00000001

#define GPIO2_TRI_REG      XPAR_GPIO_0_BASEADDR + 0x000C

#define GPIO_GIER_REG      XPAR_GPIO_0_BASEADDR + 0x011C
#define GIE_MASK           0x80000000

#define GPIO_IPISR_REG     XPAR_GPIO_0_BASEADDR + 0x0120
#define C1IS_MASK          0x00000001
#define C2IS_MASK          0x00000002
#define GPIO_IPIER_REG     XPAR_GPIO_0_BASEADDR + 0x0128
#define C1IE_MASK          0x00000001
#define C2IE_MASK          0x00000002

//from AXI interrupt controller v4.1 LogiCORE IP product guide
//https://www.xilinx.com/support/documentation/ip_documentation/axi_intc/v4_1/pg099-axi-intc.pdf
#define IRQCTRL_INTERRUPT_STATUS_REGISTER_OFFSET                  0x00
#define IRQCTRL_INTERRUPT_PENDING_REGISTER_OFFSET                 0x04
#define IRQCTRL_INTERRUPT_ENABLE_REGISTER_OFFSET                  0x08
#define IRQCTRL_INTERRUPT_ACKNOWLEDGE_REGISTER_OFFSET             0x0C
#define IRQCTRL_SET_INTERRUPT_ENABLES_OFFSET                      0x10
#define IRQCTRL_CLEAR_INTERRUPT_ENABLES_OFFSET                    0x14
#define IRQCTRL_INTERRUPT_VECTOR_REGISTER_OFFSET                  0x18
#define IRQCTRL_MASTER_ENABLE_REGISTER_OFFSET                     0x1C
#define IRQCTRL_INTERRUPT_MODE_REGISTER_OFFSET                    0x20
#define IRQCTRL_INTERRUPT_LEVEL_REGISTER_OFFSET                   0x24
#define IRQCTRL_INTERRUPT_VECTOR_ADDRESS_REGISTER_OFFSET          0x100
#define IRQCTRL_INTERRUPT_VECTOR_EXTENDED_ADDRESS_REGISTER_OFFSET 0x200
#define IRQCTRL_SIZE 0x300

#define IRQCTRL_TIMER_ENABLE                   0x0001
#define IRQCTRL_DMA0_CMD_QUEUE_EMPTY           0x0002
#define IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY       0x0004
#define IRQCTRL_DMA2_CMD_QUEUE_EMPTY           0x0008
#define IRQCTRL_DMA2_STS_QUEUE_NON_EMPTY       0x0010

#define IRQCTRL_MASTER_ENABLE_REGISTER_HARDWARE_INTERRUPT_ENABLE 0x0002
#define IRQCTRL_MASTER_ENABLE_REGISTER_MASTER_ENABLE 0x0001

//from AXI Timer v2.0 LogiCORE IP Product Guide
//https://www.xilinx.com/support/documentation/ip_documentation/axi_timer/v2_0/pg079-axi-timer.pdf
#define TIMER0_CONTROL_AND_STATUS_REGISTER_OFFSET                 0x00
#define TIMER0_LOAD_REGISTER_OFFSET                               0x04
#define TIMER0_COUNTER_REGISTER_OFFSET                            0x08
#define TIMER1_CONTROL_AND_STATUS_REGISTER_OFFSET                 0x10
#define TIMER1_LOAD_REGISTER_OFFSET                               0x14
#define TIMER1_COUNTER_REGISTER_OFFSET                            0x18
#define TIMER_SIZE                                                0x20
//TIMERX_CONTROL_AND_STATUS_REGISTER
#define CONTROL_AND_STATUS_REGISTER_CASC_MASK                     0x0001 <<11
#define CONTROL_AND_STATUS_REGISTER_ENABLE_ALL_MASK               0x0001 <<10 
#define CONTROL_AND_STATUS_REGISTER_PWMX_ENABLE_MASK              0x0001 << 9 
#define CONTROL_AND_STATUS_REGISTER_INTERRUPT_MASK                0x0001 << 8 
#define CONTROL_AND_STATUS_REGISTER_ENABLE_MASK                   0x0001 << 7 
#define CONTROL_AND_STATUS_REGISTER_INTERRUPT_ENABLE_MASK         0x0001 << 6
#define CONTROL_AND_STATUS_REGISTER_LOAD_TIMER_MASK               0x0001 << 5
#define CONTROL_AND_STATUS_REGISTER_AUTO_RELOAD_TIMER_MASK        0x0001 << 4
#define CONTROL_AND_STATUS_REGISTER_CAPTURE_MASK                  0x0001 << 3
#define CONTROL_AND_STATUS_REGISTER_GENERATE_MASK                 0x0001 << 2
#define CONTROL_AND_STATUS_REGISTER_UP_DOWN_MASK                  0x0001 << 1
#define CONTROL_AND_STATUS_REGISTER_MODE_TIMER_MASK               0x0001 << 0

//EXCEPTIONS
#define COLLECTIVE_OP_SUCCESS       0    
#define DMA_MISMATCH_ERROR          1     
#define DMA_INTERNAL_ERROR          2     
#define DMA_DECODE_ERROR            3  
#define DMA_SLAVE_ERROR             4 
#define DMA_NOT_OKAY_ERROR          5     
#define DMA_NOT_END_OF_PACKET_ERROR 6             
#define DMA_NOT_EXPECTED_BTT_ERROR  7
#define DMA_TIMEOUT_ERROR           8             
#define CONFIG_SWITCH_ERROR         9
#define DEQUEUE_BUFFER_TIMEOUT_ERROR 10
#define DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR 11
#define RECEIVE_TIMEOUT_ERROR       12
#define DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH             13
#define DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR 14
#define COLLECTIVE_NOT_IMPLEMENTED 15
#define RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID 16
#define OPEN_PORT_NOT_SUCCEEDED    17
#define OPEN_COM_NOT_SUCCEEDED     18

#define S_AXI_CONTROL -1

#define Xil_Out32(offset, value) *(volatile unsigned int *)(offset) = (value)
#define Xil_In32(offset) ({unsigned int value = *(volatile unsigned int *)(offset); value; })
#define SET(offset, mask) Xil_Out32(offset, Xil_In32(offset) | (mask))
#define CLR(offset, mask) Xil_Out32(offset, Xil_In32(offset) & ~(mask))

//push data to stream
#define putd(channel, value) asm volatile ("putd\t%0,%1" :: "d" (value), "d" (channel))
//push data to stream and set control bit
#define cputd(channel, value) asm volatile ("cputd\t%0,%1" :: "d" (value), "d" (channel))
//test-only non-blocking get from stream channel
#define tngetd(channel) ({unsigned int value; asm volatile ("tngetd\tr0,%1\n\taddic\t%0,r0,0" : "=d" (value) : "d" (channel)); value; })
//blocking get from stream channel
#define getd(channel) ({unsigned int value; asm volatile ("getd\t%0,%1" : "=d" (value) : "d" (channel)); value; })

typedef enum {
  type_control,
  type_int,
  type_bool,
  type_char,
  type_uchar,
  type_short,
  type_ushort,
  type_uint,
  type_long,
  type_ulong, 
  type_float,
  type_double,
  type_half,
  type_intptr,
} type_t;

typedef struct {
  char   *name;
  type_t type;
  int    interface;
} arg_t;

// Total number of arguments
static const int argc = 4;

// Argument name, type, and interface
static const arg_t argv[] = {
  { "Control", type_control, S_AXI_CONTROL },
  { "bar", type_uint, S_AXI_CONTROL },
  { "m_axi_0", type_intptr, 0 },
  { "m_axi_1", type_intptr, 1 }
};

// Number of arguments for each interface
static const unsigned int num_args[] = {
  1, 1
};

// Offset to first argument for each interface
static const unsigned int offsets[] = {
  0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50, 0x5c
};

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

#define RX_BUFFER_COUNT_OFFSET 0x800

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

#define TAG_ANY 0xFFFFFFFF


#ifdef __cplusplus
}
#endif
#endif // _CCL_OFFLOAD_CONTROL_H_
