# Host ctrl address space explanation and details
the host_ctrl address space is used as a mailbox and divided in 2 regions:
-   0x000-0x800   follows ap_ctrl_hs;  
-   0x1000-0x2000 is to exchange spare buffers and communicators.

## 0x0000-0x0800 control region
0x000-0x800 should be similar to this:
````
0x1820000 ['0x4', '0x0', '0x0' , '0x0']
0x1820010 ['0x2', '0x0', '0x3d0900', '0x0']
0x1820020 ['0x1244', '0x0', '0x2', '0x0']
0x1820030 ['0x0', '0x0', '0x19', '0x0']
0x1820040 ['0x0', '0x0', '0x0', '0x0']
0x1820050 ['0x0', '0x0', '0xd0b73000', '0x0']
0x1820060['0x0', '0x0', '0x0', '0x0']
0x1820070 ['0x0', '0x0', '0x0', '0x0']
````

first 4 32b int are for the ctrl of [hls_ip](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/managing_interface_synthesis.html#:~:text=r1%3D*((float*)%26u[…]ardware,-In%20this%20example) and then  15 x 32b integers for CCLO parameters each parameter is followed by a 32 bit integer that is reserved. 
If we parse that memory from 0x10 offset(i.e. 0x1820010) you have in order:
 addr      |parameter       | value
-----------|----------------|------
 0x1820010 |scenario        |    2
 0x1820018 |len             |    4*10^6 = 0x3d0900
 0x1820020 |comm_addr       |    0x1244
 0x1820028 |root_src_dst    |    0x2
 0x1820030 |function        |    0
 0x1820038 |msg_tag         |    19 = 25
 0x1820040 |buf0_type       |    0 
 0x1820048 |buf1_type       |    0
 0x1820050 |buf2_type       |    0
(note that from now on we have 64 bit addr so they go in two consecutive 32 bit register with 1 32 bit integer reserved as a trailer  ) | |
 0x1820058 | buf0_addrl      |    0xd0b73000  
 0x182005c | buf0_addrh      |    0
 0x1820064 | buf1_addrl      |    0
 0x1820068 | buf1_addrh      |    0
 0x1820070 | buf2_addrl      |    0
 0x1820074 | buf2_addrh      |    0

## 0x1000-0x1FF4
This region is used to communicate data structures with MB.
You can took at [cclo_offload.h](../kernel/fw/sw_apps/ccl_offload_control/src/ccl_offload_control.h) for more info.

These are the data structure used.
````
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
#define STATUS_ERROR    0x04

#define RX_BUFFER_COUNT_OFFSET 0x1000

#define COMM_OFFSET (RX_BUFFER_COUNT_OFFSET+4*(1 + Xil_In32(RX_BUFFER_COUNT_OFFSET)*9))

typedef struct {
	unsigned int size;
	unsigned int local_rank;
	comm_rank [size];
} communicator;

typedef struct {
	unsigned int ip;
	unsigned int port;
  unsigned int inbound_seq;
  unsigned int outbound_seq;
  unsigned int session;
} comm_rank;
````
The memory is filled as follow
1. first 32 bit integer contains the number of spare buffers allocated (RX_BUFFER_COUNT)
2. follows RX_BUFFER_COUNT ``rx_buffer`` struct
3. then we have a ``communicator`` struct
    1. inside the communicator we have ``communicator.size`` x ``comm_rank`` structs
## 0x1FF8 HWID

we have a an id to identify the hw is use

## 0x1FFF RETVAL

we have a 32 bit integer that specifies the return code for the collective. You can find all of them in [cclo_offload.h](../kernel/fw/sw_apps/ccl_offload_control/src/ccl_offload_control.h).

````
//EXCEPTIONS
#define COLLECTIVE_OP_SUCCESS                         0    
#define DMA_MISMATCH_ERROR                            1     
#define DMA_INTERNAL_ERROR                            2     
...
````
