# Table of contents:
- [Software interface](#Software-interface)
 - [Kernel Parameters (BASEADDR+0)](#Kernel-Parameters-(BASEADDR+0));
 - [RX Buffers (BASEADDR+0x800)](#RX-Buffers-(BASEADDR+0x800));
 - [Communicators (BASEADDR+0x800 + 4*(1 + buf_count*9))](#Communicators-(BASEADDR+0x800-+-4*(1-+-buf_count*9)));
 - [Miscellaneous](#Miscellaneous);
- [Drivers](#Drivers)
## Software interface

The CCLO communicate with the host through a 8kB IO space which starts at ``BASEADDR`` and is implemented by the ``s_axi_control`` port.
The IO space is divided into 4 sections:

1. Control and arguments section, from address 0 to 0x7FF - this is physically implemented in the kernel with a HLS ip (the ``hostcrtl`` sources here [../kernel/hls/hostcrtl](../kernel/hls/hostcrtl)) whose sole purpose is to gather arguments, forward them to the ``CTRL`` module, and signal completion to the driver. Info at [HW Kernel Parameters (BASEADDR+0)](#HW-Kernel-Parameters-(BASEADDR+0));
2. RX buffers configuration,  from address 0x800 to (0x800 + 4*(1 + rxbuf_count*9)). This memory is contained in the [Exchange memory module](#exchange-memory-module) and is used to store information about ``temporary_buffers``. More info at [RX Buffers (BASEADDR+0x800)](#RX-Buffers-(BASEADDR+0x800));
3. Communication configuration, from address (0x800 + 4*(1 + buf_count*9)). This memory is contained in the [Exchange memory module](#exchange-memory-module). More info at [Communicators (BASEADDR+0x800 + 4*(1 + buf_count*9))](#Communicators-(BASEADDR+0x800-+-4*(1-+-buf_count*9)));
4. Miscellaneous section, from address (0x0FF8 to 0x0FFF). This memory is contained in the [Exchange memory module](#exchange-memory-module). More info at [Miscellaneous](#Miscellaneous).


  ## Kernel Parameters (BASEADDR+0)
  first 4 32b int are for the ctrl of [hls_ip](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/managing_interface_synthesis.html#:~:text=r1%3D*((float*)%26u[…]ardware,-In%20this%20example) and then  15 x 32b integers for CCLO parameters each parameter is followed by a 32 bit integer that is reserved. 

name            | size | offset | type                   
----------------|------|--------|-----------
 call_type      | 4    |0x010   |uint                       
 byte_count     | 4    |0x018   |uint                       
 comm_addr      | 4    |0x020   |uint                       
 root_src_dst   | 4    |0x028   |uint                       
 function       | 4    |0x030   |uint                       
 tag            | 4    |0x038   |uint                       
 buf0_type      | 4    |0x040   |uint                       
 buf1_type      | 4    |0x048   |uint                       
 buf2_type      | 4    |0x050   |uint                       
 buf0_ptr_addrl | 4    |0x058   |uint                       
 buf0_ptr_addrh | 4    |0x05c   |uint                       
 buf1_ptr_addrh | 4    |0x064   |uint                       
 buf1_ptr_addrl | 4    |0x068   |uint                       
 buf2_ptr_addrl | 4    |0x070   |uint                       
 buf2_ptr_addrh | 4    |0x074   |uint                       

Example
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

If the base address is 0x1820000, and we can parse the memory from 0x10 offset to get the paremters. So you have:

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
 0x1820058 | buf0_addrl      |    0xd0b73000  
 0x182005c | buf0_addrh      |    0
 0x1820064 | buf1_addrl      |    0
 0x1820068 | buf1_addrh      |    0
 0x1820070 | buf2_addrl      |    0
 0x1820074 | buf2_addrh      |    0
  
  Here follows a small description:
  - ``call_type``: (a.k.a. ``scenario``) since a single kernel will implement the offloading functions (send, receive, gather, scatter, etc..) we need one parameter to differentiate between the operation requested`. Inside each operation/collective we the user may specify a different subfunction to execute via ``reduce_op``. 
	Here follows a some of ``call_type`` (a.k.a. ``scenario``) values  [cclo_offload.h](../kernel/fw/sw_apps/ccl_offload_control/src/ccl_offload_control.h):

	| Scenario		  	  | value |
	|-------------------|-------|
	| XCCL_CONFIG       |	0	    |
	| XCCL_SEND			    |	1	    |
	| XCCL_RECV			    |	2	    |
	| ...               | ...   |

  -  ``byte_count`` that the user wants to use in the collective;
  - the address of the MPI [``communicator``](https://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/) the user is willing to use. In this way the CCLO can support multiple MPI communicator at the same time;
  - ``source/dest/root`` can share an integer parameter. Depending on the collective requested it may indicate the root, the destination or the source;
  - ``reduce op`` specifies better what kind of operation is requested. (E.g. in case of a reduce operation the user can specify 4 different modes 0: float, 1: double, 2: int, 3: long);
  - ``tag``. The tag identifies identifiers is used to determine the recipients of the MPI message. A special tag value of 0xFFFFFFFF is used to address all;
  - ``memory buffers``  the user must be indicate explicitly in any call (send/recv buf)  the buffers on which the collective has to be executed. There can be at most two buffers.
 

  ## RX Buffers (BASEADDR+0x800)
  -------

  Beyond the parameters is the RX buffer definition space, consisting of:
  - one 32-bit integer (``rx_buffer_count`` from now on) containing the number of RX buffers available for the CCLO to use.
  - ``rx_buffer_count`` buffer definitions. Each buffer is defined by the following struct:

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
    ````

  The ``addrl`` and ``addrh`` points in a region of the DDR where data coming from the network stack is accumulated waiting for the user to request. In this way it is possible for the MPI commands across the XCCL_cores do not have to be tightly synchronized. So these staging areas, called `rx_sparse_buffers`, which do not need to be contiguous, are used to keep data that the user has not yet claimed. The number of `rx_sparse_buffers` and their location is configured by the [cclo driver function  `def setup_rx_buffers(self, nbufs, bufsize, devicemem)`](../driver/python/cclo.py#L87). Thus, it is important that the message to be transported doesn't exceed in size the ``rx_sparse_buffers`` unless bytes of it may be dropped.

  In case the user doesn't claim the data are they overwritten? NO

  In the moment in which the user recalls the receive driver function, or any other MPI primitive the CCL_Offload will move data from the ``rx_sparse_buffers`` to any user specified memory location.

  ## Communicators (BASEADDR+0x800 + 4*(1 + buf_count*9))
  -------
  A communicator is built from ranks.
  The ranks are nothing else than a dictionary that describes all the peers in the ACCL_collectives communications. So the ranks are identical in each CCL_Offload instance.

1. each entry in the dictionary is a couple ip, port
2. the local rank identifies the local CCLO. In practical terms the local rank points to the entry in the ranks dictionary.
3. an id inside MPI/XCCL_collectives refers to position inside ranks data structure.
  The communicator that describe all those information is placed in memory so that it is accessible from the CCL_Offload kernel, so that is possible to locate the other CCL_Offload instances.

  The communicator space holds instances of communicator data structure. 
  Each communicator consists of:
  - number of ranks (``world_size``);
  - local rank, an index ( <= ``world_size`` -1 ) that indicates one of the entry in the following list;
  - ``world_size `` entries. These specifies the peers involved in the communication. The order in the list reflects the rank. Each entry is composed of the following struct

    ````
    typedef struct {
    unsigned int ip;
    unsigned int port;
    unsigned int inbound_seq;
    unsigned int outbound_seq;
    unsigned int session;
  } comm_rank;
    ````

   The CCL_Offload can support more than a communicator.
  ## Miscellaneous
  -------
  A couple of registers:
  - Hardware ID at BASEADDR + 0x0FF8 (4B) , it is used to detect which version of the CCL_Offload is in use (e.g. if VNX, TCP,  ecc..) TODO: specify the encoding.
  - Return value at BASEADDR + 0x0FFC (4B), it is used by the CTRL module to return a relevant numerical result. 
  we have a 32 bit integer that specifies the return code for the collective. You can find all of them in [cclo_offload.h](../kernel/fw/sw_apps/ccl_offload_control/src/ccl_offload_control.h). It may be used to specify errors.

  ````
  //EXCEPTIONS
  #define COLLECTIVE_OP_SUCCESS                         0    
  #define DMA_MISMATCH_ERROR                            1     
  #define DMA_INTERNAL_ERROR                            2     
  ...
  ````

# Drivers
We have both Python and C++ bindings to abstract 

## Expected flow:

0. As soon as the design is loaded into the FPGA, the ``CTRL`` module initialize itself.
0. The user declares a number of temporary buffers to store in flight packets and their maximal dimension.
    - The driver allocate in DDR the number of temporary buffers that the user requested.
    - The driver then writes the location and the length of those buffer in the [RX Buffers (BASEADDR+0x800)](#rx-buffers-(BASEADDR+0x800)) memory region.
    - Then it determines the start of the [Communicators (BASEADDR+0x800 + 4*(1 + buf_count*9))](#communicators-(BASEADDR+0x800-+-4*(1-+-buf_count*9))) memory region.
    - then enables  ``Microblaze`` interrupts.
    - and starts the ``packetizer`` and the ``depacketizer``.
0. The user describes the communicators and he/she describes the communication world: a list of peers involved in the communication, a poses the peers in a ranked list and identifies itself in the ranked list. The driver then fills the [Communicators (BASEADDR+0x800 + 4*(1 + buf_count*9))](#communicators-(BASEADDR+0x800-+-4*(1-+-buf_count*9))) memory region [source](/../driver/python/cclo.py#L147).
0. The user can now define the user memory buffers (pay attention: their dimension and consequently the maximum dimension of the data transfer  should not exceed the dimensions of the temporary buffers )
0. and then she/he can start requesting to the driver to perform a collective. The driver will :

    - send a call via a write in [HW Kernel Parameters (BASEADDR+0)](#hw-kernel-parameters-(BASEADDR+0)) memory region to the ``host_ctrl`` module specifying all the parameters. 
    - These info are then sent by the ``host_ctrl`` module to the ``MicroBlaze``. 
    - The ``MicroBlaze`` obtains those info and based on the ``call_type`` it performs the requested operation.
    - at the end it will signal finish to the ``host_ctrl``, it will write the return val in the ``BRAM`` in the ``exchange_memory`` module and signal completion via ``host_sts``.

0. At the end of the application the user recalls the driver [``deint`` function](../driver/python/cclo.py#83)
    - The driver asks the ``MicroBlaze`` via the ``host_ctrl`` module to reset its peripherals.
     - The driver asks the ``MicroBlaze`` via the ``host_ctrl`` module to disable interrupts.
