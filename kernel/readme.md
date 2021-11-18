
# Table of Contents:
- [Architecture](#Architecture);
  - [CCL_Offload_kernel](#CCL_Offload_kernel);
  - [CTRL module](#CTRL-module);
  - [Exchange memory module](#Exchange-memory-module);
- [Building and package the ccl_offload.xo Kernel](#building-and-package-the-ccl_offloadxo-kernel)

# Architecture

## CCL_Offload_kernel
![schematic](https://xilinx.github.io/ACCL/images/drawing_offload.svg)

The  CCL  Offload  (CCLO)  kernel  implements  the  ACCL primitives  by  orchestrating  data  movement  between  the  network  fabric,  FPGA  external  memory,  and  FPGA  compute kernels, with no host CPU involvement. 

Data movement to and from  the  network  is  accomplished  through  custom  interface blocks  to  the  TCP/UDP  network  protocol  offload  engines,while  FPGA  external  memory  (DDR  or  HBM)  is  read  and written through DataMover engines (DMA).

Orchestrating the required transfers and the interaction between various subsystems at high speed is a challenge. For this reason, the CCLO kernel consists of two subsystems: a software-programmable control plane which is extremely flexible, and a high-throughput data plane consisting of DMAs, configurable routing,  and  pipelined  arithmetic,  which  is  fast.  Different parts  of  the  data  plane  can  be  activated  at  the  same  time  in parallel, e.g. to transmit and receive data simultaneously from the  network.  


### Data Plane structure

![schematic](https://xilinx.github.io/ACCL/images/drawing_offload_data_plane_with_datapath_simplified.svg)

The Figure presents a  high  level  overview  of  the  CCLO data  plane  structure  which  consists  of  multiple  functional units (FUs) - three AXI DataMover engines (``DMA0``, ``DMA1``,``DMA2``),  AXI  Stream  (``AXIS``)  interconnects  (e.g.  ``S0``,  ``S1``),  an internal arithmetic unit (``A0``) and network interface logic (``UD``,``UP``,``TP``,``TD``). Data flows through all those components via 512bit wide AXIS interfaces [15], that are connected together via the central AXIS Switch.

 Modules description:
  - ``Arithmetic`` unit : CCLO uses either external or internal arithmetic units to performs sums in different way. This module would be used to optimize certain kind of operations like a gather that fuses all the data in an integer, which can be useful in certain cases (e.g. distributed training to perform stochastic gradient descent)
  - [``UP``](hls/vnx_intf/vnx_depacketizer.cpp), ``TP`` are ``Packetizer``s: collects data from the the rest of the system and broke them into smaller pieces that can be sent by the network stack in a single MTU. Those pieces are then sent via AXI Stream to the network stack. The ``CTRL`` module can orchestrate the start/end of procedure via ctrl AXIS interface. The module gets the total number of bytes to transfer, the destination and the tag from ``CTRL`` and prepends them to the outgoing stream.
  - [``UD``](hls/vnx_intf/vnx_packetizer.cpp), ``TD`` are ``Depacketizer``s: they get data  from the network stack and extracts application header from the payload, such as the source and the number of bytes in the packet. Those metadata are relayed to the ``CTRL`` module via ``sts stream``. The payload are directly sent to the ``DMA0`` or ``DMA2`` depending on the protocol.
  - The ``DMA``X modules are [AXI DataMovers](https://www.xilinx.com/support/documentation/ip_documentation/axi_datamover/v5_1/pg022_axi_datamover.pdf), a Xilinx IP that enables high throughput transfer of data between the AXI4 memory-mapped and AXI4-Stream domains. The AXI DataMover provides the MM2S(read interface) and S2MM(write interface) AXI4-Stream channels that operate independently in a full-duplex like method. To be more specific:
    
  - The [AXI Switch is a Xilinx IP](https://www.xilinx.com/support/documentation/ip_documentation/axis_infrastructure_ip_suite/v1_1/pg085-axi4stream-infrastructure.pdf) that allows to exchange AXI_stream packages between the modules and is controlled by the CTRL module.

The CCL_Offload relies extensively on AXI protocols (both MMAP and STREAM) and IP to exchange data. For more info on the protocol go to [AXI MMAP spec](https://developer.arm.com/docs/ihi0022/e?_ga=2.67820049.1631882347.1556009271-151447318.1544783517), [AXI STREAM spec](https://developer.arm.com/docs/ihi0051/latest), [UG761](https://www.xilinx.com/support/documentation/ip_documentation/axi_ref_guide/latest/ug761_axi_reference_guide.pdf) and [UG1037](https://www.xilinx.com/support/documentation/ip_documentation/axi_ref_guide/latest/ug1037-vivado-axi-reference-guide.pdf).

### Control Plane
![schematic](https://xilinx.github.io/ACCL/images/drawing_ctrl_plane_simplified.svg)

The Figure illustrates  the  detailed  structure  of  the  control plane.  Central  to  the  control  plane  is  a  MicroBlaze  single-core,  FPGA-optimized  microprocessor.  The  role  of  the Microblaze  CPU  in  the  CCLO  design  is  to  provide  the  re-quired flexibility to execute many different collectives utilizing a  multitude  of  FUs  in  combination,  which  is  difficult  to achieve  utilizing  logic  described  in  HLS  or  RTL.  

Firmware executing  on  the  Microblaze  is  able  to  generate  commands to  the  various  FUs  -  the  DMAs,  (de)packetizers,  switch, etc.  These  commands  can  be  assembled  into data  movement primitives,  e.g.  copy,  transmit  to  network,  which  themselves are assembled into collectives. By implementing the control in software (compiled C code), the control logic can run at fairly high  frequencies  (up  to  250  MHz)  with  reasonable  resource utilization  and  can  also  be  adapted  and  improved  overtime  with  relative  ease,  without  requiring  re-synthesis  of  theCCLO kernel. 

Compiling and debugging the firmware is possible utilizing Vitis and the Xilinx command-line tool XSCT, which  enables  all  common  software  development  features  breakpoints, step-by-step execution, profiling and others. 

However, the latency of firmware-directed control is higher when compared to RTL control logic. To counteract the higher latency  of  the  MicroBlaze  we  employed  a  pipelined  control plane  architecture  as  illustrated  in  Figure  3  whereby  most functional  blocks  are  controlled  through  hardware  First-In-First-Out  (FIFO)  memories,  decoupling  control  issuing  from the  execution.  Less  performance-sensitive  FUs  (AXI  Stream switches, arithmetic) are controlled through an AXI bus using register-mapped  interfaces.  These  AXI  interfaces  to  the  FUs are  utilized  at  most  once  per  collective  execution,  to  set long-duration  configurations,  such  as  the  reduction  function,network MTU, etc.

To be more specific, the control plane is divided in CTRL module and the exchange memory
### CTRL module
![schematic_ctrl](https://xilinx.github.io/ACCL/images/drawing_ctrl.svg)

The ``CTRL`` module takes the responsibility of orchestrating the data transfer between components inside the the CCLO kernel. As you can see from the Figure, a  ``MicroBlaze`` is currently employed to implement those functionalities.
The following Figure shows the main connections to and from the ``MicroBlaze`` to other components in the design.

The CTRL module is divided in 7 sub-components:
- ``AXI STREAM FIFOS``: that keep intermediate sts/data/cmd to be read/send to CCL_Offload modules. These are directly connected to ``AXIS (AXI Stream) M0-M10 ``and ``S0-12``. A thing that catches the eye is that the interrupt signal of the Microblaze is driven by a signal that originates from count of two FIFOs.
- The interrupt for the MicroBlaze is raised either when:
  - no data is present on ``DMA0_s2mm(write)_cmd`` queue or
  - when there is data on ``DMA0_s2mm(write)_sts`` queue 
  - no data is present on ``DMA2_s2mm(write)_cmd`` queue or
  - when there is data on ``DMA2_s2mm(write)_sts`` queue.
  As we can see from the CCLO_kernel schematic the depacketizer is directly connected to ``DMA0``. Since we plan to use 100Gbit interface timing is critical to ensure that depacketizer output does not saturate the ``AXIS FIFO``s and eventually result in a packet drop. In that sense if we have no cmd written we should activate the MicroBlaze to identify the location where to put incoming data. In the same way, if the ``DMA0`` completes an write operation we should reactivate the MicroBlaze to issue a new command to execute. 
- reset for all ``FIFO``s is provided by a ``GPIO``, that strictly speaking is inside the ``exchange_memory`` module. But the point is that this pin can be triggered by the MicroBlaze via a AXI LITE register write. 
-  the ``exchange_memory`` submodule give access to the ``MicroBlaze`` from the outside.   There is a memory that can be read/write both from the ``MicroBlaze`` and from the outside of ``CTRL block``. An AXI MMAP port give physical access to ``exchange_memory`` module.  Among the other things, it contains a constant (accessible via an ``AXI GPIO IP`` in [Exchange memory module](#exchange-memory-module)) that identifies the combination of CCLO and network stack in use. 
 ``MicroBlaze S10/M10`` are used to collect/send data from/to ``exchange_memory``. 
- The ``AXI interconnect`` empowers the communication from ``M_AXI DP`` to other AXI MMAP ìnterfaces (``AXI switch ctrl``, ``depacketizer ctrl``, ``packetizer ctrl`` and ``arithmetic ctrl``)
- A memory for instruction and data that the MicroBlaze access via ``ILMB/DLMB`` which are Instruction/Data interface, Local Memory Bus [from MicroBlaze reference guide](https://www.xilinx.com/support/documentation/sw_manuals/mb_ref_guide.pdf).
- The MicroBlaze. info about the ctrl sw in [/sw/sw_apps/ccl_offload_control](/sw/sw_apps/ccl_offload_control)

For more information on how the ``MicroBlaze`` works, take a look at [sw/sw_apps/ccl_offload_control/ccl_offload_control.c](sw/sw_apps/ccl_offload_control/ccl_offload_control.c)

### Exchange memory module
The exchange memory module is the user point of access to the CTRL module.

![schematic_exchange_mem](https://xilinx.github.io/ACCL/images/drawing_exchange_mem.svg)
- hosts_sts comes from ``MicroBlaze M10``;
- [host_ctrl](hls/hostctrl/hostctrl.cpp) receives configuration data from python driver (via ``s_axi_control``, ``AXI crossbar0``) to request an operation to the CCLO kernel as specified  [earlier](#hw-kernel-parameters-(BASEADDR+0));
- Moreover the user can write/read via ``s_axi_control`` -> ``AXI crossbar0``  the ``BRAM`` in this module;
- Furthermore MicroBlaze can write/read the ``BRAM`` in this module, but it has to use ``S00_AXI`` interface passing through ``crossbar1`` and ``crossbar0``. As [previous section](#ctrl-module) showed, the Microblaze can access the ``exchange memory`` module via an AXI MMAP interface.There are mainly 4 reasons for the ``MicroBlaze`` to write/reading something from ``S00_AXI``:
  1. verifying that hw_id corresponds to what is exepected by [ctrl_software](/sw/sw_apps/ccl_offload_control/src/ccl_offload_control.c)
  2. to read some extra parameters in the ``BRAM``
  4. to start/end the ``AXI_FIFO`` reset procedure. i.e. setting a 0 on ``encore_aresetn`` 
  5. to start/end the reset procedure by disabling/enabling the ``AXI register``. This is done by enforcing the ``init_done`` signal to 1. As a consequence the ``AXI register`` stops to reset (note: ``AXI register`` reset is active low).
- The [``AXI GPIO`` is a Xilinx IP](https://www.xilinx.com/support/documentation/ip_documentation/axi_gpio/v2_0/pg144-axi-gpio.pdf) that offers a simple way to read values (e.g. the ``hw_id`` constant) and write values (e.g. the ``encore_resetn`` signal or the ``init_done`` signal). 

# Building and package the ccl_offload.xo Kernel
The [Makefile](Makefile) automates the CCLO kernel build, producing a Vitis-compatible object file (.xo). The Makefile takes the following variables:
  - `PLATFORM` indicates the target Alveo shell.
  - `DEBUG` controls Chipscope insertion. There are three debugging levels:
    - [dma](kernel/tcl/debug_dma.tcl): attaches probes to the internal DMAs and AXI-Stream interconnect
    - [pkt](kernel/tcl/debug_pkt.tcl):  attaches probes to the send and receive datapaths
    - [all](kernel/tcl/debug_all.tcl): combines the above two levels
    - none: does not put any chipscopes

You are ready to go.

1. Source Vitis 2020.2
1. run make to build ``ccl_offload.xo`` under ``ccl_offload_ex/exports``
