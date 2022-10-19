..
   comment:: SPDX-License-Identifier: Apache-2.0
   comment:: Copyright (C) 2022 Advanced Micro Devices, Inc

###################################################
ACCL: The Alveo Collective Communication Library
###################################################

ACCL enables MPI-like communication between network-attached FPGAs, facilitating the development of FPGA-accelerated distributed applications.
There are several advantages to utilizing ACCL instead of MPI for such applications:

* FPGAs communicate directly rather than through system NICs, reducing latency and putting less pressure on the system bus.
* No CPU involvement in communication, as collective orchestration is performed in the FPGA.
* FPGA kernels can request communication directly, reducing CPU involvement in application orchestration, as well as eliminating kernel invocation latencies.

The figure below illustrates a typical ACCL-enabled application. The FPGA contains ACCL-specific components (the :ref:`overview_cclo_subsystem`)
and optionally user-defined FPGA kernels. Either these kernels or the user application running on the CPU host, via the :ref:`overview_accl_driver`, may initiate
communication primitives, which are executed by moving data across the FPGA 100-gigabit transceivers.

.. rst-class:: centered
.. tikz:: FPGA-accelerated and ACCL-enabled distributed application
   :include: diagrams/system.tex
   :libs: arrows,shapes
   :xscale: 50

ACCL currently supports the following communication primitives:

* Send, Receive, and Put
* Broadcast
* Scatter
* Gather and All-Gather
* Reduce, All-Reduce, and Reduce-Scatter

While ACCL does provide the above communication primitives to an application,
it is important to note that ACCL is not a NIC, it cannot be utilized to perform general-purpose communication between the systems it populates. 
The system must be equipped with a NIC to enable CPUs in the distributed system to communicate.
We refer to the FPGA network as the back-end network while the NICs connect to the front-end network.
In practice, these can be the same network. 


.. toctree::
   :maxdepth: 1
   :caption: Contents
   
   overview
   install
   cpp_bindings/index
   hls/index
   troubleshooting