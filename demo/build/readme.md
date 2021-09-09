# How to build a CCL_Offload ``.xclbin``
The [Makefile](Makefile) automates the building process of the CCL_Offload IP and its connection.
  The objective of the Makefile is to build a CCL_Offload [``.xclbin``](https://xilinx.github.io/XRT/2018.3/html/formats.html) under a directory that reflects the name of the configuration that has been chosen. This directory starts with ``link_`` and the name is set in the [Makefile](Makefile) preamble depending on the selected mode. 
  
  Let's take a look at the [Makefile](Makefile). The preamble sets some variables:
  - `PLATFORM` indicates the shekk you are targeting. You can get it from:` xbutil list`  or ` xbutil query -d0 `.

  - ``MODE``. There are two modes in which the CCLO can be connected in a demo system: tcp_cmac, vnx  and tri 
      - tcp_cmac: uses 1 CCL_Offload connected to ethernet interface thorugh a network stack provided by [Vitis_with_100Gbps_TCP-IP](https://github.com/fpgasystems/Vitis_with_100Gbps_TCP-IP). This works on top of [UltraScale+  Integrated  100GEthernet Subsystem (CMAC)](https://www.xilinx.com/products/intellectual-property/cmac_usplus.html) to which it provides frames. Vitis_with_100Gbps_TCP-IP currently supports only Vitis 2020.1 while CCLO supports Vitis 2020.2. As a consequence the build sequence is:
        1. configure Vivado 2020.1 
        
        2. ``make tcp_stack_ips``

        3. configure Vivado 2020.2

        4. ``make``

      - vnx: uses 1 CCL_Offload which is connected to an ethernet interface through VNX, a Vitis-compatible 100Gbps UDP stack which works on top of [UltraScale+  Integrated  100GEthernet Subsystem (CMAC)](https://www.xilinx.com/products/intellectual-property/cmac_usplus.html) to which it provides frames. VNX currently supports only Vitis 2020.1 while CCLO supports Vitis 2020.2. As a consequence the build sequence is:
          
        1. configure Vivado 2020.1  
        
        2. ``make vnx``

        3. configure Vivado 2020.2

        4. ``make``

      - tri: uses 3 CCLO kernels connected through an AXI-Stream switch to emulate a 3-node system with a single Alveo 


    Notes:
      - In terms of Make, a phony target(e.g. ``.PHONY: elf``) is simply a target that is always out-of-date, so whenever you ask make <phony_target> (e.g. Make elf) , it will always run, independent from the state of the file. [more info here](https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html);
      - indicating a file as a dependency in a recipe means means that the results of the recipe has to be rebuilt in case the file has been changed from the last modification of the target;
      - make -C invokes make in a sub-directory.


To run the example go in [../host](../host)