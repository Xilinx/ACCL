## Create a demo .xclbin that uses TCP-IP stack

1. modify the Makefile.
    1. the change ``PLATFORM`` variable according to your need (default: xilinx_u280_xdma_201920_3, other: xilinx_u250_gen3x16_xdma_3_1_202020_1)
    
    1. set ``MODE`` variable. There are many different demo design that you can build the example mode is `tcp_cmac`

    1. set ``DEBUG`` variable to ``none``

1. build `Vitis_with_100Gbps_TCP-IP`:
    
    0. source Vitis 2020.1
         
    1. run make tcp_stack_ips

    `make tcp_stack_ips`

1. Source Vitis 2020.2

    1. run make
    
    `make`

4. Congrats you built a xclbin. Look in link_.* directory you will find a ``ccl_offload.xclbin``

To run the example go in [../host](../host)