## Create a demo .xclbin that uses TCP-IP stack



1. clone the submodule

    `git submodule update --init --recursive Vitis_with_100Gbps_TCP-IP`
2. build `Vitis_with_100Gbps_TCP-IP`:
    
    `cd Vitis_with_100Gbps_TCP-IP/ `

    0. source Vitis 2020.1
         
    1. create a subdir for build 

        `mkdir build`

    2. `cd build`

    3. run cmake changing ``DFDEV_NAME`` if needed (supported values u280, u250):

        `cmake .. -DFDEV_NAME=u280 -DVIVADO_HLS_ROOT_DIR=/proj/xbuilds/2020.1_released/installs/lin64/Vivado/2020.1 -DVIVADO_ROOT_DIR=/proj/xbuilds/2020.1_released/installs/lin64/Vivado/2020.1 -DTCP_STACK_EN=1 -DTCP_STACK_RX_DDR_BYPASS_EN=1  -DTCP_STACK_WINDOW_SCALING=0`

    4. run `make installip`

1. go back to the demo directory.

    `cd ../..`

1. modify the Makefile.
    1. the change ``PLATFORM`` variable according to your need (default: xilinx_u280_xdma_201920_3, other: xilinx_u250_gen3x16_xdma_3_1_202020_1)
    
    1. set ``MODE`` variable. There are many different demo design that you can build the example mode is `tcp_cmac`


    1. run make
    
    `make `

4. Congrats you built a xclbin. Look in link_.* directory you will find a ``ccl_offload.xclbin``

To run the example go in [../host](../host)