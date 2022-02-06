1. clone the submodule

    `git submodule update --init --recursive Vitis_with_100Gbps_TCP-IP`
2. build `Vitis_with_100Gbps_TCP-IP`:
    
    `cd Vitis_with_100Gbps_TCP-IP/ `

    0. source Vitis 2020.1
         
    1. create a subdir for build 

        `mkdir build`

    2. `cd build`

    3. run cmake changing the target platform is needed:

        `cmake .. -DFDEV_NAME=u280 -DVIVADO_HLS_ROOT_DIR=/proj/xbuilds/2020.1_released/installs/lin64/Vivado/2020.1 -DVIVADO_ROOT_DIR=/proj/xbuilds/2020.1_released/installs/lin64/Vivado/2020.1 -DTCP_STACK_EN=1 -DTCP_STACK_RX_DDR_BYPASS_EN=1  -DTCP_STACK_WINDOW_SCALING=0`

    4. run `make installip`

3. Run make in the demo directory.

    `make `