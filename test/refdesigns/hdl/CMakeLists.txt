cmake_minimum_required(VERSION 3.0)
project(test)

set(CYT_DIR ${CMAKE_SOURCE_DIR}/../Coyote)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)

find_package(CoyoteHW REQUIRED)

#
# ACCL Test
#
if(EXAMPLE STREQUAL "rdma")
    message("** ACCL-RDMA Test")
    set(SHELL_PROBE 7)
    set(N_REGIONS 1)
    set(EN_STRM 1)
    set(EN_RDMA 1)
    set(N_STRM_AXI 3)
    set(N_CARD_AXI 3)
    set(N_RDMA_AXI 2)
    set(ACLK_F 250)
    set(TLBL_A 12)
    set(EN_MEM 1)

    validation_checks_hw()

    # load_apps (
    #     VFPGA_C0_0 "coyote_rdma_top.sv"
    # )

    create_hw()
endif()
