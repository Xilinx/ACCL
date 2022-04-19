# /*****************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
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
# *****************************************************************************/

cmake_minimum_required(VERSION 3.9)
project(accl_simulation)

set(CMAKE_CXX_STANDARD 17)

set(ACCL_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../)
set(HLSLIB_INCLUDE ${ACCL_REPO_ROOT}/hlslib/include/hlslib/xilinx/)
set(CCLO_HLS_ROOT ${ACCL_REPO_ROOT}/kernels/cclo/hls)
set(MB_FW_DIR ${ACCL_REPO_ROOT}/kernels/cclo/fw/sw_apps/ccl_offload_control/src)
set(ZMQ_INTF_DIR ${ACCL_REPO_ROOT}/test/model/zmq)
set(LOG_DIR ${ACCL_REPO_ROOT}/test/log)

if (NOT EXISTS $ENV{XILINX_HLS})
  message(FATAL_ERROR "Xilinx HLS not loaded!")
endif (NOT EXISTS $ENV{XILINX_HLS})

find_package(MPI REQUIRED)
find_package(jsoncpp REQUIRED)

set(SIM_INCLUDES ${HLSLIB_INCLUDE} $ENV{XILINX_HLS}/include/ $ENV{XILINX_VIVADO}/data/xsim/include/ ${MB_FW_DIR} ${CCLO_HLS_ROOT} ${ZMQ_INTF_DIR} ${LOG_DIR})
set(SIM_SOURCES cclo_sim.cpp xsi_dut.cpp xsi_loader.cpp ${ZMQ_INTF_DIR}/zmq_server.cpp)

add_executable(cclo_sim ${SIM_SOURCES})
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(cclo_sim PUBLIC ${JSON_INC_PATH} ${MPI_CXX_INCLUDE_PATH} ${SIM_INCLUDES})
target_link_libraries(cclo_sim PUBLIC MPI::MPI_CXX zmq zmqpp pthread jsoncpp_lib dl rt)
target_compile_definitions(cclo_sim PUBLIC ZMQ_CALL_VERBOSE)
target_compile_options(cclo_sim PRIVATE -Wno-attributes -fdiagnostics-color=always -g -fmax-errors=3)

set(XSIM_COMPILE_FOLDER ${ACCL_REPO_ROOT}/kernels/cclo/ccl_offload_ex/ccl_offload_ex.sim/sim_1/behav/xsim/)

if (NOT EXISTS ${XSIM_COMPILE_FOLDER})
    message(FATAL_ERROR "Simulation kernel not build. Run make simdll in ${ACCL_REPO_ROOT}/kernels/cclo")
endif ()

set(SYMLINKS xsim.dir
             ccl_offload_bd_axi_bram_ctrl_0_bram_0.mem ccl_offload_bd_dlmb_bram_if_cntlr_bram_0.mem
             dma_mover_instruction_decode_arcfg_6_RAM_AUTO_1R1W.dat rxbuf_session_rxbuf_session_command_mem_active_RAM_AUTO_1R1W.dat
             rxbuf_session_rxbuf_session_command_mem_index_RAM_AUTO_1R1W.dat tcp_depacketizer_target_strm_V_RAM_AUTO_1R1W.dat
             tcp_depacketizer_remaining_RAM_AUTO_1R1W.dat rxbuf_session_rxbuf_session_command_mem_header_dst_V_RAM_AUTO_1R1W.dat
             rxbuf_session_rxbuf_session_command_mem_address_V_RAM_AUTO_1R1W.dat)

foreach(SYMLINK ${SYMLINKS})
  add_custom_target(${SYMLINK} ALL
    COMMAND ln -sf ${XSIM_COMPILE_FOLDER}/${SYMLINK} ${SYMLINK}
    BYPRODUCTS ${SYMLINK}
  )
endforeach()