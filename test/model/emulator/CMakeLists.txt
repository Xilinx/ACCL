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
project(accl_emulation)

set(CMAKE_CXX_STANDARD 17)

# Consider using cmake_path from CMake 3.20 (2021) for better error messages:
# https://cmake.org/cmake/help/latest/command/cmake_path.html
set(ACCL_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../)
set(HLSLIB_INCLUDE ${ACCL_REPO_ROOT}/hlslib/include/hlslib/xilinx/)
set(ACCL_HLSDRV_DIR ${ACCL_REPO_ROOT}/driver/hls/)
set(CCLO_HLS_ROOT ${ACCL_REPO_ROOT}/kernels/cclo/hls)
set(REDUCTION_DIR ${ACCL_REPO_ROOT}/kernels/plugins/reduce_ops)
set(COMPRESSOR_DIR ${ACCL_REPO_ROOT}/kernels/plugins/hp_compression)
set(DUMMY_TCP_DIR ${ACCL_REPO_ROOT}/kernels/plugins/dummy_tcp_stack)
set(DUMMY_CYT_RDMA_DIR ${ACCL_REPO_ROOT}/kernels/plugins/dummy_cyt_rdma_stack)
set(HOSTCTRL_DIR ${ACCL_REPO_ROOT}/kernels/plugins/hostctrl)
set(ARBITER_DIR ${ACCL_REPO_ROOT}/kernels/plugins/client_arbiter)
set(CCLO_ETH_DIR ${CCLO_HLS_ROOT}/eth_intf)
set(SEGMENTER_DIR ${CCLO_HLS_ROOT}/segmenter)
set(RXBUF_OFFLOAD_DIR ${CCLO_HLS_ROOT}/rxbuf_offload)
set(DMA_MOVER_DIR ${CCLO_HLS_ROOT}/dma_mover)
set(MB_FW_DIR ${ACCL_REPO_ROOT}/kernels/cclo/fw/sw_apps/ccl_offload_control/src)
set(ZMQ_INTF_DIR ${ACCL_REPO_ROOT}/test/model/zmq)
set(LOG_DIR ${ACCL_REPO_ROOT}/test/log)
set(HLS_HEADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/hls_sim_headers/include/)

find_package(jsoncpp REQUIRED)

if (NOT EXISTS ${HLSLIB_INCLUDE})
  message(FATAL_ERROR "hlslib submodule not loaded! Run git submodule update --init --recursive in ${ACCL_REPO_ROOT}")
endif (NOT EXISTS ${HLSLIB_INCLUDE})

if ((NOT EXISTS $ENV{XILINX_HLS}) AND (NOT EXISTS ${HLS_HEADERS_DIR}))
  message(FATAL_ERROR "Xilinx HLS not loaded and no alternative HLS headers present!")
endif ()

if (NOT EXISTS $ENV{XILINX_HLS})
  message(WARNING "Xilinx HLS not loaded, falling back to open-source HLS headers")
  set(HLS_INCLUDES ${HLS_HEADERS_DIR} ${HLSLIB_INCLUDE})
  if (NOT EXISTS ${HLS_HEADERS_DIR})
    message(FATAL_ERROR "Open-source HLS headers submodule not loaded! Run git submodule update --init --recursive in ${ACCL_REPO_ROOT}")
  endif (NOT EXISTS ${HLS_HEADERS_DIR})
else ()
  message(NOTICE "Using Xilinx HLS headers")
  set(HLS_INCLUDES $ENV{XILINX_HLS}/include/ ${HLSLIB_INCLUDE})
endif (NOT EXISTS $ENV{XILINX_HLS})

set_source_files_properties(${MB_FW_DIR}/ccl_offload_control.c PROPERTIES LANGUAGE CXX)

set(EMU_INCLUDES ${HLS_INCLUDES} ${REDUCTION_DIR} ${COMPRESSOR_DIR} ${CCLO_ETH_DIR} ${SEGMENTER_DIR} ${MB_FW_DIR} ${CCLO_HLS_ROOT} ${DMA_MOVER_DIR} ${RXBUF_OFFLOAD_DIR} ${DUMMY_TCP_DIR} ${DUMMY_CYT_RDMA_DIR} ${HOSTCTRL_DIR} ${ARBITER_DIR} ${ACCL_HLSDRV_DIR} ${ZMQ_INTF_DIR} ${LOG_DIR})
set(EMU_SOURCES cclo_emu.cpp ${MB_FW_DIR}/ccl_offload_control.c ${REDUCTION_DIR}/reduce_ops.cpp ${COMPRESSOR_DIR}/hp_compression.cpp
                ${CCLO_ETH_DIR}/udp_depacketizer.cpp ${CCLO_ETH_DIR}/tcp_txHandler.cpp ${CCLO_ETH_DIR}/tcp_rxHandler.cpp ${CCLO_ETH_DIR}/udp_packetizer.cpp ${CCLO_ETH_DIR}/tcp_packetizer.cpp ${CCLO_ETH_DIR}/tcp_depacketizer.cpp
                ${CCLO_ETH_DIR}/rdma_depacketizer.cpp ${CCLO_ETH_DIR}/rdma_packetizer.cpp ${CCLO_ETH_DIR}/rdma_sq_handler.cpp
                ${SEGMENTER_DIR}/stream_segmenter.cpp ${RXBUF_OFFLOAD_DIR}/rxbuf_dequeue.cpp ${RXBUF_OFFLOAD_DIR}/rxbuf_enqueue.cpp ${RXBUF_OFFLOAD_DIR}/rxbuf_seek.cpp ${RXBUF_OFFLOAD_DIR}/rxbuf_session.cpp
                ${DMA_MOVER_DIR}/dma_mover.cpp ${DUMMY_TCP_DIR}/dummy_tcp_stack.cpp ${DUMMY_CYT_RDMA_DIR}/dummy_cyt_rdma_stack.cpp ${HOSTCTRL_DIR}/hostctrl.cpp ${ARBITER_DIR}/client_arbiter.cpp ${ZMQ_INTF_DIR}/zmq_server.cpp)

add_executable(cclo_emu ${EMU_SOURCES})

get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(cclo_emu PUBLIC ${JSON_INC_PATH} ${EMU_INCLUDES})
target_link_libraries(cclo_emu PUBLIC zmq zmqpp pthread jsoncpp_lib)
target_compile_definitions(cclo_emu PUBLIC MB_FW_EMULATION NUM_CTRL_STREAMS=3)
if (NOT EXISTS $ENV{XILINX_HLS})
  target_compile_definitions(cclo_emu PUBLIC OSS_HALF_PRECISION)
else ()
  target_compile_definitions(cclo_emu PUBLIC HLS_NO_XIL_FPO_LIB)
endif (NOT EXISTS $ENV{XILINX_HLS})
target_compile_options(cclo_emu PRIVATE -Wno-attributes -fdiagnostics-color=always -g)

install(TARGETS cclo_emu DESTINATION bin)
