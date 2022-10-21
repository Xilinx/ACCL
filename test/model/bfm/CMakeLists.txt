# /*******************************************************************************
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
# *******************************************************************************/

cmake_minimum_required(VERSION 3.9)
project(cclobfm VERSION 0.1.0 DESCRIPTION "CCLO Bus Functional Model")

set(CMAKE_CXX_STANDARD 17)

# Set useful compile warnings
add_compile_options(-Wall -Wno-attributes -Wno-unknown-pragmas)


if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endif()

# Consider switching to PROJECT_IS_TOP_LEVEL from CMake 3.21 (2021)
# (https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html)
get_directory_property(HAS_PARENT PARENT_DIRECTORY)

set(CCLO_BFM_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR})
set(CCLO_BFM_HEADER_PATH ${CMAKE_CURRENT_LIST_DIR})
set(ACCL_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../)
set(HLSLIB_INCLUDE ${ACCL_REPO_ROOT}/hlslib/include/hlslib/xilinx/)
set(ACCL_XRTDRV_INCLUDE_PATH ${ACCL_REPO_ROOT}/driver/xrt/include)
set(ACCL_HLSDRV_INCLUDE_PATH ${ACCL_REPO_ROOT}/driver/hls)
set(ZMQ_INTF_DIR ${ACCL_REPO_ROOT}/test/model/zmq)

set(CCLO_BFM_HEADERS ${CCLO_BFM_HEADER_PATH}/cclo_bfm.h)
set(CCLO_BFM_SOURCES ${CCLO_BFM_SOURCE_PATH}/cclo_bfm.cpp)

set(CCLO_BFM_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR} ${ZMQ_INTF_DIR})
if (HAS_PARENT)
  set(CCLO_BFM_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR} ${ZMQ_INTF_DIR} PARENT_SCOPE)
endif (HAS_PARENT)

# HLS(Lib)
if (NOT EXISTS ${HLSLIB_INCLUDE})
  message(FATAL_ERROR "hlslib submodule not loaded! Run git submodule update --init --recursive in ${ACCL_REPO_ROOT}")
endif (NOT EXISTS ${HLSLIB_INCLUDE})

if (NOT EXISTS $ENV{XILINX_HLS})
  message(FATAL_ERROR "Xilinx HLS not loaded!")
endif (NOT EXISTS $ENV{XILINX_HLS})

# XRT
if (NOT EXISTS $ENV{XILINX_XRT})
  message(FATAL_ERROR "Xilinx XRT not found, make sure to source setup.sh")
endif ()

# Json
find_package(jsoncpp REQUIRED)
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)

add_library(cclobfm SHARED ${CCLO_BFM_SOURCES})
target_include_directories(cclobfm PUBLIC ${ACCL_XRTDRV_INCLUDE_PATH} ${ACCL_HLSDRV_INCLUDE_PATH} ${CCLO_BFM_INCLUDE_PATH} ${HLSLIB_INCLUDE} $ENV{XILINX_HLS}/include/ ${JSON_INC_PATH} $ENV{XILINX_XRT}/include)
target_link_libraries(cclobfm PUBLIC jsoncpp_lib zmqpp zmq pthread)

set_target_properties(cclobfm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/cclo_bfm.h
)


