# /*******************************************************************************
#  Copyright (C) 2022 Advanced Micro Devices, Inc
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

cmake_minimum_required(VERSION 3.13)

# Consider switching to PROJECT_IS_TOP_LEVEL from CMake 3.21 (2021)
# (https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html)
get_directory_property(HAS_PARENT PARENT_DIRECTORY)

set(XCLBIN_SCAN_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include)
if (HAS_PARENT)
  set(XCLBIN_SCAN_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include PARENT_SCOPE)
endif (HAS_PARENT)

add_library(xclbin_scan STATIC src/xclbin_scan.cpp)

target_include_directories(xclbin_scan PUBLIC ${XCLBIN_SCAN_INCLUDE_PATH})

# XRT
if (NOT EXISTS $ENV{XILINX_XRT})
  message(FATAL_ERROR "Xilinx XRT not found, make sure to source setup.sh")
endif ()

target_link_directories(xclbin_scan PUBLIC $ENV{XILINX_XRT}/lib)
target_link_libraries(xclbin_scan PUBLIC xrt_coreutil)
target_include_directories(xclbin_scan PUBLIC $ENV{XILINX_XRT}/include)

if (XCLBIN_SCAN_DEBUG)
  target_compile_definitions(xclbin_scan PRIVATE XCLBIN_SCAN_DEBUG)
endif ()
