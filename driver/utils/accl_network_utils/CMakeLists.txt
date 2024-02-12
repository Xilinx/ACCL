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

set(CMAKE_CXX_STANDARD 17)

# Consider switching to PROJECT_IS_TOP_LEVEL from CMake 3.21 (2021)
# (https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html)
get_directory_property(HAS_PARENT PARENT_DIRECTORY)

set(ACCL_NETWORK_UTILS_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include)
if (HAS_PARENT)
  set(ACCL_NETWORK_UTILS_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include PARENT_SCOPE)
endif (HAS_PARENT)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../xrt ${CMAKE_CURRENT_BINARY_DIR}/xrt)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../test/refdesigns/xup_vitis_network_example/xrt_host_api ${CMAKE_CURRENT_BINARY_DIR}/vnx)

add_library(accl_network_utils STATIC src/accl_network_utils.cpp)

target_include_directories(accl_network_utils PUBLIC ${ACCL_NETWORK_UTILS_INCLUDE_PATH})

# XRT
if (NOT EXISTS $ENV{XILINX_XRT})
  message(FATAL_ERROR "Xilinx XRT not found, make sure to source setup.sh")
endif ()

target_link_directories(accl_network_utils PUBLIC $ENV{XILINX_XRT}/lib)
target_link_libraries(accl_network_utils PUBLIC xilinxopencl xrt_coreutil xrt_core accl vnx)
target_include_directories(accl_network_utils PUBLIC $ENV{XILINX_XRT}/include)

if (ACCL_NETWORK_UTILS_MPI)
  find_package(MPI REQUIRED)
  target_compile_definitions(accl_network_utils PRIVATE ACCL_NETWORK_UTILS_MPI)
  target_include_directories(accl_network_utils PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(accl_network_utils PRIVATE MPI::MPI_CXX)
endif ()

if (ACCL_NETWORK_UTILS_DEBUG)
  target_compile_definitions(accl_network_utils PRIVATE ACCL_NETWORK_UTILS_DEBUG)
endif ()
