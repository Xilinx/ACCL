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
project(host_driven_example)

set(CMAKE_CXX_STANDARD 17)

# Set useful compile warnings
add_compile_options(-Wall -Wextra -Wno-unused-variable -Wno-unknown-pragmas -Wno-deprecated-copy
                    -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-label -Wno-reorder
                    -Wno-missing-field-initializers)

if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endif()


set(ACCL_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../)
set(HLSLIB_INCLUDE ${ACCL_REPO_ROOT}/hlslib/include/hlslib/xilinx/)
set(ACCL_HLSDRV_DIR ${ACCL_REPO_ROOT}/driver/hls/)
set(CCLO_BFM_DIR ${ACCL_REPO_ROOT}/test/model/bfm/)
set(EMU_INCLUDES ${HLSLIB_INCLUDE} $ENV{XILINX_HLS}/include/ ${ACCL_HLSDRV_DIR} ${CCLO_BFM_DIR})


set(CMAKE_BUILD_TYPE Debug)




set(ACCL_NETWORK_UTILS_MPI 1)
set(ACCL_NETWORK_UTILS_DEBUG 1)
set(ACCL_DEBUG 1)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../driver/utils/accl_network_utils ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/accl_network_utils)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../model/bfm ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/cclobfm)

find_package(MPI REQUIRED)
add_executable(scatter-vadd-gather test.cpp)
target_include_directories(scatter-vadd-gather PUBLIC ${MPI_CXX_INCLUDE_PATH} ${EMU_INCLUDES})
# Explicitely link against vnx and network_roce_v2 because CMake doesn't support nested object libraries
# https://stackoverflow.com/a/71040950
target_link_libraries(scatter-vadd-gather PUBLIC MPI::MPI_CXX accl_network_utils vnx cclobfm network_roce_v2)
