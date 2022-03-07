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
project(xrt_test)

set(CMAKE_BUILD_TYPE Debug)

set(ACCL_DEBUG 1)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../driver/xrt/CMakeLists.txt)

find_package(MPI REQUIRED)
add_executable(test test.cpp)
target_include_directories(test PUBLIC ${MPI_CXX_INCLUDE_PATH} ${ACCL_INCLUDE_PATH} cxxopts/include)
target_link_libraries(test PUBLIC MPI::MPI_CXX accl)
