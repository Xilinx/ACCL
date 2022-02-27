# /*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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

project(xlnx-dacusr)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

# MPI
find_package(MPI REQUIRED)

# Json
find_package(jsoncpp REQUIRED)
link_libraries(jsoncpp_lib)

# ZMQ
link_libraries(zmq)

# link_directories($ENV{XILINX_XRT}/lib)
# link_libraries(xilinxopencl)
# link_libraries(xrt_coreutil)
# link_libraries(xrt_core)
# include_directories($ENV{XILINX_XRT}/include)
include_directories(include)
add_subdirectory(src)
