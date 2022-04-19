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
project(accl VERSION 0.1.0 DESCRIPTION "ACCL")

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endif()

# Consider switching to PROJECT_IS_TOP_LEVEL from CMake 3.21
# (https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html)
get_directory_property(HAS_PARENT PARENT_DIRECTORY)

set(ACCL_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/src)
set(ACCL_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../)
set(ZMQ_INTF_DIR ${ACCL_REPO_ROOT}/test/model/zmq)

set(ACCL_SOURCES
    ${ACCL_SOURCE_PATH}/accl.cpp
    ${ACCL_SOURCE_PATH}/common.cpp
    ${ACCL_SOURCE_PATH}/communicator.cpp
    ${ACCL_SOURCE_PATH}/simdevice.cpp
    ${ACCL_SOURCE_PATH}/simbuffer.cpp
    ${ACCL_SOURCE_PATH}/fpgadevice.cpp
    ${ZMQ_INTF_DIR}/zmq_client.cpp
    ${ZMQ_INTF_DIR}/zmq_common.cpp
)

set(ACCL_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include ${ZMQ_INTF_DIR})
if (HAS_PARENT)
  set(ACCL_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include ${ZMQ_INTF_DIR} PARENT_SCOPE)
endif (HAS_PARENT)

add_library(accl SHARED ${ACCL_SOURCES})

# XRT
if (EXISTS $ENV{XILINX_XRT})
  set(ACCL_HARDWARE_SUPPORT TRUE)
  if (HAS_PARENT)
    set(ACCL_HARDWARE_SUPPORT TRUE PARENT_SCOPE)
  endif (HAS_PARENT)
  target_link_directories(accl PUBLIC $ENV{XILINX_XRT}/lib)
  target_link_libraries(accl PUBLIC xilinxopencl xrt_coreutil xrt_core)
  target_include_directories(accl PUBLIC $ENV{XILINX_XRT}/include)
  target_compile_definitions(accl PUBLIC ACCL_HARDWARE_SUPPORT)
else (EXISTS $ENV{XILINX_XRT})
  set(ACCL_HARDWARE_SUPPORT FALSE)
  if (HAS_PARENT)
    set(ACCL_HARDWARE_SUPPORT FALSE PARENT_SCOPE)
  endif (HAS_PARENT)
  message(WARNING "Xilinx XRT not found, compiling without hardware support")
endif (EXISTS $ENV{XILINX_XRT})

target_link_libraries(accl PUBLIC zmqpp)
target_include_directories(accl PUBLIC ${ACCL_INCLUDE_PATH})

# Json

find_package(jsoncpp REQUIRED)
target_link_libraries(accl PUBLIC jsoncpp_lib)
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(accl PUBLIC ${JSON_INC_PATH})


if (ACCL_DEBUG)
  target_compile_definitions(accl PUBLIC ACCL_DEBUG)
  message("Defining ACCL_DEBUG")
endif (ACCL_DEBUG)

set_target_properties(accl PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/accl.hpp
)

# docs
find_package(Doxygen)

if (HAS_PARENT)
  set(DOC_TARGET accl_docs)
else (HAS_PARENT)
  set(DOC_TARGET docs)
endif (HAS_PARENT)

if (DOXYGEN_FOUND)
    add_custom_target(${DOC_TARGET}
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
    message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
