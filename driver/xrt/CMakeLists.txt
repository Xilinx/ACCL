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

cmake_minimum_required(VERSION 3.9)
project(accl VERSION 0.1.0 DESCRIPTION "ACCL")

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endif()

# Json
find_package(jsoncpp REQUIRED)

set(ACCL_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/src)

set(ACCL_SOURCES
    ${ACCL_SOURCE_PATH}/accl.cpp
    ${ACCL_SOURCE_PATH}/common.cpp
    ${ACCL_SOURCE_PATH}/communicator.cpp
    ${ACCL_SOURCE_PATH}/simdevice.cpp
)

set(ACCL_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(accl SHARED ${ACCL_SOURCES})

target_link_libraries(accl jsoncpp_lib zmq)
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(accl PUBLIC ${JSON_INC_PATH} ${ACCL_INCLUDE_PATH})
if (EXISTS $ENV{XILINX_XRT})
  target_link_directories(accl PUBLIC $ENV{XILINX_XRT}/lib)
  target_link_libraries(accl PUBLIC xilinxopencl xrt_coreutil xrt_core)
  target_include_directories(accl PUBLIC $ENV{XILINX_XRT}/include)
  target_compile_definitions(accl PUBLIC ACCL_HARDWARE_SUPPORT)
endif (EXISTS $ENV{XILINX_XRT})

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
if (DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
    message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
