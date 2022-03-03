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



set(ACCL_SOURCES
    src/accl.cpp
    src/common.cpp
    src/communicator.cpp
    src/simdevice.cpp
)

add_library(accl SHARED
    ${ACCL_SOURCES}
)

target_link_libraries(accl jsoncpp_lib zmq)
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(accl PUBLIC ${JSON_INC_PATH} include)
if (EXISTS $ENV{XILINX_XRT})
  target_link_directories(accl PUBLIC $ENV{XILINX_XRT}/lib)
  target_link_libraries(accl PUBLIC xilinxopencl xrt_coreutil xrt_core)
  target_include_directories(accl PUBLIC $ENV{XILINX_XRT}/include)
  target_compile_definitions(accl PUBLIC ACCL_HARDWARE_SUPPORT)
endif (EXISTS $ENV{XILINX_XRT})

set_target_properties(accl PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/accl.hpp
)

# docs
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
    message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

# check if called directly
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # MPI
  find_package(MPI)
  if (MPI_FOUND)
      add_executable(test src/test.cpp ${ACCL_SOURCES})
      target_include_directories(test PUBLIC ${MPI_CXX_INCLUDE_PATH} include)
      target_link_libraries(test PUBLIC MPI::MPI_CXX accl)
  else (MPI_FOUND)
      message("MPI needs to be installed to generate test executable")
  endif (MPI_FOUND)
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
