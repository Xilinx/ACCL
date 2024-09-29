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

# Consider switching to PROJECT_IS_TOP_LEVEL from CMake 3.21 (2021)
# (https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html)
get_directory_property(HAS_PARENT PARENT_DIRECTORY)

set(ACCL_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/src)
set(ACCL_HEADER_PATH ${CMAKE_CURRENT_LIST_DIR}/include)
set(ACCL_DOCS_PATH ${CMAKE_CURRENT_LIST_DIR}/docs)
set(ACCL_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../)
set(ZMQ_INTF_DIR ${ACCL_REPO_ROOT}/test/model/zmq)
set(COYOTE_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../test/refdesigns/Coyote/sw/src)
set(COYOTE_HEADER_PATH ${CMAKE_CURRENT_LIST_DIR}/../../test/refdesigns/Coyote/sw/include)

set(ACCL_HEADERS
    ${ACCL_HEADER_PATH}/accl.hpp
    ${ACCL_HEADER_PATH}/common.hpp
    ${ACCL_HEADER_PATH}/communicator.hpp
    ${ACCL_HEADER_PATH}/constants.hpp
    ${ACCL_HEADER_PATH}/simdevice.hpp
    ${ACCL_HEADER_PATH}/simbuffer.hpp
    ${ACCL_HEADER_PATH}/xrtdevice.hpp
    ${ACCL_HEADER_PATH}/acclrequest.hpp
)

set(ACCL_SOURCES
    ${ACCL_SOURCE_PATH}/accl.cpp
    ${ACCL_SOURCE_PATH}/common.cpp
    ${ACCL_SOURCE_PATH}/communicator.cpp
    ${ACCL_SOURCE_PATH}/constants.cpp
    ${ACCL_SOURCE_PATH}/simdevice.cpp
    ${ACCL_SOURCE_PATH}/simbuffer.cpp
    ${ACCL_SOURCE_PATH}/xrtdevice.cpp
    ${ZMQ_INTF_DIR}/zmq_client.cpp
    ${ZMQ_INTF_DIR}/zmq_common.cpp
)

set(ACCL_DOCS_RST
    ${ACCL_DOCS_PATH}/index.rst
    ${ACCL_DOCS_PATH}/Cpp_reference/index.rst
    ${ACCL_DOCS_PATH}/Cpp_reference/accl.rst
    ${ACCL_DOCS_PATH}/Cpp_reference/buffer.rst
    ${ACCL_DOCS_PATH}/Cpp_reference/cclo.rst
    ${ACCL_DOCS_PATH}/Cpp_reference/communicator.rst
    ${ACCL_DOCS_PATH}/Cpp_reference/misc.rst
)
        
set(EN_COYOTE ON)
set(EN_AVX 1 CACHE STRING "AVX environment.")
if(EN_COYOTE)
    message("Enable Coyote")
    set(ACCL_HEADERS 
        ${ACCL_HEADERS} 
        ${ACCL_HEADER_PATH}/coyotebuffer.hpp
        ${ACCL_HEADER_PATH}/coyotedevice.hpp
    )
    set(ACCL_SOURCES 
        ${ACCL_SOURCES} 
        ${ACCL_SOURCE_PATH}/coyotedevice.cpp
    )
    file(GLOB COYOTE_SOURCE "${COYOTE_SOURCE_PATH}/*.cpp")
    
    if(EN_AVX)
        add_definitions(-DEN_AVX)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -mavx -march=native -O3")
    else()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -march=native -O1")
    endif()
endif(EN_COYOTE)


set(ACCL_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include ${ZMQ_INTF_DIR})
if (HAS_PARENT)
  set(ACCL_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include ${ZMQ_INTF_DIR} PARENT_SCOPE)
endif (HAS_PARENT)


if(EN_COYOTE)
    set(ACCL_SOURCES ${ACCL_SOURCES} ${COYOTE_SOURCE})
    set(ACCL_INCLUDE_PATH ${ACCL_INCLUDE_PATH} ${COYOTE_HEADER_PATH})
endif(EN_COYOTE)


add_library(accl SHARED ${ACCL_SOURCES})
target_include_directories(accl PUBLIC ${ACCL_INCLUDE_PATH})

# XRT
if (NOT EXISTS $ENV{XILINX_XRT})
  message(FATAL_ERROR "Xilinx XRT not found, make sure to source setup.sh")
endif ()

target_link_directories(accl PUBLIC $ENV{XILINX_XRT}/lib)
target_link_libraries(accl PUBLIC xilinxopencl xrt_coreutil xrt_core)
target_include_directories(accl PUBLIC $ENV{XILINX_XRT}/include)

# ZMQ
target_link_libraries(accl PUBLIC zmq pthread)

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

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(Sphinx)

if (HAS_PARENT)
  set(DOC_TARGET accl_docs)
else (HAS_PARENT)
  set(DOC_TARGET docs)
endif (HAS_PARENT)

if (DOXYGEN_FOUND AND SPHINX_FOUND)
    set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_LIST_DIR}/docs/xml/index.xml)
    set(SPHINX_INDEX_FILE ${CMAKE_CURRENT_LIST_DIR}/docs/sphinx/index.html)
    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        BYPRODUCTS
        ${CMAKE_CURRENT_LIST_DIR}/docs/xml
        ${CMAKE_CURRENT_LIST_DIR}/docs/latex
        ${CMAKE_CURRENT_LIST_DIR}/docs/html
        DEPENDS ${ACCL_HEADERS}
        MAIN_DEPENDENCY ${CMAKE_CURRENT_LIST_DIR}/Doxyfile
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )

    add_custom_target(${DOC_TARGET}_doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

    add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
        COMMAND ${SPHINX_EXECUTABLE} -b html ${CMAKE_CURRENT_LIST_DIR}/docs ${CMAKE_CURRENT_LIST_DIR}/docs/sphinx
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        BYPRODUCTS
        ${CMAKE_CURRENT_LIST_DIR}/docs/sphinx
        DEPENDS ${ACCL_DOCS_RST}
        MAIN_DEPENDENCY ${DOXYGEN_INDEX_FILE}
        COMMENT "Generating API documentation with Sphinx"
        VERBATIM
        )

    add_custom_target(${DOC_TARGET} ALL DEPENDS ${SPHINX_INDEX_FILE})

    set_target_properties(${DOC_TARGET}_doxygen ${DOC_TARGET} PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
else()
    message("Doxygen needs to be installed to generate the doxygen documentation")
endif()
