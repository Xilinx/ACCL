# /*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
#  Modifications Copyright (c) 2024, Advanced Micro Devices, Inc.
#  All rights reserved.
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

cmake_minimum_required(VERSION 3.16)
if(COYOTE_HSA_SUPPORT)

    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()

    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
        elseif(DEFINED ENV{HIP_PATH})
            set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
        else()
            set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
        endif()
    endif()

    if(NOT DEFINED HCC_PATH)
        if(DEFINED ENV{HCC_PATH})
            set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
        else()
            set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
        endif()
        set(HCC_HOME "${HCC_PATH}")
    endif()

    if(NOT DEFINED HIP_CLANG_PATH)
        if(NOT DEFINED ENV{HIP_CLANG_PATH})
            set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
        else()
            set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
        endif()
    endif()

    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    list(APPEND CMAKE_PREFIX_PATH
        "${HIP_PATH}/lib/cmake"
        "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies
    )


    project(accl VERSION 0.1.0 DESCRIPTION "ACCL" LANGUAGES CXX HIP)
else()
    project(accl VERSION 0.1.0 DESCRIPTION "ACCL" LANGUAGES CXX)
endif()

if(COYOTE_HSA_SUPPORT)
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif()
    find_package(hip REQUIRED)
endif()

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

if(XRT_REQUIRED)
    add_compile_definitions(XRT_REQUIRED)
endif()

if (NOT XRT_REQUIRED)
set(ACCL_HEADERS
    ${ACCL_HEADER_PATH}/accl.hpp
    ${ACCL_HEADER_PATH}/common.hpp
    ${ACCL_HEADER_PATH}/communicator.hpp
    ${ACCL_HEADER_PATH}/constants.hpp
    ${ACCL_HEADER_PATH}/acclrequest.hpp
)
else()
set(ACCL_HEADERS
    ${ACCL_HEADER_PATH}/accl.hpp
    ${ACCL_HEADER_PATH}/common.hpp
    ${ACCL_HEADER_PATH}/communicator.hpp
    ${ACCL_HEADER_PATH}/constants.hpp
    ${ACCL_HEADER_PATH}/simdevice.hpp
    ${ACCL_HEADER_PATH}/simbuffer.hpp
    ${ACCL_HEADER_PATH}/fpgadevice.hpp
    ${ACCL_HEADER_PATH}/acclrequest.hpp
)
endif()

if(COYOTE_HSA_SUPPORT)
set(ACCL_SOURCES
    ${ACCL_SOURCE_PATH}/accl.cpp
    ${ACCL_SOURCE_PATH}/common.cpp
    ${ACCL_SOURCE_PATH}/communicator.cpp
    ${ACCL_SOURCE_PATH}/constants.cpp
    ${ZMQ_INTF_DIR}/zmq_client.cpp
    ${ZMQ_INTF_DIR}/zmq_common.cpp
)
else()
set(ACCL_SOURCES
    ${ACCL_SOURCE_PATH}/accl.cpp
    ${ACCL_SOURCE_PATH}/common.cpp
    ${ACCL_SOURCE_PATH}/communicator.cpp
    ${ACCL_SOURCE_PATH}/constants.cpp
    ${ACCL_SOURCE_PATH}/simdevice.cpp
    ${ACCL_SOURCE_PATH}/simbuffer.cpp
    ${ACCL_SOURCE_PATH}/fpgadevice.cpp
    ${ZMQ_INTF_DIR}/zmq_client.cpp
    ${ZMQ_INTF_DIR}/zmq_common.cpp
)
endif()

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
if(EN_COYOTE)
    message("Enable Coyote")
    set(ACCL_HEADERS 
        ${ACCL_HEADERS} 
        ${ACCL_HEADER_PATH}/coyotebuffer.hpp
        ${ACCL_HEADER_PATH}/coyotedevice.hpp
    )
    if(COYOTE_HSA_SUPPORT)
        set(ACCL_SOURCES 
            ${ACCL_SOURCES} 
            ${ACCL_SOURCE_PATH}/coyotedevice.cpp
        )
        file(GLOB COYOTE_SOURCE "${COYOTE_SOURCE_PATH}/*.cpp")
    else()
        set(ACCL_SOURCES 
        ${ACCL_SOURCES} 
        ${ACCL_SOURCE_PATH}/coyotedevice.cpp
        )
        file(GLOB COYOTE_SOURCE "${COYOTE_SOURCE_PATH}/*.cpp")
    endif()

    if(EN_AVX)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -mavx -march=native -O3")
        set (CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -pthread -mavx -march=native -O3")
    else()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -march=native -O1")
        set (CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -pthread -march=native -O1")
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

if(COYOTE_HSA_SUPPORT)
    file(STRINGS "${ROCM_PATH}/.info/version" ROCM_VERSION)
    if(${ROCM_VERSION} VERSION_LESS "6.0")
        hip_add_library(accl STATIC ${ACCL_SOURCES})
    else()
        add_library(accl STATIC ${ACCL_SOURCES})
    endif()
else()
    add_library(accl STATIC ${ACCL_SOURCES})
endif()

target_include_directories(accl PUBLIC ${ACCL_INCLUDE_PATH})

if(COYOTE_HSA_SUPPORT)
    #hip, rocm, hsa
    target_include_directories(accl
        PUBLIC
            $<BUILD_INTERFACE:${ROCM_PATH}/include>
            $<BUILD_INTERFACE:${ROCM_PATH}/include/hsa>)
set_target_properties(accl PROPERTIES LINKER_LANGUAGE HIP)
target_link_libraries(accl PUBLIC hip::device  numa pthread drm drm_amdgpu rt dl hsa-runtime64 hsakmt)
add_compile_definitions(COYOTE_HSA_SUPPORT)
endif()

# XRT
if ((NOT EXISTS $ENV{XILINX_XRT}) AND XRT_REQUIRED)
  message(FATAL_ERROR "Xilinx XRT not found, make sure to source setup.sh")
endif ()
if ((NOT EXISTS $ENV{XILINX_XRT}) AND NOT XRT_REQUIRED)
  message("Xilinx XRT not found, make sure you don't need it")
endif ()



if(XRT_REQUIRED)
target_link_directories(accl PUBLIC $ENV{XILINX_XRT}/lib)
endif()


if(COYOTE_HSA_SUPPORT)  
    set_target_properties(accl PROPERTIES LINKER_LANGUAGE HIP)
    target_link_libraries(accl PUBLIC hip::device  numa pthread drm drm_amdgpu rt dl hsa-runtime64 hsakmt)
endif()

target_link_directories(accl PUBLIC /usr/local/lib)
if (XRT_REQUIRED)
    target_link_libraries(accl PUBLIC xilinxopencl xrt_coreutil xrt_core)
endif()


if(XRT_REQUIRED)
    target_include_directories(accl PUBLIC $ENV{XILINX_XRT}/include)
endif()
# ZMQ
target_link_libraries(accl PUBLIC zmqpp zmq pthread)

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
