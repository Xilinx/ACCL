#Copyright (c) 2024, Advanced Micro Devices, Inc.
#All rights reserved.
cmake_minimum_required(VERSION 3.16)

if (NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /opt/rocm-6.0.2/bin/hipcc)
endif()


set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

set(XRT_REQUIRED OFF)
set(COYOTE_HSA_SUPPORT OFF CACHE BOOL "enables the HSA runtime - HSA environment required")
if(XRT_REQUIRED AND COYOTE_HSA_SUPPORT)
    message(FATAL_ERROR "XRT and HSA are not supported in combination.")
endif()    

if(COYOTE_HSA_SUPPORT)
    message("check HSA")
    set(AMD_GPU "gfx908" CACHE STRING "AMD GPU device to use")
    add_compile_definitions(COYOTE_HSA_SUPPORT)
endif()

message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

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
endif()

if(COYOTE_HSA_SUPPORT)
    project(accl_on_coyote LANGUAGES CXX HIP)
else()
    project(accl_on_coyote LANGUAGES CXX)
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


# set(COYOTE_DRIVER_DIR ${CMAKE_SOURCE_DIR}/../../refdesigns/Coyote/sw/)
set(ACCL_DRIVER_DIR ${CMAKE_SOURCE_DIR}/../../../driver/xrt/)

set(ACCL_DEBUG 1)

# Sources
# file(GLOB SOURCES ${COYOTE_DRIVER_DIR}/src/*.cpp ${ACCL_DRIVER_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/*.cpp)

# Enable Coyote
# set(EN_COYOTE 1 CACHE STRING "Enable Coyote")
# AVX support (Disable on Enzian)
set(EN_AVX 1 CACHE STRING "AVX environment.")
add_compile_definitions(COYOTE_DEBUG)
add_subdirectory(${CMAKE_SOURCE_DIR}/../../../driver/xrt/ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/xrt/)

if(COYOTE_HSA_SUPPORT)
    add_compile_definitions( __HIP_PLATFORM_AMD__ )
endif()
# Exec
set(EXEC accl_on_coyote)

if(COYOTE_HSA_SUPPORT)
    file(STRINGS "${ROCM_PATH}/.info/version" ROCM_VERSION)
    if(${ROCM_VERSION} VERSION_LESS "6.0")
        hip_add_executable(${EXEC} ${CMAKE_SOURCE_DIR}/test.cpp)
    else()
        add_executable(${EXEC} ${CMAKE_SOURCE_DIR}/test.cpp)
    endif()

    #hip, rocm, hsa
    target_include_directories(${EXEC}
    PUBLIC
        $<BUILD_INTERFACE:${ROCM_PATH}/include>
        $<BUILD_INTERFACE:${ROCM_PATH}/include/hsa>)
      


#acclgpu
if(COYOTE_HSA_SUPPORT)
add_subdirectory(${CMAKE_SOURCE_DIR}/../../../driver/gpu/ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/gpu/)
endif()

target_include_directories(${EXEC} PUBLIC ${ACCLGPU_DRIVER_DIR}/include)
target_link_libraries(${EXEC} PUBLIC acclgpu)  
else()
    add_executable(${EXEC} ${CMAKE_SOURCE_DIR}/test.cpp)
endif()





# accl
target_include_directories(${EXEC} PUBLIC ${ACCL_DRIVER_DIR}/include)
target_link_libraries(${EXEC} PUBLIC accl)

# MPI
#set(MPI_CXX_HEADER_DIR /scratch/users/gsorrent/mpich_install/include)
#set(MPI_mpi_LIBRARY /scratch/users/gsorrent/mpich_install/lib/libmpi.so.12.4.0)
#set(MPI_mpicxx_LIBRARY /scratch/users/gsorrent/mpich_install/lib/libmpicxx.so)
#set(MPI_INCLUDE_PATH /scratch/users/gsorrent/mpich_install) 
#set(MPI_HOME /scratch/users/gsorrent/mpich_install) # this is the one required
#

# MPI
find_package(MPI REQUIRED)
message(STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_PATH}")

target_include_directories(${EXEC} PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(${EXEC} PUBLIC MPI::MPI_CXX)


#set_property(TARGET ${EXEC} PROPERTY CXX_STANDARD 20)
if(COYOTE_HSA_SUPPORT)
    set_property(TARGET ${EXEC} PROPERTY HIP_ARCHITECTURES ${AMD_GPU})
endif()

if (ACCL_DEBUG)
  target_compile_definitions(${EXEC} PUBLIC ACCL_DEBUG)
  message("Defining ACCL_DEBUG")
endif (ACCL_DEBUG)

target_link_directories(${EXEC} PUBLIC /usr/local/lib)



if(COYOTE_HSA_SUPPORT)  
    target_link_libraries(${EXEC} PUBLIC hip::device  numa pthread drm drm_amdgpu rt dl hsa-runtime64 hsakmt)
endif()


# ZMQ
target_link_libraries(${EXEC} PUBLIC zmqpp zmq pthread)

# # Json
# find_package(jsoncpp REQUIRED)
# target_link_libraries(${EXEC} PUBLIC jsoncpp_lib)
# get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
# target_include_directories(${EXEC} PUBLIC ${JSON_INC_PATH})


