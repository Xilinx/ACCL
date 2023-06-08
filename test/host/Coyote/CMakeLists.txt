cmake_minimum_required(VERSION 3.16)
project(accl_on_coyote)

# set(COYOTE_DRIVER_DIR ${CMAKE_SOURCE_DIR}/../../refdesigns/Coyote/sw/)
set(ACCL_DRIVER_DIR ${CMAKE_SOURCE_DIR}/../../../driver/xrt/)

set(ACCL_DEBUG 1)

# Sources
# file(GLOB SOURCES ${COYOTE_DRIVER_DIR}/src/*.cpp ${ACCL_DRIVER_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/*.cpp)

# Enable Coyote
# set(EN_COYOTE 1 CACHE STRING "Enable Coyote")
# AVX support (Disable on Enzian)
set(EN_AVX 1 CACHE STRING "AVX environment.")

add_subdirectory(${CMAKE_SOURCE_DIR}/../../../driver/xrt/ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/xrt/)


# Exec
set(EXEC accl_on_coyote)
add_executable(${EXEC} ${CMAKE_SOURCE_DIR}/test.cpp)

# accl
target_include_directories(${EXEC} PUBLIC ${ACCL_DRIVER_DIR}/include)
target_link_libraries(${EXEC} PUBLIC accl)

# MPI
find_package(MPI REQUIRED)
message(STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_PATH}")

target_include_directories(${EXEC} PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(${EXEC} PUBLIC MPI::MPI_CXX)

# XRT
if (NOT EXISTS $ENV{XILINX_XRT})
  message(FATAL_ERROR "Xilinx XRT not found, make sure to source setup.sh")
endif ()

target_link_directories(${EXEC} PUBLIC $ENV{XILINX_XRT}/lib)
target_link_libraries(${EXEC} PUBLIC xilinxopencl xrt_coreutil xrt_core)
target_include_directories(${EXEC} PUBLIC $ENV{XILINX_XRT}/include)

# ZMQ
target_link_libraries(${EXEC} PUBLIC zmqpp zmq pthread)

# # Json
# find_package(jsoncpp REQUIRED)
# target_link_libraries(${EXEC} PUBLIC jsoncpp_lib)
# get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
# target_include_directories(${EXEC} PUBLIC ${JSON_INC_PATH})


