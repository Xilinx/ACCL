cmake_minimum_required(VERSION 3.9)
project(xrt_test)

set(CMAKE_BUILD_TYPE Debug)

set(ACCL_DEBUG 1)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../driver/xrt/CMakeLists.txt)

find_package(MPI REQUIRED)
add_executable(test test.cpp)
target_include_directories(test PUBLIC ${MPI_CXX_INCLUDE_PATH} ${ACCL_INCLUDE_PATH} cxxopts/include)
target_link_libraries(test PUBLIC MPI::MPI_CXX accl)
