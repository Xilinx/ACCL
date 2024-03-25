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

set(CMAKE_CXX_STANDARD 17)
set(ACCL_SIM_NUM_BANKS 2 CACHE STRING "Number of ACCL simulator memory banks (must correspond to simdll)")
set(ACCL_SIM_MEM_SIZE_KB 262144 CACHE STRING "Size of ACCL simulator memory, in KB (must correspond to simdll)")

#GTest config
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Set useful compile warnings
add_compile_options(-Wall -Wextra -Wno-unused-variable
                    -Wno-unused-but-set-variable -Wno-unused-parameter -march=ivybridge)

if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endif()

set(CMAKE_BUILD_TYPE Debug)

set(ACCL_NETWORK_UTILS_MPI 1)
set(ACCL_NETWORK_UTILS_DEBUG 1)
set(ACCL_DEBUG 1)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../driver/utils/accl_network_utils ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/accl_network_utils)

find_package(MPI REQUIRED)
add_executable(test ${CMAKE_CURRENT_LIST_DIR}/src/test.cpp ${CMAKE_CURRENT_LIST_DIR}/src/utility.cpp)
add_executable(stress ${CMAKE_CURRENT_LIST_DIR}/src/stress.cpp ${CMAKE_CURRENT_LIST_DIR}/src/utility.cpp)
add_executable(bench ${CMAKE_CURRENT_LIST_DIR}/src/bench.cpp ${CMAKE_CURRENT_LIST_DIR}/src/utility.cpp)
target_include_directories(test PUBLIC ${MPI_CXX_INCLUDE_PATH} ${CMAKE_CURRENT_LIST_DIR}/include/)
target_include_directories(stress PUBLIC ${MPI_CXX_INCLUDE_PATH}  ${CMAKE_CURRENT_LIST_DIR}/include/)
target_include_directories(bench PUBLIC ${MPI_CXX_INCLUDE_PATH} ${CMAKE_CURRENT_LIST_DIR}/include/)
# Explicitely link against vnx  because CMake doesn't support nested object libraries
# https://stackoverflow.com/a/71040950
target_link_libraries(test PUBLIC MPI::MPI_CXX accl_network_utils vnx GTest::gtest_main)
target_link_libraries(stress PUBLIC MPI::MPI_CXX accl_network_utils vnx GTest::gtest_main)
target_link_libraries(bench PUBLIC MPI::MPI_CXX accl_network_utils vnx GTest::gtest_main)
