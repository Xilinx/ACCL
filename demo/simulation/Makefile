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

ACCL_REPO_ROOT=$(shell pwd)/../../
MPI_INCLUDES=-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIBPATHS=-L/usr/lib/x86_64-linux-gnu/openmpi/lib
MB_FW_DIR=$(ACCL_REPO_ROOT)/kernels/cclo/fw/sw_apps/ccl_offload_control/src

tb: testbench.cpp xsi_loader.cpp xsi_loader.h
	g++ -std=c++17 -fdiagnostics-color=always -g testbench.cpp xsi_loader.cpp -DZMQ_CALL_VERBOSE -I. -I${XILINX_VIVADO}/data/xsim/include/ -I${MPI_INCLUDES} -I${MB_FW_DIR} -o $@ -L${MPI_LIBPATHS} -ldl -lrt -lpthread -lmpi_cxx -lmpi -lzmqpp -lzmq -ljsoncpp

run: tb
	LD_LIBRARY_PATH=${XILINX_VIVADO}/lib/lnx64.o ./tb xsim.dir/ccl_offload/xsimk.so

distclean:
	git clean -xfd
