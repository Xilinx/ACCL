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
HLSLIB_INCLUDE=$(ACCL_REPO_ROOT)/hlslib/include/hlslib/xilinx/
MPI_INCLUDES=-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIBPATHS=-L/usr/lib/x86_64-linux-gnu/openmpi/lib
MB_FW_DIR=$(ACCL_REPO_ROOT)/kernels/cclo/fw/sw_apps/ccl_offload_control/src
CCLO_HLS_ROOT=$(ACCL_REPO_ROOT)/kernels/cclo/hls
ZMQ_DIR=$(ACCL_REPO_ROOT)/test/zmq

INCLUDES = -I$(XILINX_HLS)/include/ -I. -I${XILINX_VIVADO}/data/xsim/include/ -I${HLSLIB_INCLUDE} -I${MPI_INCLUDES} -I${MB_FW_DIR} -I${ZMQ_DIR} -I${CCLO_HLS_ROOT}
XSIMK_PATH_TAIL = xsim.dir/ccl_offload_behav/xsimk.so
XSIM_COMPILE_FOLDER ?= $(ACCL_REPO_ROOT)/kernels/cclo/ccl_offload_ex/ccl_offload_ex.sim/sim_1/behav/xsim/

SYMLINKS := $(shell ls $(XSIM_COMPILE_FOLDER) | grep -E '\.mem') $(shell ls $(XSIM_COMPILE_FOLDER) | grep -E '\.dat')

STACKTYPE ?= "udp"
NRANKS ?= 1
START_PORT ?= 5500

all: cclo_sim

cclo_sim: cclo_sim.cpp xsi_loader.cpp xsi_loader.h xsi_dut.cpp xsi_dut.h
	g++ -std=c++17 -fmax-errors=3 -fdiagnostics-color=always -g cclo_sim.cpp xsi_dut.cpp xsi_loader.cpp ${ZMQ_DIR}/*.cpp -DZMQ_CALL_VERBOSE ${INCLUDES} -o $@ -L${MPI_LIBPATHS} -ldl -lrt -lpthread -lmpi_cxx -lmpi -lzmqpp -lzmq -ljsoncpp

.PHONY = symlinks

symlinks: $(SYMLINKS)

$(SYMLINKS):
	ln -s ${XSIM_COMPILE_FOLDER}/$@ $@

run: cclo_sim $(SYMLINKS)
	LD_LIBRARY_PATH=${XILINX_VIVADO}/lib/lnx64.o mpirun -np ${NRANKS} ./cclo_sim ${STACKTYPE} ${START_PORT} ${XSIMK_PATH_TAIL}

clean:
	-rm $(SYMLINKS) cclo_sim *.wdb vivado*

distclean:
	git clean -xfd
