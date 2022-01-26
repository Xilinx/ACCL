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

PLATFORM ?= xilinx_u280_xdma_201920_3
HW_DEBUG ?= none
STACK_TYPE ?= UDP
EN_DMA ?= 1
EN_ARITH ?= 1
EN_COMPRESS ?= 1
EN_EXT_KRNL ?= 1
MB_DEBUG_LEVEL ?= 0
SIM_MEMSIZE ?= 256K
XCCL_ELF=vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf
XCCL_SIMDLL=ccl_offload_ex/ccl_offload_ex.sim/sim_1/behav/xsim/xsim.dir/ccl_offload_behav/xsimk.so
XCCL_XSA=ccl_offload_ex/ccl_offload.xsa
XCCL_XO=ccl_offload.xo

FW_SOURCES = $(shell find fw -name '*.c') $(shell find fw -name '*.h') $(shell find fw -name '*.tcl')

ifeq (u250,$(findstring u250, $(PLATFORM)))
	FPGAPART=xcu250-figd2104-2L-e
	BOARDPART=xilinx.com:au250:part0:1.3
else ifeq (u280,$(findstring u280, $(PLATFORM)))
	FPGAPART=xcu280-fsvh2892-2L-e
	BOARDPART=xilinx.com:au280:part0:1.1
else
	$(error Unsupported PLATFORM)
endif

GEN_KERNEL_TCL := tcl/generate_kernel.tcl
REBUILD_BD_TCL := tcl/rebuild_bd.tcl tcl/control_bd.tcl tcl/rx_bd.tcl tcl/tx_bd.tcl

all: $(XCCL_XO)

.PHONY: xsa
xsa: $(XCCL_XSA)

$(XCCL_XSA): $(GEN_KERNEL_TCL) $(REBUILD_BD_TCL)
	$(MAKE) -C hls/ DEVICE=$(FPGAPART)
	vivado -mode batch -source $< -tclargs $(FPGAPART) $(BOARDPART) $(HW_DEBUG) $(XCCL_XSA) $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(MB_DEBUG_LEVEL)

OPTIMIZATION=s
#valid values 0,1,2,3,s,g
ifeq ($(OPTIMIZATION),0)
   OPTIMIZATION_STRING = "None (-O0)"
else ifeq ($(OPTIMIZATION),1)
   OPTIMIZATION_STRING = "Optimize (-O1)"
else ifeq ($(OPTIMIZATION),2)
   OPTIMIZATION_STRING = "Optimize more (-O2)"
else ifeq ($(OPTIMIZATION),3)
   OPTIMIZATION_STRING = "Optimize most (-O3)"
else ifeq ($(OPTIMIZATION),s)
   OPTIMIZATION_STRING = "Optimize for size (-Os)"
endif

.PHONY: elf simdll xo distclean

elf: $(XCCL_ELF)

$(XCCL_ELF): tcl/generate_sw.tcl tcl/associate_elf.tcl $(FW_SOURCES) $(XCCL_XSA)
	rm -rf vitis_ws && mkdir vitis_ws
	xsct $< ccl_offload $(XCCL_XSA) ./fw $(OPTIMIZATION_STRING)
	vivado -mode batch -source tcl/associate_elf.tcl -tclargs $(XCCL_ELF)

simdll: $(XCCL_SIMDLL)

$(XCCL_SIMDLL): tcl/generate_sim.tcl elf
	vivado -mode batch -source $< -tclargs $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(SIM_MEMSIZE)
	cd ccl_offload_ex/ccl_offload_ex.sim/sim_1/behav/xsim/ && ./compile.sh && ./elaborate.sh

xo: $(XCCL_XO)

$(XCCL_XO): tcl/generate_xo.tcl $(XCCL_XSA) $(XCCL_ELF)
	vivado -mode batch -source $< -tclargs $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(MB_DEBUG_LEVEL)

distclean:
	git clean -xfd
