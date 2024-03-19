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
SIM_MEM_SIZE_LOG ?= 28
SIM_MEM_LATENCY ?= 50
CCLO_ELF=vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf
CCLO_SIMDLL=ccl_offload_ex/ccl_offload_ex.sim/sim_1/behav/xsim/xsim.dir/ccl_offload_behav/xsimk.so
CCLO_XSA=ccl_offload_ex/ccl_offload.xsa
CCLO_DCP=ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp
CCLO_XO=ccl_offload.xo

FW_SOURCES = $(shell find fw -name '*.c') $(shell find fw -name '*.h') $(shell find fw -name '*.tcl')

ifeq (u250,$(findstring u250, $(PLATFORM)))
	FPGAPART=xcu250-figd2104-2L-e
else ifeq (u280,$(findstring u280, $(PLATFORM)))
	FPGAPART=xcu280-fsvh2892-2L-e
else ifeq (u55c,$(findstring u55c, $(PLATFORM)))
	FPGAPART=xcu55c-fsvh2892-2L-e
else ifeq (u200,$(findstring u200, $(PLATFORM)))
	FPGAPART=xcu200-fsgd2104-2-e
else ifeq (u50,$(findstring u50, $(PLATFORM)))
	FPGAPART=xcu50-fsvh2104-2-e
else
	$(error Unsupported PLATFORM)
endif

COMMIT_HASH := 0x$(shell git rev-parse HEAD | cut -c 1-6)
GEN_KERNEL_TCL := tcl/generate_kernel.tcl
REBUILD_BD_TCL := tcl/rebuild_bd.tcl tcl/control_bd.tcl tcl/rx_bd.tcl tcl/tx_bd.tcl

all: $(CCLO_XO)

.PHONY: xsa
xsa: $(CCLO_XSA)

$(CCLO_XSA): $(GEN_KERNEL_TCL) $(REBUILD_BD_TCL)
	$(MAKE) -C hls/ DEVICE=$(FPGAPART)
	vivado -mode batch -source $< -tclargs $(FPGAPART) $(HW_DEBUG) $(CCLO_XSA) $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(MB_DEBUG_LEVEL) $(COMMIT_HASH)

OPTIMIZATION=3
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

.PHONY: elf simdll xo clean dcp

elf: $(CCLO_ELF)

$(CCLO_ELF): tcl/generate_sw.tcl tcl/associate_elf.tcl $(FW_SOURCES) $(CCLO_XSA)
	rm -rf vitis_ws && mkdir vitis_ws
	xsct $< ccl_offload $(CCLO_XSA) ./fw $(OPTIMIZATION_STRING)
	vivado -mode batch -source tcl/associate_elf.tcl -tclargs $(CCLO_ELF)

simdll: $(CCLO_SIMDLL)

$(CCLO_SIMDLL): tcl/generate_sim.tcl elf
	$(MAKE) -C ../plugins PLATFORM=$(PLATFORM) NUM_EXTDMA_AXI=2
	vivado -mode batch -source $< -tclargs $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(SIM_MEM_SIZE_LOG) $(SIM_MEM_LATENCY)
	cd ccl_offload_ex/ccl_offload_ex.sim/sim_1/behav/xsim/ && ./compile.sh && ./elaborate.sh

dcp: $(CCLO_DCP)

$(CCLO_DCP): tcl/generate_dcp.tcl $(CCLO_ELF)
	vivado -mode batch -source $< -tclargs $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(MB_DEBUG_LEVEL)

xo: $(CCLO_XO)

ccl_offload.xml:
	python3 gen_xml.py $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL)

$(CCLO_XO): tcl/generate_xo.tcl ccl_offload.xml $(CCLO_DCP)
	rm -rf ccl_offload_ex/ccl_offload
	vivado -mode batch -source $< -tclargs $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(MB_DEBUG_LEVEL)

clean:
	$(MAKE) -C hls/ clean
	rm -rf ccl_offload_ex ccl_offload.xml ccl_offload.xo vitis_ws .Xil *.jou *.log
