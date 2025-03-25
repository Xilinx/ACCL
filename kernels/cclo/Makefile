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
MODE ?= xo
EN_DMA ?= 1
EN_ARITH ?= 1
EN_COMPRESS ?= 1
EN_EXT_KRNL ?= 1
MB_DEBUG_LEVEL ?= 0
SIM_MEM_SIZE_LOG ?= 28
SIM_MEM_LATENCY ?= 50

FW_SOURCES = $(shell find fw -name '*.c') $(shell find fw -name '*.h') $(shell find fw -name '*.tcl')

ifeq ($(MODE), simdll)
	EN_DMA=1
	EN_ARITH=1
	EN_COMPRESS=1
	EN_EXT_KRNL=1
	MB_DEBUG_LEVEL=0
endif

ifeq (u250,$(findstring u250, $(PLATFORM)))
	FPGAPART=xcu250-figd2104-2L-e
	BOARD=u250
else ifeq (u280,$(findstring u280, $(PLATFORM)))
	FPGAPART=xcu280-fsvh2892-2L-e
	BOARD=u280
else ifeq (u55c,$(findstring u55c, $(PLATFORM)))
	FPGAPART=xcu55c-fsvh2892-2L-e
	BOARD=u55c
else ifeq (u200,$(findstring u200, $(PLATFORM)))
	FPGAPART=xcu200-fsgd2104-2-e
	BOARD=u200
else ifeq (u50,$(findstring u50, $(PLATFORM)))
	FPGAPART=xcu50-fsvh2104-2-e
	BOARD=u50
else
	$(error Unsupported PLATFORM)
endif

COMMIT_HASH := 0x$(shell git rev-parse HEAD | cut -c 1-6)
GEN_KERNEL_TCL := tcl/generate_kernel.tcl
REBUILD_BD_TCL := tcl/rebuild_bd.tcl tcl/control_bd.tcl tcl/rx_bd.tcl tcl/tx_bd.tcl

ifeq ($(MODE), simdll)
	FPGAPART=xcu280-fsvh2892-2L-e
	BOARD=u280
	BUILD_FOLDER = $(STACK_TYPE)_sim
else
	BUILD_FOLDER = $(STACK_TYPE)_$(EN_DMA)$(EN_ARITH)$(EN_COMPRESS)$(EN_EXT_KRNL)$(MB_DEBUG_LEVEL)_$(FPGAPART)
endif

CCLO_ELF=$(BUILD_FOLDER)/vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf
CCLO_SIMDLL=$(BUILD_FOLDER)/ccl_offload_ex/ccl_offload_ex.sim/sim_1/behav/xsim/xsim.dir/ccl_offload_behav/xsimk.so
CCLO_XSA=$(BUILD_FOLDER)/ccl_offload.xsa
CCLO_DCP=$(BUILD_FOLDER)/ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp
CCLO_XO=$(BUILD_FOLDER)/ccl_offload.xo

OUTPUT_PRODUCT=$(CCLO_XO)
ifeq ($(MODE), simdll)
	OUTPUT_PRODUCT=$(CCLO_SIMDLL)
endif

all: $(OUTPUT_PRODUCT)

.PHONY: xsa
xsa: $(CCLO_XSA)

$(CCLO_XSA): $(GEN_KERNEL_TCL) $(REBUILD_BD_TCL)
	$(MAKE) -C hls/ DEVICE=$(FPGAPART)
	mkdir -p $(BUILD_FOLDER)
	cd $(BUILD_FOLDER) && vivado -mode batch -source ../$< -tclargs $(FPGAPART) $(HW_DEBUG) ./ccl_offload.xsa $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(MB_DEBUG_LEVEL) $(COMMIT_HASH)

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

.PHONY: elf xo clean dcp

elf: $(CCLO_ELF)

$(CCLO_ELF): tcl/generate_sw.tcl tcl/associate_elf.tcl $(FW_SOURCES) $(CCLO_XSA)
	cd $(BUILD_FOLDER) && xsct ../tcl/generate_sw.tcl ccl_offload ./ccl_offload.xsa ../fw $(OPTIMIZATION_STRING)
	cd $(BUILD_FOLDER) && vivado -mode batch -source ../tcl/associate_elf.tcl -tclargs vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf

$(CCLO_SIMDLL): tcl/generate_sim.tcl elf
	$(MAKE) -C ../plugins DEVICE=$(FPGAPART)
	cd $(BUILD_FOLDER) && vivado -mode batch -source ../tcl/generate_sim.tcl -tclargs $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(SIM_MEM_SIZE_LOG) $(SIM_MEM_LATENCY)
	cd $(BUILD_FOLDER)/ccl_offload_ex/ccl_offload_ex.sim/sim_1/behav/xsim/ && ./compile.sh && ./elaborate.sh

dcp: $(CCLO_DCP)

$(CCLO_DCP): tcl/generate_dcp.tcl $(CCLO_ELF)
	cd $(BUILD_FOLDER) && vivado -mode batch -source ../tcl/generate_dcp.tcl -tclargs $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(MB_DEBUG_LEVEL)

xo: $(CCLO_XO)

$(BUILD_FOLDER)/ccl_offload.xml:
	python3 gen_xml.py $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL)
	mkdir -p $(BUILD_FOLDER)
	mv ccl_offload.xml $@

$(CCLO_XO): tcl/generate_xo.tcl $(BUILD_FOLDER)/ccl_offload.xml $(CCLO_DCP)
	rm -rf $(BUILD_FOLDER)/ccl_offload_ex/ccl_offload
	cd $(BUILD_FOLDER) && vivado -mode batch -source ../tcl/generate_xo.tcl -tclargs $(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL) $(MB_DEBUG_LEVEL)

clean:
	git clean -xfd
