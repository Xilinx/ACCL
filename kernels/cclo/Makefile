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
DEBUG ?= none
XCCL_ELF=vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf
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
REBUILD_BD_TCL := tcl/rebuild_bd.tcl


all: $(XCCL_XO)

.PHONY: xsa
xsa: $(XCCL_XSA)

$(XCCL_XSA): $(GEN_KERNEL_TCL) $(REBUILD_BD_TCL)  tcl/control_bd.tcl tcl/dma_bd.tcl tcl/rx_bd.tcl tcl/tx_bd.tcl
	$(MAKE) -C hls/hostctrl DEVICE=$(FPGAPART)
	$(MAKE) -C hls/vnx_intf DEVICE=$(FPGAPART)
	$(MAKE) -C hls/tcp_intf DEVICE=$(FPGAPART)
	vivado -mode batch -source $< -tclargs $(FPGAPART) $(BOARDPART) $(DEBUG) $(XCCL_XSA)

OPTIMIZATION=0
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

.PHONY: elf
elf: $(XCCL_ELF)

$(XCCL_ELF): tcl/generate_sw.tcl $(FW_SOURCES) $(XCCL_XSA)
	rm -rf vitis_ws && mkdir vitis_ws
	xsct $< ccl_offload $(XCCL_XSA) ./fw $(OPTIMIZATION_STRING)

.PHONY: xo
xo: $(XCCL_XO)

$(XCCL_XO): tcl/generate_xo.tcl $(XCCL_XSA) $(XCCL_ELF)
	vivado -mode batch -source $< -tclargs $(XCCL_ELF)

distclean:
	git clean -xfd
