# /*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
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
NUM_DMA ?= 1

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

all: vitis_external_dma.xo

vitis_external_dma.xo: bd.tcl kernel.xml
	vivado -mode batch -source $< -notrace -tclargs $(FPGAPART) $(NUM_DMA)

check-devices:
ifndef DEVICE
	$(error DEVICE not set. Please set the DEVICE properly and rerun. Run "make help" for more details.)
endif

#Checks for XILINX_VIVADO
check-vivado:
ifndef XILINX_VIVADO
	$(error XILINX_VIVADO variable is not set, please set correctly and rerun)
endif

check-interface:
ifeq (2,$(INTERFACE))
	$(error INTERFACE cannot value 2, to use generate both interfaces xo use INTERFACE=3)
endif
