
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

PERIPHERAL_IPS = hostctrl loopback reduce_ops hp_compression dummy_tcp_stack client_arbiter vadd_put cyt_adapter external_dma dummy_cyt_rdma_stack dummy_cyt_dma tcp_session_handler
TARGET=ip
PLATFORM ?= xilinx_u280_xdma_201920_3
DEBUG ?= none
STACK_TYPE ?= UDP
NUM_EXTDMA_AXI ?= 1

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

all: $(PERIPHERAL_IPS)


.PHONY: hostctrl loopback reduce_ops hp_compression dummy_tcp_stack client_arbiter vadd_put cyt_adapter external_dma dummy_cyt_rdma_stack dummy_cyt_dma tcp_session_handler

$(PERIPHERAL_IPS):
	$(MAKE) -C $@ DEVICE=$(FPGAPART) TARGET=$(TARGET) STACK_TYPE=$(STACK_TYPE)
