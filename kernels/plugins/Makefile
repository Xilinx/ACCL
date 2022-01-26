
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

PERIPHERAL_IPS = loopback reduce_sum fp_hp_stream_conv hp_fp_stream_conv dummy_tcp_stack
TARGET=ip
PLATFORM ?= xilinx_u280_xdma_201920_3
DEBUG ?= none

ifeq (u250,$(findstring u250, $(PLATFORM)))
	FPGAPART=xcu250-figd2104-2L-e
else ifeq (u280,$(findstring u280, $(PLATFORM)))
	FPGAPART=xcu280-fsvh2892-2L-e
else
	$(error Unsupported PLATFORM)
endif

all: $(PERIPHERAL_IPS) cclo

.PHONY: loopback cclo reduce_sum fp_hp_stream_conv hp_fp_stream_conv dummy_tcp_stack

$(PERIPHERAL_IPS):
	$(MAKE) -C $@ DEVICE=$(FPGAPART) TARGET=$(TARGET)

cclo:
	$(MAKE) -C $@ PLATFORM=$(FPGAPART) DEBUG=$(DEBUG)
