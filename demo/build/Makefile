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
XSA := $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))
DEBUG ?= all
PROFILE ?= all
MODE ?= tcp
FREQUENCY = 250
XCCL_XO=../../kernel/ccl_offload_ex/exports/ccl_offload.xo

VNX=xup_vitis_network_example
NETLAYERDIR = $(VNX)/NetLayers
CMACDIR     = $(VNX)/Ethernet
NETLAYERHLS = $(NETLAYERDIR)/100G-fpga-network-stack-core

FW_SOURCES = $(shell find fw -name '*.c') $(shell find fw -name '*.h') $(shell find fw -name '*.tcl')

#At XACC ETHZ: ETH_IF 0 goes through network switch, 1 is direct connect
ETH_IF=0
CMAC_UDP_XO=$(VNX)/Ethernet/_x.$(PLATFORM)/cmac_$(ETH_IF).xo
UDP_XO=$(VNX)/NetLayers/_x.$(PLATFORM)/networklayer.xo
LOOPBACK_XO=hls/loopback/build_vnx_loopback/sol1/impl/export.xo
TCP_DUMMY_XO=hls/dummy_tcp_stack/build_tcp_stack/sol1/impl/export.xo
TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/network_krnl.xo
CMAC_TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/cmac_krnl.xo
XCLBIN=$(BUILD_DIR)/ccl_offload.xclbin
HP_SUM_XO=hls/hp_stream_add/build_hp_stream_add/sol1/impl/export.xo
HP_UPCAST_XO=hls/hp_fp_stream_conv/build_hp_fp_stream_conv/sol1/impl/export.xo
HP_DOWNCAST_XO=hls/fp_hp_stream_conv/build_fp_hp_stream_conv/sol1/impl/export.xo
COMPRESSION_XO = $(HP_SUM_XO) $(HP_UPCAST_XO) $(HP_DOWNCAST_XO)

ifeq (all,$(PROFILE))
	PROFILE_OPTS := --profile.data all:all:all --profile.exec all:all:all --profile.stall all:all:all
else ifeq (data,$(PROFILE))
	PROFILE_OPTS := --profile.data all:all:all
else ifeq (exec,$(PROFILE))
	PROFILE_OPTS := --profile.exec all:all:all
else ifeq (stall,$(PROFILE))
	PROFILE_OPTS := --profile.stall all:all:all
else ifeq (none,$(PROFILE))
	PROFILE_OPTS :=
else
	$(error Unsupported PROFILE setting)
endif

ifeq (u250,$(findstring u250, $(PLATFORM)))
	FPGAPART=xcu250-figd2104-2L-e
	BOARD=u250
else ifeq (u280,$(findstring u280, $(PLATFORM)))
	FPGAPART=xcu280-fsvh2892-2L-e
	BOARD=u280
else
	$(error Unsupported PLATFORM)
endif

CONFIGFILE := config/link_config_$(BOARD)_$(MODE).ini
CONFIG := --config $(CONFIGFILE)
ifeq (vnx,$(MODE))
	CMAC_IP_FOLDER = $(shell readlink -f ./$(CMACDIR)/cmac)
	ifeq (u50,$(findstring u50, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_HMB)
	endif
	ifeq (u200,$(findstring u200, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_noHMB)
	endif
	ifeq (u250,$(findstring u250, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_noHMB)
	endif
	ifeq (u280,$(findstring u280, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_HMB)
	endif
	CONFIG += --connectivity.nk cmac_$(ETH_IF):1:cmac_inst
	CONFIG += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/$(VNX)/post_sys_link.tcl
	OTHER_XO = $(CMAC_UDP_XO) $(UDP_XO) $(COMPRESSION_XO)
	CONFIG += --user_ip_repo_paths $(CMAC_IP_FOLDER)
	CONFIG += --user_ip_repo_paths $(HLS_IP_FOLDER)
else ifeq (tri,$(MODE))
	ETH_IF := none
	OTHER_XO = $(TCP_DUMMY_XO) $(COMPRESSION_XO)
	ifeq (u250,$(findstring u250, $(PLATFORM)))
		CONFIGFILE = config/link_config_u250_tri.ini
	endif
	ifeq (u280,$(findstring u280, $(PLATFORM)))
		CONFIGFILE = config/link_config_u280_tri.ini
	endif
	CONFIG += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/tcl/post_sys_link_tri.tcl
else ifeq (tcp, $(MODE))
	CMAC_TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/cmac_krnl.xo
	OTHER_XO = $(TCP_XO) $(CMAC_TCP_XO) $(LOOPBACK_XO) $(COMPRESSION_XO)
	CONFIGFILE = config/link_config_$(BOARD)_tcp_cmac.ini
	CONFIG := --config $(CONFIGFILE) 
	CONFIG += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/Vitis_with_100Gbps_TCP-IP/scripts/post_sys_link.tcl
	CONFIG += --user_ip_repo_paths Vitis_with_100Gbps_TCP-IP/build/fpga-network-stack/iprepo
else
	$(error Unsupported MODE)
endif

BUILD_DIR := link_$(MODE)_eth_$(ETH_IF)_debug_$(DEBUG)_$(XSA)

all: $(XCLBIN)

.PHONY: vnx
vnx: $(CMAC_UDP_XO) $(UDP_XO)

$(CMAC_UDP_XO) &:	
	git submodule update --init --recursive xup_vitis_network_example	
	$(MAKE) -C xup_vitis_network_example/Ethernet DEVICE=$(PLATFORM) INTERFACE=$(ETH_IF) all

$(UDP_XO):
	git submodule update --init --recursive xup_vitis_network_example
	$(MAKE) -C xup_vitis_network_example/NetLayers DEVICE=$(PLATFORM) all

$(CMAC_TCP_XO):
	git submodule update --init --recursive Vitis_with_100Gbps_TCP-IP
	$(MAKE) -C Vitis_with_100Gbps_TCP-IP/ cmac_krnl DEVICE=$(PLATFORM) TEMP_DIR=_x.hw.$(XSA)/ XSA=$(XSA)

$(TCP_DUMMY_XO):
	$(MAKE) -C hls/dummy_tcp_stack DEVICE=$(FPGAPART) all

.PHONY: compression
compression: $(COMPRESSION_XO)
$(HP_SUM_XO):
	$(MAKE) -C hls/hp_stream_add DEVICE=$(FPGAPART) all
$(HP_DOWNCAST_XO):
	$(MAKE) -C hls/fp_hp_stream_conv DEVICE=$(FPGAPART) all
$(HP_UPCAST_XO):
	$(MAKE) -C hls/hp_fp_stream_conv DEVICE=$(FPGAPART) all

.PHONY: network_krnl
network_krnl: $(TCP_XO)

$(TCP_XO):
	git submodule update --init --recursive Vitis_with_100Gbps_TCP-IP
	$(MAKE) -C Vitis_with_100Gbps_TCP-IP/ network_krnl DEVICE=$(PLATFORM) TEMP_DIR=_x.hw.$(XSA)/ XSA=$(XSA)

$(XCLBIN): $(XCCL_XO) $(OTHER_XO) $(CONFIGFILE)
	v++ --link --platform $(PLATFORM) --kernel_frequency $(FREQUENCY) --save-temps --temp_dir $(BUILD_DIR) $(CONFIG) -o $@ $(XCCL_XO) $(OTHER_XO)

.PHONY: loopback
loopback: $(LOOPBACK_XO)

$(LOOPBACK_XO):
	$(MAKE) -C hls/loopback DEVICE=$(FPGAPART) loopback

.PHONY: kernel
kernel: $(XCCL_XO)

$(XCCL_XO):
	$(MAKE) -C ../../kernel PLATFORM=$(PLATFORM) DEBUG=$(DEBUG) MODE=$(MODE)

distclean:
	git clean -xfd
