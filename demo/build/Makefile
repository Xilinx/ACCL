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
#xilinx_u250_xdma_201830_2 not supported 
#xilinx_u250_gen3x16_xdma_3_1_202020_1
#xilinx_u280_xdma_201920_3
XSA := $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))
DEBUG ?= all
MODE ?= tcp_cmac
FREQUENCY = 100
XCCL_XO=../../kernel/ccl_offload_ex/exports/ccl_offload.xo
VNX=xup_vitis_network_example

NETLAYERDIR = $(VNX)/NetLayers
CMACDIR     = $(VNX)/Ethernet
NETLAYERHLS = $(NETLAYERDIR)/100G-fpga-network-stack-core

FW_SOURCES = $(shell find fw -name '*.c') $(shell find fw -name '*.h') $(shell find fw -name '*.tcl')

#ETH_IF 0 goes through network switch, 1 direct connect
ETH_IF=0
CMAC_UDP_XO=$(VNX)/Ethernet/_x.$(PLATFORM)/cmac_$(ETH_IF).xo
UDP_XO=$(VNX)/NetLayers/_x.$(PLATFORM)/networklayer.xo
LOOPBACK_XO=hls/loopback/build_vnx_loopback/sol1/impl/export.xo
TCP_DUMMY_XO=hls/dummy_tcp_stack/build_tcp_stack/sol1/impl/export.xo
TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/network_krnl.xo
CMAC_TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/cmac_krnl.xo
XCLBIN=$(BUILD_DIR)/ccl_offload.xclbin
ARITH_XO=hls/arith/build_arith/sol1/impl/export.xo

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
	VNX_XO = $(CMAC_UDP_XO) $(UDP_XO)
	CONFIG += --user_ip_repo_paths $(CMAC_IP_FOLDER)
	CONFIG += --user_ip_repo_paths $(HLS_IP_FOLDER)
	CMAC_IP_FOLDER ?= $(shell readlink -f ./$(CMACDIR)/cmac)
else ifeq (vnx_dual,$(MODE))
	CMAC_IP_FOLDER = $(shell readlink -f ./$(CMACDIR)/cmac)
	CMAC_UDP_XO := $(VNX)/Ethernet/_x.$(PLATFORM)/cmac_0.xo
	CMAC_UDP_XO += $(VNX)/Ethernet/_x.$(PLATFORM)/cmac_1.xo
	ETH_IF := 3
	ifeq (u50,$(findstring u50, $(PLATFORM)))
		$(error Unsupported MODE for Alveo U50)
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
	CONFIG += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/$(VNX)/post_sys_link.tcl
	VNX_XO = $(CMAC_UDP_XO) $(UDP_XO)
	CONFIG += --user_ip_repo_paths $(CMAC_IP_FOLDER)
	CONFIG += --user_ip_repo_paths $(HLS_IP_FOLDER)
	CMAC_IP_FOLDER ?= $(shell readlink -f ./$(CMACDIR)/cmac)
else ifeq (loopback,$(MODE))
	ETH_IF  := none
	VNX_XO = $(LOOPBACK_XO)
else ifeq (dual,$(MODE))
	ETH_IF := none
	VNX_XO = 
else ifeq (tri,$(MODE))
	ETH_IF := none
	VNX_XO = $(TCP_DUMMY_XO) $(LOOPBACK_XO) $(ARITH_XO)
	ifeq (u250,$(findstring u250, $(PLATFORM)))
		CONFIGFILE = config/link_config_u280_tri.ini
	endif
	ifeq (u280,$(findstring u280, $(PLATFORM)))
		CONFIGFILE = config/link_config_u280_tri.ini
	endif
	
	CONFIG += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/tcl/post_sys_link_tri.tcl
else ifeq (quad,$(MODE))
	ETH_IF := none
	VNX_XO = $(TCP_DUMMY_XO) $(LOOPBACK_XO) $(ARITH_XO)
	ifeq (u250,$(findstring u250, $(PLATFORM)))
		CONFIGFILE = config/link_config_u280_quad.ini
	endif
	ifeq (u280,$(findstring u280, $(PLATFORM)))
		CONFIGFILE = config/link_config_u280_quad.ini
	endif
	
	CONFIG += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/tcl/post_sys_link_quad.tcl
else ifeq (tcp_dummy_dual, $(MODE))
	VNX_XO = $(TCP_DUMMY_XO)
	CONFIGFILE = config/link_config_tcp_dual.ini
	CONFIG := --config $(CONFIGFILE)
else ifeq (tcp_real_dual, $(MODE))
	VNX_XO = $(TCP_XO)  $(LOOPBACK_XO) $(ARITH_XO)
	CONFIGFILE = config/link_config_tcp_real_dual.ini
	CONFIG := --config $(CONFIGFILE)
	CONFIG += --user_ip_repo_paths Vitis_with_100Gbps_TCP-IP/build/fpga-network-stack/iprepo
else ifeq (tcp_cmac, $(MODE))
	CMAC_TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/cmac_krnl.xo
	VNX_XO = $(TCP_XO) $(CMAC_TCP_XO) $(LOOPBACK_XO) $(ARITH_XO)
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
	make -C xup_vitis_network_example/Ethernet DEVICE=$(PLATFORM) INTERFACE=$(ETH_IF) all

$(UDP_XO):
	git submodule update --init --recursive xup_vitis_network_example
	make -C xup_vitis_network_example/NetLayers DEVICE=$(PLATFORM) all

$(CMAC_TCP_XO):
	git submodule update --init --recursive Vitis_with_100Gbps_TCP-IP
	make -C Vitis_with_100Gbps_TCP-IP/ cmac_krnl DEVICE=$(PLATFORM) TEMP_DIR=_x.hw.$(XSA)/ XSA=$(XSA)

$(TCP_DUMMY_XO):
	make -C hls/dummy_tcp_stack DEVICE=$(FPGAPART) all

.PHONY: arith
arith: $(ARITH_XO)
$(ARITH_XO):
	make -C hls/arith DEVICE=$(FPGAPART) all

.PHONY: network_krnl
network_krnl: $(TCP_XO)

$(TCP_XO):
	git submodule update --init --recursive Vitis_with_100Gbps_TCP-IP
	make -C Vitis_with_100Gbps_TCP-IP/ network_krnl DEVICE=$(PLATFORM) TEMP_DIR=_x.hw.$(XSA)/ XSA=$(XSA)

$(XCLBIN): $(XCCL_XO) $(VNX_XO) $(CONFIGFILE)
	v++ --link --platform $(PLATFORM) --kernel_frequency $(FREQUENCY) --save-temps --temp_dir $(BUILD_DIR) $(CONFIG) -o $@ $(XCCL_XO) $(VNX_XO)

.PHONY: loopback
loopback: $(LOOPBACK_XO)

$(LOOPBACK_XO):
	make -C hls/loopback DEVICE=$(FPGAPART) loopback

.PHONY: kernel
kernel: $(XCCL_XO)

$(XCCL_XO):
	make -C ../../kernel PLATFORM=$(PLATFORM) DEBUG=$(DEBUG) MODE=$(MODE)

distclean:
	git clean -xfd
