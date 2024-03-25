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

PLATFORM ?= xilinx_u55c_gen3x16_xdma_3_202210_1
# Supported Alveo platforms are:
# U50: xilinx_u50_gen3x16_xdma_5_202210_1
# U200: xilinx_u200_gen3x16_xdma_2_202110_1
# U55C: xilinx_u55c_gen3x16_xdma_3_202210_1
# U280: xilinx_u280_gen3x16_xdma_1_202211_1
# U250: xilinx_u250_gen3x16_xdma_4_1_202210_1

XSA := $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))
DEBUG ?= none
PROFILE ?= none
MODE ?= tcp
USE_HOSTMEM ?= 0
VPP_TARGET ?= hw
USER_KERNEL ?= none
FREQUENCY = 250
CCLO_XO = ../../kernels/cclo/ccl_offload.xo
CCLO_MB_DEBUG_LEVEL ?= 0

VNX=xup_vitis_network_example
NETLAYERDIR = $(VNX)/NetLayers
CMACDIR     = $(VNX)/Ethernet
NETLAYERHLS = $(NETLAYERDIR)/100G-fpga-network-stack-core

FW_SOURCES = $(shell find fw -name '*.c') $(shell find fw -name '*.h') $(shell find fw -name '*.tcl')

#At XACC ETHZ: ETH_IF 0 goes through network switch, 1 is direct connect
ETH_IF=0
CMAC_UDP_XO=$(VNX)/Ethernet/_x.$(PLATFORM)/cmac_$(ETH_IF).xo
UDP_XO=$(VNX)/NetLayers/_x.$(PLATFORM)/networklayer.xo
TCP_DUMMY_XO=../../kernels/plugins/dummy_tcp_stack/dummy_tcp_stack.xo
TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/network_krnl.xo
TCP_VIVADO_ROOTDIR ?= $$XILINX_VIVADO
TCP_HLS_ROOTDIR ?= $$XILINX_HLS
TCP_RX_BYPASS ?= 0
CMAC_TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/cmac_krnl.xo
EXTERNAL_DMA_XO=../../kernels/plugins/external_dma/external_dma.xo
CYT_BIT=Coyote/hw/build_$(CCLO_STACK_TYPE)/bitstreams/cyt_top.bit

HOSTCTRL_XO=../../kernels/plugins/hostctrl/hostctrl.xo
CLIENT_ARB_XO=../../kernels/plugins/client_arbiter/client_arbiter.xo
SUM_XO=../../kernels/plugins/reduce_ops/reduce_ops.xo
COMPRESSION_XO=../../kernels/plugins/hp_compression/hp_compression.xo
LOOPBACK_XO=../../kernels/plugins/loopback/loopback.xo
TCP_SESS_XO=../../kernels/plugins/tcp_session_handler/tcp_session_handler.xo
VADD_XO=../../kernels/plugins/vadd_put/vadd_put.xo

HWEMU_MST_XO=$$XILINX_VITIS/data/emulation/XO/sim_ipc_axis_master_512.xo
HWEMU_SLV_XO=$$XILINX_VITIS/data/emulation/XO/sim_ipc_axis_slave_512.xo
HWEMU_XO = $(HWEMU_MST_XO) $(HWEMU_SLV_XO)

ARITHMETIC_XO = $(SUM_XO) $(COMPRESSION_XO)

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

BUILD_DIR := link_$(MODE)_eth_$(ETH_IF)_debug_$(DEBUG)_$(XSA)
XCLBIN=$(BUILD_DIR)/ccl_offload.xclbin
OUTPUT_PRODUCT := $(XCLBIN)

CONFIGFILE := link_config.ini
VPP_CONFIG := -t $(VPP_TARGET) --config $(CONFIGFILE)

ADV_CMD :=
IPREPO_CMD :=
OTHER_XO := $(CCLO_XO) $(HOSTCTRL_XO) $(CLIENT_ARB_XO) $(SUM_XO) $(COMPRESSION_XO) $(EXTERNAL_DMA_XO)

GEN_CONFIG_FLAGS := 
ifeq (vadd, $(USER_KERNEL))
	GEN_CONFIG_FLAGS += --vadd
	OTHER_XO += $(VADD_XO)
else
	OTHER_XO += $(LOOPBACK_XO)
endif

ifeq (hw_emu, $(VPP_TARGET))
	VPP_CONFIG += -g
	GEN_CONFIG_FLAGS += --hwemu
	ifneq (axis3x, $(MODE))
		OTHER_XO += $(HWEMU_XO)
	endif
endif

ifeq (udp,$(MODE))
	ifeq (u50,$(findstring u50, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_HBM)
	endif
	ifeq (u55c,$(findstring u55c, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_HBM)
	endif
	ifeq (u200,$(findstring u200, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_noHBM)
	endif
	ifeq (u250,$(findstring u250, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_noHBM)
	endif
	ifeq (u280,$(findstring u280, $(PLATFORM)))
		HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERHLS)/synthesis_results_HBM)
	endif
	NET_XO = $(UDP_XO)
	ifneq (hw_emu, $(VPP_TARGET))
		NET_XO += $(CMAC_UDP_XO)
		ADV_CMD += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/$(VNX)/Ethernet/post_sys_link.tcl
	endif
	IPREPO_CMD += --user_ip_repo_paths $(HLS_IP_FOLDER)
	CCLO_STACK_TYPE = UDP
else ifeq (axis3x,$(MODE))
	ETH_IF := none
	NET_XO = $(TCP_DUMMY_XO)
	ADV_CMD += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/tcl/post_sys_link_tri.tcl
	CCLO_STACK_TYPE = TCP
	CCLO_MB_DEBUG_LEVEL = 2
else ifeq (tcp, $(MODE))
	CMAC_TCP_XO=Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/cmac_krnl.xo
	NET_XO = $(TCP_XO)
	ifneq (hw_emu, $(VPP_TARGET))
		NET_XO += $(CMAC_TCP_XO)
		ADV_CMD += --advanced.param compiler.userPostSysLinkOverlayTcl=$(shell pwd)/Vitis_with_100Gbps_TCP-IP/scripts/post_sys_link.tcl
	endif
	OTHER_XO += $(LOOPBACK_XO) $(TCP_SESS_XO)
	IPREPO_CMD += --user_ip_repo_paths Vitis_with_100Gbps_TCP-IP/build/fpga-network-stack/iprepo
	CCLO_STACK_TYPE = TCP
else ifeq (coyote_tcp, $(MODE))
	N_DDR_CHAN = 0
	ifeq (u200,$(findstring u200, $(PLATFORM)))
		N_DDR_CHAN = 2
	endif
	ifeq (u250,$(findstring u250, $(PLATFORM)))
		N_DDR_CHAN = 2
	endif
	OTHER_XO = coyote_shell
	COYOTE_CONFIG = -DFDEV_NAME=$(BOARD) -DEN_MEM=1 -DEN_STRM=1 -DEN_BPSS=1 -DEN_TCP_0=1 -DN_STRM_AXI=2 -DN_CARD_AXI=2 -DEN_HLS=0 -DACLK_F=250 -DN_DDR_CHAN=$(N_DDR_CHAN)
	CCLO_STACK_TYPE = TCP
	OUTPUT_PRODUCT = $(CYT_BIT)
	USE_HOSTMEM = 1
else ifeq (coyote_rdma, $(MODE))
	N_DDR_CHAN = 0
	ifeq (u200,$(findstring u200, $(PLATFORM)))
		N_DDR_CHAN = 2
	endif
	ifeq (u250,$(findstring u250, $(PLATFORM)))
		N_DDR_CHAN = 2
	endif
	OTHER_XO = coyote_shell
	COYOTE_CONFIG = -DFDEV_NAME=$(BOARD) -DEN_MEM=1 -DEN_STRM=1 -DEN_BPSS=1 -DEN_RDMA_0=1 -DEN_RPC=1 -DN_STRM_AXI=3 -DN_CARD_AXI=3 -DEN_HLS=0 -DACLK_F=250 -DTLBL_A=12 -DN_DDR_CHAN=$(N_DDR_CHAN)
	CCLO_STACK_TYPE = RDMA
	OUTPUT_PRODUCT = $(CYT_BIT)
	USE_HOSTMEM = 1
else
	$(error Unsupported MODE)
endif

ifeq (1, $(USE_HOSTMEM))
	GEN_CONFIG_FLAGS += --host
	NUM_EXTDMA_AXI = 2
endif

ifneq (none, $(ETH_IF))
	GEN_CONFIG_FLAGS += --ethif $(ETH_IF)
endif

all: $(OUTPUT_PRODUCT)

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
	$(MAKE) -C ../../kernels/plugins/dummy_tcp_stack DEVICE=$(FPGAPART) all

.PHONY: coyote_shell
coyote_shell: Coyote/hw/build_$(CCLO_STACK_TYPE)/lynx/lynx.xpr

Coyote/hw/build_$(CCLO_STACK_TYPE)/lynx/lynx.xpr:
	cd Coyote/hw && mkdir build_$(CCLO_STACK_TYPE) && cd build_$(CCLO_STACK_TYPE) && cmake .. $(COYOTE_CONFIG)
	$(MAKE) -C Coyote/hw/build_$(CCLO_STACK_TYPE)/ shell

$(CYT_BIT): coyote_shell
	$(MAKE) -C ../../kernels/cclo PLATFORM=$(PLATFORM) STACK_TYPE=$(CCLO_STACK_TYPE) MB_DEBUG_LEVEL=$(CCLO_MB_DEBUG_LEVEL) EN_DMA=0 EN_EXT_DMA=1
	$(MAKE) -C ../../kernels/plugins PLATFORM=$(PLATFORM) DEBUG=$(DEBUG) STACK_TYPE=$(CCLO_STACK_TYPE) NUM_EXTDMA_AXI=$(NUM_EXTDMA_AXI)
	vivado -mode tcl -source tcl/coyote.tcl -tclargs $(CCLO_STACK_TYPE) Coyote/hw/build_$(CCLO_STACK_TYPE)
	cp hdl/$(MODE)_top.sv Coyote/hw/build_$(CCLO_STACK_TYPE)/lynx/hdl/config_0/user_logic_c0_0.sv
	$(MAKE) -C Coyote/hw/build_$(CCLO_STACK_TYPE)/ compile

.PHONY: tcp_stack
tcp_stack: $(TCP_XO) $(CMAC_TCP_XO)

$(TCP_XO):
	git submodule update --init --recursive Vitis_with_100Gbps_TCP-IP
	mkdir -p Vitis_with_100Gbps_TCP-IP/build && \
	cd Vitis_with_100Gbps_TCP-IP/build && \
	cmake ../ -DFDEV_NAME=$(BOARD) -DVIVADO_HLS_ROOT_DIR=$(TCP_HLS_ROOTDIR) -DVIVADO_ROOT_DIR=$(TCP_VIVADO_ROOTDIR) -DTCP_STACK_EN=1 -DTCP_STACK_RX_DDR_BYPASS_EN=$(TCP_RX_BYPASS) -DDTCP_STACK_WINDOW_SCALING_EN=0
	$(MAKE) -C Vitis_with_100Gbps_TCP-IP/build installip
	$(MAKE) -C Vitis_with_100Gbps_TCP-IP/ network_krnl DEVICE=$(PLATFORM) TEMP_DIR=_x.hw.$(XSA)/ XSA=$(XSA)

$(CCLO_XO):
	$(MAKE) -C ../../kernels/cclo PLATFORM=$(PLATFORM) STACK_TYPE=$(CCLO_STACK_TYPE) MB_DEBUG_LEVEL=$(CCLO_MB_DEBUG_LEVEL)

$(XCLBIN): $(NET_XO) $(CCLO_XO)
	python3 gen_config.py --board $(BOARD) --poe $(MODE) $(GEN_CONFIG_FLAGS) -o $(CONFIGFILE)
	$(MAKE) -C ../../kernels/plugins PLATFORM=$(PLATFORM) DEBUG=$(DEBUG) STACK_TYPE=$(CCLO_STACK_TYPE) NUM_EXTDMA_AXI=$(NUM_EXTDMA_AXI)
	v++ --link --platform $(PLATFORM) --kernel_frequency $(FREQUENCY) --save-temps --temp_dir $(BUILD_DIR) $(VPP_CONFIG) $(ADV_CMD) $(IPREPO_CMD) -o $@ $(NET_XO) $(OTHER_XO)

.PHONY: distclean
distclean:
	git clean -xfd
