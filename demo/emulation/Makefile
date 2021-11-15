
ACCL_REPO_ROOT=$(shell pwd)/../../
HLSLIB_INCLUDE=$(ACCL_REPO_ROOT)/hlslib/include/hlslib/xilinx/
REDUCTION_DIR=$(ACCL_REPO_ROOT)/kernels/reduce_sum
CCLO_VNX_DIR=$(ACCL_REPO_ROOT)/kernels/cclo/hls/vnx_intf
MB_FW_DIR=$(ACCL_REPO_ROOT)/kernels/cclo/fw/sw_apps/ccl_offload_control/src

all: cclo_emu

cclo_emu: cclo_emu.cpp $(MB_FW_DIR)/ccl_offload_control.c
	g++ -fdiagnostics-color=always -g -DZMQ_CALL_VERBOSE -DACCL_BD_SIM -I$(HLSLIB_INCLUDE) -I$(XILINX_VIVADO)/include/ -I$(REDUCTION_DIR) -I$(CCLO_VNX_DIR) -I$(MB_FW_DIR) cclo_emu.cpp $(MB_FW_DIR)/ccl_offload_control.c $(REDUCTION_DIR)/reduce_sum.cpp $(CCLO_VNX_DIR)/vnx_depacketizer.cpp $(CCLO_VNX_DIR)/vnx_packetizer.cpp -o $@ -lpthread -lzmqpp -lzmq -ljsoncpp


