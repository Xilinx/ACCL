
ACCL_REPO_ROOT=$(shell pwd)/../../
HLSLIB_INCLUDE=$(ACCL_REPO_ROOT)/hlslib/include/hlslib/xilinx/
CCLO_HLS_ROOT=$(ACCL_REPO_ROOT)/kernels/cclo/hls
REDUCTION_DIR=$(ACCL_REPO_ROOT)/kernels/reduce_sum
CCLO_ETH_DIR=$(CCLO_HLS_ROOT)/eth_intf
SEGMENTER_DIR=$(CCLO_HLS_ROOT)/segmenter
RXBUF_OFFLOAD_DIR=$(CCLO_HLS_ROOT)/rxbuf_offload
DMA_MOVER_DIR=$(CCLO_HLS_ROOT)/dma_mover
MB_FW_DIR=$(ACCL_REPO_ROOT)/kernels/cclo/fw/sw_apps/ccl_offload_control/src
NRANKS=2
START_PORT=5500

INCLUDES=-I$(HLSLIB_INCLUDE) -I$(XILINX_VIVADO)/include/ -I$(REDUCTION_DIR) -I$(CCLO_ETH_DIR) -I$(SEGMENTER_DIR) -I$(MB_FW_DIR) -I$(CCLO_HLS_ROOT) -I$(DMA_MOVER_DIR) -I$(RXBUF_OFFLOAD_DIR)
SOURCES=cclo_emu.cpp $(MB_FW_DIR)/ccl_offload_control.c $(REDUCTION_DIR)/reduce_sum.cpp $(CCLO_ETH_DIR)/*.cpp $(SEGMENTER_DIR)/stream_segmenter.cpp $(RXBUF_OFFLOAD_DIR)/rxbuf_enqueue.cpp $(RXBUF_OFFLOAD_DIR)/rxbuf_dequeue.cpp $(RXBUF_OFFLOAD_DIR)/rxbuf_seek.cpp $(DMA_MOVER_DIR)/dma_mover.cpp

all: cclo_emu

cclo_emu: cclo_emu.cpp $(MB_FW_DIR)/ccl_offload_control.c
	g++ -std=c++17 -Wno-attributes -fdiagnostics-color=always -g -DZMQ_CALL_VERBOSE -DACCL_BD_SIM $(INCLUDES) $(SOURCES) -o $@ -lpthread -lzmqpp -lzmq -ljsoncpp

.PHONY: run
run: cclo_emu
	./start_ranks.sh $(NRANKS) $(START_PORT)

