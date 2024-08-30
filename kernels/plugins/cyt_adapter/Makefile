# /*******************************************************************************
#  Copyright (C) 2021\3 Advanced Micro Devices, Inc
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

TARGET=ip
DEVICE=xcu55c-fsvh2892-2L-e
CYT_DMA_ADAPTER=cyt_dma_adapter_$(DEVICE).xo
CYT_RDMA_ARBITER=cyt_rdma_arbiter_$(DEVICE).xo
CYT_RDMA_MUX=cyt_rdma_mux_$(DEVICE).xo
CCLO_SQ_ADAPTER=cclo_sq_adapter_$(DEVICE).xo

all: $(CYT_DMA_ADAPTER) $(CYT_RDMA_ARBITER) $(CCLO_SQ_ADAPTER)

$(CYT_DMA_ADAPTER): build_cyt_dma_adapter.tcl cyt_dma_adapter.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(CYT_RDMA_ARBITER): build_cyt_rdma_arbiter.tcl cyt_rdma_arbiter.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(CYT_RDMA_MUX): build_cyt_rdma_mux.tcl cyt_rdma_mux.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(CCLO_SQ_ADAPTER): build_cclo_sq_adapter.tcl cclo_sq_adapter.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

