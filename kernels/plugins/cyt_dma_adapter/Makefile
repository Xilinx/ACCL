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
DEVICE=xcu250-figd2104-2L-e
CYT_DMA_ADAPTER=cyt_dma_adapter.xo

all: $(CYT_DMA_ADAPTER)

$(CYT_DMA_ADAPTER): build_cyt_dma_adapter.tcl cyt_dma_adapter.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

