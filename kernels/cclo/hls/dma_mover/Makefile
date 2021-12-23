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

TARGET=ip
DEVICE=xcu280-fsvh2892-2L-e
DMA_MOVER_IP=build_dma_mover/sol1/impl/ip/xilinx_com_hls_dma_mover_1_0.zip

all: $(DMA_MOVER_IP)

$(DMA_MOVER_IP): ../build.tcl dma_mover.cpp 
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) dma_mover

clean:
	rm -r build_dma_mover vitis_hls.log

