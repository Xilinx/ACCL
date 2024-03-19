# /*******************************************************************************
#  Copyright (C) 2024 Advanced Micro Devices, Inc
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
DEVICE=xcu280-fsvh2892-2L-e
DUMMY_DMA_IP=dummy_cyt_dma.xo

TARGET=ip

all: $(DUMMY_DMA_IP)

$(DUMMY_DMA_IP): build_dummy_cyt_dma.tcl dummy_cyt_dma.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

