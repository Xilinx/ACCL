# /*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
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

DEVICE ?= xcu280-fsvh2892-2L-e

all: external_dma_1port.xo external_dma_2port.xo external_dma_3port.xo external_dma_4port.xo

external_dma_%port.xo: bd.tcl
	python3 gen_files.py -n $*
	vivado -mode batch -source $< -notrace -tclargs $(DEVICE) $*
