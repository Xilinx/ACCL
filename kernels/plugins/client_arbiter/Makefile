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

DEVICE=xcu250-figd2104-2L-e
ARBITER_IP=client_arbiter.xo
TARGET=ip
NCLIENTS=2

all: $(ARBITER_IP) 

$(ARBITER_IP): build_client_arbiter.tcl client_arbiter.cpp client_arbiter.h
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) $(NCLIENTS)

