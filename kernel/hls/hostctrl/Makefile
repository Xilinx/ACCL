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
HOSTCTRL_IP=build_hostctrl/sol1/impl/ip/xilinx_com_hls_hostctrl_1_0.zip

all: $(HOSTCTRL_IP)

$(HOSTCTRL_IP): build_hostctrl.tcl hostctrl.cpp 
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

clean:
	rm -r build_hostctrl vitis_hls.log

