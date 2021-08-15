
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

IPS = loopback reduce_sum fp_hp_stream_conv hp_fp_stream_conv cclo 
DEVICE=xcu250-figd2104-2L-e
TARGET=ip

all: $(IPS)

.PHONY: loopback cclo reduce_sum fp_hp_stream_conv hp_fp_stream_conv
$(IPS):
	$(MAKE) -C $@ DEVICE=$(DEVICE) TARGET=$(TARGET)
