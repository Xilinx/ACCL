
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

PERIPHERAL_IPS = hostctrl loopback reduce_ops hp_compression dummy_tcp_stack client_arbiter vadd_put cyt_adapter external_dma dummy_cyt_rdma_stack dummy_cyt_dma tcp_session_handler
DEVICE=xcu280-fsvh2892-2L-e
TARGET=ip

all: $(PERIPHERAL_IPS)

.PHONY: hostctrl loopback reduce_ops hp_compression dummy_tcp_stack client_arbiter vadd_put cyt_adapter external_dma dummy_cyt_rdma_stack dummy_cyt_dma tcp_session_handler

$(PERIPHERAL_IPS):
	$(MAKE) -C $@ DEVICE=$(DEVICE) TARGET=$(TARGET)
