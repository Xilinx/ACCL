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
DEVICE=xcu250-figd2104-2L-e
RXBUF_DEQUEUE_IP=build_rxbuf_dequeue/sol1/impl/ip/xilinx_com_hls_rxbuf_dequeue_1_0.zip
RXBUF_ENQUEUE_IP=build_rxbuf_enqueue/sol1/impl/ip/xilinx_com_hls_rxbuf_enqueue_1_0.zip
RXBUF_SEEK_IP=build_rxbuf_seek/sol1/impl/ip/xilinx_com_hls_rxbuf_seek_1_0.zip
RXBUF_SESSION_IP=build_rxbuf_session/sol1/impl/ip/xilinx_com_hls_rxbuf_session_1_0.zip

all: $(RXBUF_DEQUEUE_IP) $(RXBUF_ENQUEUE_IP) $(RXBUF_SEEK_IP) $(RXBUF_SESSION_IP)

$(RXBUF_DEQUEUE_IP): ../build.tcl rxbuf_dequeue.cpp 
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) rxbuf_dequeue

$(RXBUF_ENQUEUE_IP): ../build.tcl rxbuf_enqueue.cpp 
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) rxbuf_enqueue

$(RXBUF_SEEK_IP): ../build.tcl rxbuf_seek.cpp 
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) rxbuf_seek

$(RXBUF_SESSION_IP): ../build.tcl rxbuf_session.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) rxbuf_session
