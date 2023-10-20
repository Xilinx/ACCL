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

DEVICE=xcu280-fsvh2892-2L-e
TCP_PACKETIZER_IP=build_tcp_packetizer/sol1/impl/ip/xilinx_com_hls_tcp_packetizer_1_0.zip
TCP_DEPACKETIZER_IP=build_tcp_depacketizer/sol1/impl/ip/xilinx_com_hls_tcp_depacketizer_1_0.zip
TCP_TXHANDLER_IP=build_tcp_txHandler//sol1/impl/ip/xilinx_com_hls_tcp_txHandler_1_0.zip
TCP_RXHANDLER_IP=build_tcp_rxHandler//sol1/impl/ip/xilinx_com_hls_tcp_rxHandler_1_0.zip
UDP_PACKETIZER_IP=build_udp_packetizer/sol1/impl/ip/xilinx_com_hls_udp_packetizer_1_0.zip
UDP_DEPACKETIZER_IP=build_udp_depacketizer/sol1/impl/ip/xilinx_com_hls_udp_depacketizer_1_0.zip
RDMA_SQ_HANDLER_IP=build_rdma_sq_handler/sol1/impl/ip/xilinx_com_hls_rdma_sq_handler_1_0.zip
RDMA_PACKETIZER_IP=build_rdma_packetizer/sol1/impl/ip/xilinx_com_hls_rdma_packetizer_1_0.zip
RDMA_DEPACKETIZER_IP=build_rdma_depacketizer/sol1/impl/ip/xilinx_com_hls_rdma_depacketizer_1_0.zip

TARGET=ip

all: $(TCP_PACKETIZER_IP) $(TCP_TXHANDLER_IP) $(TCP_RXHANDLER_IP) $(TCP_DEPACKETIZER_IP) $(UDP_PACKETIZER_IP) $(UDP_DEPACKETIZER_IP) $(RDMA_SQ_HANDLER_IP) $(RDMA_PACKETIZER_IP) $(RDMA_DEPACKETIZER_IP) 

tcp_packetizer: $(TCP_PACKETIZER_IP)
tcp_depacketizer: $(TCP_DEPACKETIZER_IP)
txHandler: $(TCP_TXHANDLER_IP)
rxHandler: $(TCP_RXHANDLER_IP)
udp_depacketizer: $(UDP_DEPACKETIZER_IP)
udp_packetizer: $(UDP_PACKETIZER_IP)
rdma_sq_handler: $(RDMA_SQ_HANDLER_IP)
rdma_packetizer: $(RDMA_PACKETIZER_IP)
rdma_depacketizer: $(RDMA_DEPACKETIZER_IP)

$(TCP_PACKETIZER_IP): ../build.tcl tcp_packetizer.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) tcp_packetizer

$(TCP_DEPACKETIZER_IP): ../build.tcl tcp_depacketizer.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) tcp_depacketizer

$(TCP_TXHANDLER_IP): ../build.tcl tcp_txHandler.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) tcp_txHandler

$(TCP_RXHANDLER_IP): ../build.tcl tcp_rxHandler.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) tcp_rxHandler

$(UDP_PACKETIZER_IP): ../build.tcl udp_packetizer.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) udp_packetizer

$(UDP_DEPACKETIZER_IP): ../build.tcl udp_depacketizer.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) udp_depacketizer

$(RDMA_SQ_HANDLER_IP): ../build.tcl rdma_sq_handler.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) rdma_sq_handler

$(RDMA_PACKETIZER_IP): ../build.tcl rdma_packetizer.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) rdma_packetizer

$(RDMA_DEPACKETIZER_IP): ../build.tcl rdma_depacketizer.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE) rdma_depacketizer
