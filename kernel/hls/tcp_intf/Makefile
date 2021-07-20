
DEVICE=xcu280-fsvh2892-2L-e
TCP_OPENPORT_IP=build_tcp_openPort/sol1/impl/ip/xilinx_com_hls_tcp_openPort_1_0.zip
TCP_OPENCONREQ_IP=build_tcp_openConReq/sol1/impl/ip/xilinx_com_hls_tcp_openConReq_1_0.zip
TCP_OPENCONRESP_IP=build_tcp_openConResp/sol1/impl/ip/xilinx_com_hls_tcp_openConResp_1_0.zip
TCP_PACKETIZER_IP=build_tcp_packetizer/sol1/impl/ip/xilinx_com_hls_tcp_packetizer_1_0.zip
TCP_DEPACKETIZER_IP=build_tcp_depacketizer/sol1/impl/ip/xilinx_com_hls_tcp_depacketizer_1_0.zip
TCP_TXHANDLER_IP=build_tcp_txHandler//sol1/impl/ip/xilinx_com_hls_tcp_txHandler_1_0.zip
TCP_RXHANDLER_IP=build_tcp_rxHandler//sol1/impl/ip/xilinx_com_hls_tcp_rxHandler_1_0.zip


TARGET=ip

all: $(TCP_OPENPORT_IP) $(TCP_OPENCONREQ_IP) $(TCP_OPENCONRESP_IP) $(TCP_PACKETIZER_IP) $(TCP_TXHANDLER_IP) $(TCP_RXHANDLER_IP) $(TCP_DEPACKETIZER_IP)

open_port: $(TCP_OPENPORT_IP)

open_con_resp: $(TCP_OPENCONRESP_IP)

packetizer: $(TCP_PACKETIZER_IP)

depacketizer: $(TCP_DEPACKETIZER_IP)

txHandler: $(TCP_TXHANDLER_IP)

rxHandler: $(TCP_RXHANDLER_IP)


$(TCP_OPENPORT_IP): build_tcp_openPort.tcl tcp_openPort.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(TCP_OPENCONREQ_IP): build_tcp_openConReq.tcl tcp_openConReq.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(TCP_OPENCONRESP_IP): build_tcp_openConResp.tcl tcp_openConResp.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(TCP_PACKETIZER_IP): build_tcp_packetizer.tcl tcp_packetizer.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(TCP_DEPACKETIZER_IP): build_tcp_depacketizer.tcl tcp_depacketizer.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(TCP_TXHANDLER_IP): build_tcp_txHandler.tcl tcp_txHandler.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

$(TCP_RXHANDLER_IP): build_tcp_rxHandler.tcl tcp_rxHandler.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)