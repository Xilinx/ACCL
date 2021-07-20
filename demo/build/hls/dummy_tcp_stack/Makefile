
DEVICE=xcu280-fsvh2892-2L-e
TCP_STACK_IP=build_tcp_stack/sol1/impl/export.xo

TARGET=ip

all: $(TCP_STACK_IP)

$(TCP_STACK_IP): build_tcp_stack.tcl dummy_tcp_stack.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

