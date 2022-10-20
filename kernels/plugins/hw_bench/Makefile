.PHONY: help

help:
	@echo "Makefile Usage:"
	@echo "  make all DEVICE=<FPGA platform>"
	@echo "      Command to generate the xo for specified device."
	@echo "      By default, DEVICE=xilinx_u280_xdma_201920_3"
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated non-hardware files."
	@echo ""
	@echo "  make distclean"
	@echo "      Command to remove all the generated files."
	@echo ""


DEVICE ?= xilinx_u280_xdma_201920_3
KRNL_NAME_HDL := hw_bench
TARGET=ip

XSA := $(strip $(patsubst %.xpfm, % , $(shell basename $(DEVICE))))
TEMP_DIR := ./
VIVADO := $(XILINX_VIVADO)/bin/vivado
CLFLAGS += -t hw --platform $(DEVICE) --save-temps

BINARY_HLD_OBJS = $(TEMP_DIR)/${KRNL_NAME_HDL}.xo

.PHONY: all clean cleanall 
all: check-devices check-vivado $(BINARY_HLD_OBJS)


# Cleaning stuff
clean:
	rm -rf *v++* *.log *.jou *.xo

distclean: clean
	rm -rf build_dir*
	rm -rf ./tmp_$(KRNL_NAME_HDL)* ./packaged_kernel* *.xo
	rm -rf _x* *.str
	rm -rf .Xil


$(TEMP_DIR)/${KRNL_NAME_HDL}.xo:
	mkdir -p $(TEMP_DIR)
	$(VIVADO) -mode batch -source package_hw_bench.tcl -notrace -tclargs $@ ${KRNL_NAME_HDL} $(XSA)


check-devices:
ifndef DEVICE
	$(error DEVICE not set. Please set the DEVICE properly and rerun. Run "make help" for more details.)
endif

#Checks for XILINX_VIVADO
check-vivado:
ifndef XILINX_VIVADO
	$(error XILINX_VIVADO variable is not set, please set correctly and rerun)
endif

