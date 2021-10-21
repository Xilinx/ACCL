# ACCL: Accelerated Collective Communication Library

### * Note: This project is under active development. We will tag a stable release in the coming weeks.*

ACCL is a Vitis kernel and associated Pynq and XRT drivers which together provide MPI-like collectives for Xilinx FPGAs. ACCL is designed to enable compute kernels resident in FPGA fabric to communicate directly under host supervision but without requiring data movement between the FPGA and host. Instead, ACCL uses Vitis-compatible TCP and UDP stacks to connect FPGAs directly over Ethernet at up to 100 Gbps on Alveo cards. 

ACCL currently supports Send/Recv and the following collectives:
* Broadcast
* Scatter
* Gather
* All-gather
* Reduce
* All-reduce

## Repository Structure

The repository is organized as follows:
- [kernel](kernel/): builds the ACCL Vitis kernel (called CCL Offload)
   - [hls](kernel/hls/): FPGA IPs generated from C++ code
   - [fw](kernel/fw/): firmware running on the embedded MicroBlaze
   - [tcl](kernel/tcl/): TCL build automation
- [driver](driver/): drivers for the ACCL.
   - [pynq](host/pynq/): a PYNQ based python driver
   - [xrt](host/xrt/): an XRT based C++ driver
- [demo](demo/): ACCL example systems on Alveo.
   - [build](demo/build/): build bitstreams for Alveo cards
     - [config](demo/build/config/): board-specific linker configuration files
     - [hls](demo/build/hls/): auxiliary modules written in HLS
     - [tcl](demo/build/tcl/): various TCL automation
     - submodules for supported network stacks
   - [host](demo/host/): host code which drives the demo
     - [debug](demo/host/debug/): host-side debug scripts for XSCT

## Build a demo with TCP stack. 

More info at [demo/build/readme.md](demo/build/readme.md)

1. Source xrt
1. Go in [demo/build](demo/build)
   1. Modify [demo/build/Makefile](demo/build/Makefile) to target the right shell (e.g. xilinx_u280_xdma_201920_3) and the right mode (e.g. tcp_cmac)
   1. Source Vitis 2020.1
   1. Build tcp network stack ips
   1. Source Vitis 2020.2
   1. run make to create a ``.xclbin``
