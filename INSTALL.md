# Installation instructions

## Pull project
```sh
git clone https://github.com/Xilinx/ACCL.git
git submodule update --init --recursive
```

## Install dependencies
The project has been tested with Xilinx Vitis 2022.1 on Ubuntu 20.04.
```sh
sudo apt update
sudo apt install python3 cmake libzmqpp-dev libjsoncpp-dev libtclap-dev libopenmpi-dev xvfb
```
Install the Xilinx Run-Time libraries (XRT)
```
wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_202120.2.12.427_20.04-amd64-xrt.deb
sudo dpkg -i xrt_202120.2.12.427_20.04-amd64-xrt.deb
sudo apt --fix-broken install
```

## Build a hardware design with ACCL

```sh
source <VITIS_INSTALL>/settings64.sh
cd "test/refdesigns"
make MODE=<Build Mode> PLATFORM=<Platform Name>
```

The following build modes are supported:
| Build Mode | Description                              |
|------------|------------------------------------------|
| udp or tcp | One ACCL instance per FPGA, with UDP or TCP transport respectively, via Ethernet port 0 on XRT shell |
| coyote_tcp | One ACCL instance deployed to a Coyote shell over TCP |
| coyote_rdma | One ACCL instance deployed to a Coyote shell over RDMA, with rendezvous mode support |
| axis3x     | Three ACCL instances connected together internally on a single FPGA, using an AXI-Stream switch. Used for testing (see below) |

The following platforms are supported for Alveo boards:
| Alveo | Platform Name                          |
|-------|----------------------------------------|
| U50   | xilinx_u50_gen3x16_xdma_5_202210_1     |
| U55C  | xilinx_u55c_gen3x16_xdma_3_202210_1    |
| U200  | xilinx_u200_gen3x16_xdma_2_202110_1    |
| U250  | xilinx_u250_gen3x16_xdma_4_1_202210_1  |
| U280  | xilinx_u280_gen3x16_xdma_1_202211_1    |

## Run ACCL tests
### Emulation or simulation tests of host-launched collectives
First start up either the emulator or simulator:
<details>
  <summary>Emulator</summary>

  ```sh
  cd "test/model/emulator"
  source <VITIS_INSTALL>/settings64.sh
  /bin/cmake .
  python3 run.py -n <RANKS>
  ```
</details>

<details>
  <summary>Simulator</summary>

  ```sh
  cd "kernels/cclo"
  source <VIVADO_INSTALL>/settings64.sh
  make STACK_TYPE=TCP EN_FANIN=1 simdll
  cd "../../test/model/simulator"
  source <VITIS_INSTALL>/settings64.sh
  /bin/cmake .
  python3 run.py -n <RANKS>
  ```
</details>

Now open a new terminal and run the tests:
```sh
cd "test/host/xrt"
/bin/cmake .
make
XCL_EMULATION_MODE=sw_emu mpirun -np <RANKS> bin/test
```

### Hardware tests of host-launched collectives
Make sure you have a 3-Rank, single FPGA ACCL test design by either [building](#build-a-hardware-design-with-accl) or [downloading](https://github.com/Xilinx/ACCL/releases/tag/axis3x) it, then open a terminal and run the tests:
```sh
cd "test/host/xrt"
/bin/cmake .
make
mpirun -np <RANKS> bin/test -a -f -x <XCLBIN FILE>
```

### Emulation or simulation tests of PL-launched collectives
We provide an example of a simple vector add kernel issuing commands and data directly to the ACCL offload kernel (CCLO), with no host intervention.

Launch the emulator or simulator with the `--no-kernel-loopback` option, which will enable PL kernels to exchange data with the CCLO via AXI Streams. Open a new terminal and run the tests:
```sh
cd "test/host/hls"
/bin/cmake .
make
mpirun -np <RANKS> bin/test
```
