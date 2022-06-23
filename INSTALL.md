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
pip3 install numpy pynq zmq mpi4py
```
Install the Xilinx Run-Time libraries (XRT)
```
wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_202120.2.12.427_20.04-amd64-xrt.deb
sudo dpkg -i xrt_202120.2.12.427_20.04-amd64-xrt.deb
sudo apt --fix-broken install
```

## Run ACCL tests
### Emulation or simulation
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
  cd "../../test/model/simulation"
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

### Hardware
Make sure you have a [hardware design](#build-a-hardware-design-with-accl) of
ACCL first.

Open a terminal and run the tests:
```sh
cd "test/host/xrt"
/bin/cmake .
make
mpirun -np <RANKS> bin/test -f -x <XCLBIN FILE>
```

## Build a hardware design with ACCL

```sh
source <VITIS_INSTALL>/settings64.sh
cd "test/hardware"
make MODE=<Build Mode> PLATFORM=<Platform Name>
```

The following build modes are supported:
| Build Mode | Description                              |
|------------|------------------------------------------|
| udp        | ACCL with UDP backend                    |
| tcp        | ACCL with TCP backend                    |
| tri        | 3-Rank ACCL Test System on a single FPGA |

The following platforms are supported for Alveo boards:
| Alveo | Platform Name                          |
|-------|----------------------------------------|
| U55C  | xilinx_u55c_gen3x16_xdma_3_202210_1    |
| U250  | xilinx_u250_gen3x16_xdma_3_1_202020_1  |
| U280  | xilinx_u280_xdma_201920_3              |
