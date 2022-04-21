# Installation instructions

## Pull project
```sh
git clone https://github.com/Xilinx/ACCL.git
git submodule update --init --recursive
```

## Install dependencies
The project has been tested with Xilinx Vitis 2021.2 on Ubuntu 20.04.
```sh
sudo apt update
sudo apt install python3 cmake libzmqpp-dev libjsoncpp-dev libtclap-dev libopenmpi-dev xvfb
pip3 install numpy pynq zmq mpi4py
```
Optionally, install the Xiling Run-Time libraries (XRT)
```
wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_202120.2.12.427_20.04-amd64-xrt.deb
sudo dpkg -i xrt_202120.2.12.427_20.04-amd64-xrt.deb
sudo apt --fix-broken install
```

## Run ACCL tests using the Emulator or Simulator
First start up either the emulator or simulator.
<details>
  <summary>Emulator</summary>

  ```sh
  cd "test/model/emulation"
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
mpirun -np <RANKS> bin/test
```

## Build a hardware design with ACCL
<details>
  <summary>ACCL with TCP Backend</summary>

  ```sh
  source <VITIS_INSTALL>/settings64.sh
  cd "kernels/cclo"
  make STACK_TYPE=TCP EN_FANIN=1 MB_DEBUG_LEVEL=2 PLATFORM=<Alveo Platform Name>
  cd "../../test/hardware"
  make MODE=tcp PLATFORM=<Alveo Platform Name>
  ```
</details>
<details>
  <summary>ACCL with TCP backend</summary>

  ```sh
  source <VITIS_INSTALL>/settings64.sh
  cd "kernels/cclo"
  make STACK_TYPE=UDP MB_DEBUG_LEVEL=2 PLATFORM=<Alveo Platform Name>
  cd "../../test/hardware"
  make MODE=udp PLATFORM=<Alveo Platform Name>
  ```
</details>
<details>
  <summary>3-Rank ACCL Test System on a single FPGA (U280/U250)</summary>

  ```sh
  source <VITIS_INSTALL>/settings64.sh
  cd "kernels/cclo"
  make STACK_TYPE=TCP EN_FANIN=1 MB_DEBUG_LEVEL=2 PLATFORM=<Alveo U280 or U250 Platform Name>
  cd "../../test/hardware"
  make MODE=tri PLATFORM=<Alveo U280 or U250 Platform Name>
  ```
</details>

The following Alveo platforms are supported:
| Alveo | Development Target Platform(s)         |
|-------|----------------------------------------|
| U55C  | xilinx_u55c_gen3x16_xdma_2_202110_1    |
| U250  | xilinx_u250_gen3x16_xdma_3_1_202020_1  |
| U280  | xilinx_u280_xdma_201920_3              |