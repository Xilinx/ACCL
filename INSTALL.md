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
sudo apt install python3 cmake libzmqpp-dev libjsoncpp-dev libopenmpi-dev xvfb
pip3 install numpy pynq zmq mpi4py
```

## Run accl tests using emulator or simulator
First start up either the emulator or simulator.
<details>
  <summary>Emulator</summary>

  ```sh
  cd "test/emulation"
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
  cd "../../test/simulation"
  source <VITIS_INSTALL>/settings64.sh
  /bin/cmake .
  python3 run.py -n <RANKS>
  ```
</details>

Now open a new terminal and run the tests:
```sh
cd "test/xrt"
/bin/cmake .
make
mpirun -np <RANKS> bin/test
```
