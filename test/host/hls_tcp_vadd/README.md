# ACCL-distributed Vector Addition over TCP/IP

This directory contains a minimal working example of a FPGA-driven distributed application. Each FPGA (node) generates a unique vector, of variable size, based on it's ID. Then, one is added to each element of the vector and it is transmitted to to the next node. Therefore, each node:
1. Performs some floating-point computation
2. Sends a vector to a neighbouring node 
3. Receives a vector from a neighbouring node

## Running in simulation
The following steps describe how to run the example in emulation/simulation. For more information on ACCL simulation/emulation, please see [here]( https://ethz.ch/content/dam/ethz/special-interest/infk/inst-cp/inst-cp-dam/research/data-processing-on-modern-hardware/ACCL_Sim_FPGA23_Tutorial.pdf). First launch the emulator/simulator, as described in the [INSTALL.md](https://github.com/Xilinx/ACCL/blob/main/INSTALL.md):
```bash
cd "<ACCL_BASE_FOLDER>/test/model/emulator"
source <VITIS_INSTALL>/settings64.sh
/bin/cmake .
python3 run.py -n <RANKS>
```

Then, compile the program and run it using mpirun:
```bash
mkdir build && cd build
/bin/cmake .. && make
cd ..
mpirun -n 2 build/bin/test
```

## Running in hardware
The following example describe how to run the example on hardware. The design is deployed in [ETH HACC](https://systems.ethz.ch/research/data-processing-on-modern-hardware/hacc.html) on Alveo U55C boards; however, the scripts can be modified to run on any compatible FPGA cluster. First, create a bitstream (which will take some time...)
 ```bash
source <VITIS_INSTALL>/settings64.sh
cd "<ACCL_BASE_FOLDER>/test/refdesigns"
make MODE=tcp USER_KERNEL=vadd PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1
```
Once complete, compile the source host code:
```bash
mkdir build && cd build
/bin/cmake .. && make
cd ..
```
Finally, launch the application using the run script. If needed, modify the script to the target cluster set-up
```bash
bash run_ethz_hacc_alveo_u55c.sh
```
