..
   comment:: SPDX-License-Identifier: Apache-2.0
   comment:: Copyright (C) 2022 Advanced Micro Devices, Inc

##################################
Installing ACCL
##################################

Pull project
************
.. code-block:: sh

    git clone https://github.com/Xilinx/ACCL.git
    git submodule update --init --recursive


Install dependencies
********************
The project has been tested with Xilinx Vitis 2022.1 on Ubuntu 20.04.
Install package dependencies as follows:

.. code-block:: sh

    sudo apt update
    sudo apt install python3 cmake libzmqpp-dev libjsoncpp-dev libtclap-dev libopenmpi-dev xvfb

Install the Xilinx Run-Time libraries (`XRT <https://www.xilinx.com/products/design-tools/vitis/xrt.html>`_) as follows:

.. code-block:: sh

    wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_202120.2.12.427_20.04-amd64-xrt.deb
    sudo dpkg -i xrt_202120.2.12.427_20.04-amd64-xrt.deb
    sudo apt --fix-broken install

Simulator/Emulator API Tests
*********************************

Tests can be executed against the ACCL emulator, simulator, or real hardware on an Alveo FPGA board.
The following sequence of commands builds the emulator:

.. code-block:: sh
  
    cd "test/model/emulator"
    source <VITIS_INSTALL>/settings64.sh
    /bin/cmake .
    python3 run.py -n <RANKS>

The equivalent sequence for the simulator is below. The simulator is a more accurate representation 
of the ACCL FPGA components, but executes slower than the emulator.

.. code-block:: sh

    cd "kernels/cclo"
    source <VIVADO_INSTALL>/settings64.sh
    make STACK_TYPE=TCP EN_FANIN=1 simdll
    cd "../../test/model/simulation"
    source <VITIS_INSTALL>/settings64.sh
    /bin/cmake .
    python3 run.py -n <RANKS>

Now open a new terminal and run the host-side API tests:

.. code-block:: sh

    cd "test/host/xrt"
    /bin/cmake . && make
    XCL_EMULATION_MODE=sw_emu mpirun -np <RANKS> bin/test

We also provide an example of a simple vector add kernel issuing commands and data directly to the ACCL offload kernel (CCLO), with no host intervention.
Launch the emulator or simulator again with the `--no-kernel-loopback` option, which will enable PL kernels to exchange data with the CCLO via AXI Streams. 
Open a new terminal and run the PL control tests:

.. code-block:: sh

    cd "test/host/hls"
    /bin/cmake .
    make
    mpirun -np <RANKS> bin/test

Hardware API Tests
*******************************************

Make sure you have a 3-Rank, single FPGA ACCL test design by `downloading <https://github.com/Xilinx/ACCL/releases/tag/axis3x>`_ the appropriate XCLBIN for your 
Alveo board, then open a terminal and run the tests:

.. code-block:: sh

    cd "test/host/xrt"
    /bin/cmake . && make
    mpirun -np 3 bin/test -a -f -x <XCLBIN FILE>

.. code-block:: sh

    cd "test/host/hls"
    /bin/cmake . && make
    mpirun -np 3 bin/test -f -x <XCLBIN FILE>

Build example FPGA designs
*********************************

.. code-block:: sh

    source <VITIS_INSTALL>/settings64.sh
    cd "test/hardware"
    make MODE=<Build Mode> PLATFORM=<Platform Name>

The following build modes are supported:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Build Mode
     - Description
   * - udp or tcp
     - | One ACCL instance per FPGA, with UDP or TCP transport respectively,
       | via Ethernet port 0
   * - axis3x
     - | Three ACCL instances connected together internally on a
       | single FPGA, using an AXI-Stream switch. Used for testing the host-side API.
   * - axis3x_vadd
     - | Similar to axis3x but includes a test kernel that issues
       | commands and data directly to the CCLO.

The following platforms are supported for Alveo boards:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Alveo
     - Platform Name
   * - U55c
     - xilinx_u55c_gen3x16_xdma_3_202210_1
   * - U250
     - xilinx_u250_gen3x16_xdma_3_1_202020_1
   * - U280
     - xilinx_u280_xdma_201920_3
