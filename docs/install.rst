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

.. code-block:: sh

    sudo apt update
    sudo apt install python3 cmake libzmqpp-dev libjsoncpp-dev libtclap-dev libopenmpi-dev xvfb

Install the Xilinx Run-Time libraries (XRT)
*******************************************
.. code-block:: sh

    wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_202120.2.12.427_20.04-amd64-xrt.deb
    sudo dpkg -i xrt_202120.2.12.427_20.04-amd64-xrt.deb
    sudo apt --fix-broken install

Build a hardware design with ACCL
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
       | single FPGA, using an AXI-Stream switch. Used for testing (see below)

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
