..
   comment:: SPDX-License-Identifier: Apache-2.0
   comment:: Copyright (C) 2022 Advanced Micro Devices, Inc

.. _troubleshooting_section:

##################################
Troubleshooting
##################################

While we make every effort to test ACCL thoroughly, problems will inevitably occur either 
when developing your application against the ACCL emulator or simulator, or when deploying
the application to hardware. This section provides some tips on debugging at these two stages.

Inspecting the CCLO configuration
***********************************************

Most problems with ACCL are likely to occur because of misconfigurations of the CCLO. There 
are three data structures configured by the host into the CCLO:

* The communicators, containing information about peers (IPs, session numbers, sequence numbers)
* The datapath configuration, which defines how the CCLO uses arithmetic and compression plugins
* Receive (RX) buffers, into which the CCLO places data received from peers

To inspect the configuration of communicator(s), use the `dump_communicator()` ACCL function, which returns a 
string detailing the structure of all defined communicators. The following output is obtained by dumping
the communicators after running the host-side API tests against the emulator:

.. code-block::

    Communicator 0 (0x204):
    local rank: 0    number of ranks: 4
    > rank 0 (ip 127.0.0.1:5500 ; session 0 ; max segment size 1024) : <- inbound seq number 0, -> outbound seq number 0
    > rank 1 (ip 127.0.0.1:5501 ; session 0 ; max segment size 1024) : <- inbound seq number 8, -> outbound seq number 49
    > rank 2 (ip 127.0.0.1:5502 ; session 1 ; max segment size 1024) : <- inbound seq number 4, -> outbound seq number 4
    > rank 3 (ip 127.0.0.1:5503 ; session 2 ; max segment size 1024) : <- inbound seq number 49, -> outbound seq number 8
    Communicator 1 (0x344):
    local rank: 0    number of ranks: 3
    > rank 0 (ip 127.0.0.1:5500 ; session 0 ; max segment size 1024) : <- inbound seq number 0, -> outbound seq number 0
    > rank 1 (ip 127.0.0.1:5502 ; session 1 ; max segment size 1024) : <- inbound seq number 1, -> outbound seq number 5
    > rank 2 (ip 127.0.0.1:5503 ; session 2 ; max segment size 1024) : <- inbound seq number 4, -> outbound seq number 0
    Communicator 2 (0x394):
    local rank: 0    number of ranks: 2
    > rank 0 (ip 127.0.0.1:5500 ; session 0 ; max segment size 1024) : <- inbound seq number 0, -> outbound seq number 0
    > rank 1 (ip 127.0.0.1:5501 ; session 0 ; max segment size 1024) : <- inbound seq number 1, -> outbound seq number 1

The output indicates the address in CCLO configuration memory where the communicator is defined, and for each rank in the communicator,
the IP and port, session number where relevant, maximum amount of data we can transfer to the respective rank in a single message,
and sequence numbers for inbound and outbound messages from/to the rank. The respective numbers are incremented
every time a message is exchanged with the rank.

To inspect the RX buffers, use the `dump_rx_buffers()` ACCL function. This returns buffer metadata and optionally contents for
all of the configured RX buffers. By default there are 16 such buffers:

.. code-block::

    CCLO address: 0
    Spare RX Buffer 0:  address: 0x0    status: ENQUEUED occupancy: 32/1024 MPI tag: ffffffff seq: 46 src: 1
    Spare RX Buffer 1:  address: 0x1000 status: ENQUEUED occupancy: 64/1024 MPI tag: ffffffff seq: 47 src: 1
    Spare RX Buffer 2:  address: 0x2000 status: ENQUEUED occupancy: 32/1024 MPI tag: ffffffff seq: 48 src: 1
    Spare RX Buffer 3:  address: 0x3000 status: ENQUEUED occupancy: 64/1024 MPI tag: ffffffff seq: 37 src: 1
    Spare RX Buffer 4:  address: 0x4000 status: ENQUEUED occupancy: 32/1024 MPI tag: ffffffff seq: 38 src: 1
    Spare RX Buffer 5:  address: 0x5000 status: ENQUEUED occupancy: 32/1024 MPI tag: ffffffff seq: 39 src: 1
    Spare RX Buffer 6:  address: 0x6000 status: ENQUEUED occupancy: 32/1024 MPI tag: ffffffff seq: 40 src: 1
    Spare RX Buffer 7:  address: 0x7000 status: ENQUEUED occupancy: 64/1024 MPI tag: ffffffff seq: 41 src: 1
    Spare RX Buffer 8:  address: 0x8000 status: ENQUEUED occupancy: 32/1024 MPI tag: ffffffff seq: 42 src: 1
    Spare RX Buffer 9:  address: 0x9000 status: ENQUEUED occupancy: 64/1024 MPI tag: ffffffff seq: 4  src: 3
    Spare RX Buffer 10: address: 0xa000 status: ENQUEUED occupancy: 32/1024 MPI tag: ffffffff seq: 5  src: 3
    Spare RX Buffer 11: address: 0xb000 status: ENQUEUED occupancy: 64/1024 MPI tag: ffffffff seq: 6  src: 3
    Spare RX Buffer 12: address: 0xc000 status: ENQUEUED occupancy: 64/1024 MPI tag: ffffffff seq: 7  src: 3
    Spare RX Buffer 13: address: 0xd000 status: ENQUEUED occupancy: 64/1024 MPI tag: ffffffff seq: 43 src: 1
    Spare RX Buffer 14: address: 0xe000 status: ENQUEUED occupancy: 64/1024 MPI tag: ffffffff seq: 44 src: 1
    Spare RX Buffer 15: address: 0xf000 status: ENQUEUED occupancy: 32/1024 MPI tag: ffffffff seq: 45 src: 1

Each buffer has an index, a starting address in FPGA memory, and metadata for the last message it stored (source, sequence number, tag, occupancy).
A `ENQUEUED` status indicates the buffer is ready to receive a message, while `RESERVED` indicates it is already
storing a message.

Debugging Emulator/Simulator based Applications
***********************************************

Because the CCLO is mostly described in C++ for high level synthesis, its internals are
visible to the debugger when running an application against the ACCL emulator. As such, developers
can get access to almost all of the execution aspects, including control algorithms and data movement inside the CCLO.
To do so, first start the emulator:

.. code-block:: sh

    $ python3 run.py -n 4
    Building executable...
    [100%] Built target cclo_emu
    Starting emulator...
    mpirun -np 4 --tag-output /home/lpetrica/git/ACCL_main/test/model/emulator/cclo_emu tcp 5500 loopback
    [1,1]<stdout>:[INFO 11:54:52] Rank 1 binding to tcp://127.0.0.1:5501 (CMD) and tcp://127.0.0.1:5505 (ETH)
    [1,2]<stdout>:[INFO 11:54:52] Rank 2 binding to tcp://127.0.0.1:5502 (CMD) and tcp://127.0.0.1:5506 (ETH)
    [1,3]<stdout>:[INFO 11:54:52] Rank 3 binding to tcp://127.0.0.1:5503 (CMD) and tcp://127.0.0.1:5507 (ETH)
    [1,0]<stdout>:[INFO 11:54:52] Rank 0 binding to tcp://127.0.0.1:5500 (CMD) and tcp://127.0.0.1:5504 (ETH)
    [1,1]<stdout>:[INFO 11:54:53] Rank 1 connecting to tcp://127.0.0.1:5504 (ETH)
    ...

Each CCLO is emulated by an instance of the `cclo_emu` executable, four in this example. Find the PIDs of the instances with:

.. code-block:: sh

    $ pgrep cclo_emu
    3262
    3263
    3264
    3265

The PIDs are listed in ascending rank order, so 3262 corresponds to rank 0 and so on. Pick the rank you'd like to debug, 
and start a GDB server attaching to the corresponding PID:

.. code-block:: sh

    gdbserver --attach :<PORT> <PID>

The GDB server attaches to the PID and listens to the specified port. We can now connect to the server, halt the emulated CCLO,
step through code and so on. The following VSCode configuration can be used to connect to the GDB server from the code editor
and step through lines of code: 

.. code-block:: json

    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Attach to gdbserver",
                "type": "gdb",
                "request": "attach",
                "executable": "${fileDirname}/${fileBasenameNoExtension}",
                "target": "localhost:<PORT>",
                "remote": true,
                "cwd": "${fileDirname}", 
                "gdbpath": "/usr/bin/gdb",
                "autorun": [ "interrupt" ]
            }
        ]
    }

Debugging hardware ACCL designs
***********************************************

The key to successful debugging of a hardware ACCL design is to have visibility into the CCLO interfaces
and control over the CCLO internal microcontroller. Once this is achieved, the design can be inspected 
during operation and the CCLO firmware can executed in stepping mode if required, or even updated with 
necessary fixes. Here are some steps to follow:

Enabling CCLO and Chipscope debug
##################################


Programming and checking the ACCL design on Alveo
######################################################

Before performing any action on the Alveo, make sure it is functional. `xbutil` is the Alveo platform management utility. 
To check the cards available on your system, run `xbutil examine`, 
which lists Alveo cards and prints XRT and system information. Here is some example output:

.. code-block:: sh

    xbutil examine
    Devices present
    BDF             :  Shell                      Platform UUID  Device ID
    [0000:81:00.1]  :  xilinx_u280_xdma_201920_3  0x5e278820     user(inst=129)
    [0000:21:00.1]  :  xilinx_u250_xdma_201830_2  0x5d14fbe6     user(inst=128)

At least one card should have a shell matching the target shell of your design. 
Notice each card has a unique BDF string, e.g. `0000:81:00.1` for the Alveo U280 in this example. 
Make a note of this string as it is used to identify the card in all other commands.
Before programming the ACCL design, you may want to check the integrity of the target Alveo card with `xbutil validate -d <BDF>`. 
This programs the board with a test design and performs diagnostic tests.

Finally we can program our ACCL design, using the Vitis-generated XCLBIN. 
Run `xbutil program -d <BDF> -u <XCLBIN file>` and wait for completion. 
This should take a few seconds as the FPGA is being programmed.
To check the design is correctly programmed and visible to XRT, we can run `xbutil examine -d <BDF>` - notice 
we're now examining only our target board, and we'll get more detail of kernels and memories present in the design.

Enable Debugging Access to the Board
######################################################

To access the design for any form of debugging we require a virtual JTAG cable and debug server connected to it. 
Both of these can be started with one command:

.. code-block:: sh

    debug_hw --xvc_pcie /dev/xfpga/xvc_pub.u<BDF number> --hw_server --hw_server_port 3121 --xvc_pcie_port 10200

For a BDF of the form `0000:B:D.F` the BDF number is `B*256 + D*8 + F`. Port numbers shown are default values and can be changed. This command will start a debug server with Xilinx Virtual Cable attached to the target board.

Access and Update CCLO Firmware
######################################################

You can inspect Microblaze targets inside the design with `xsct` either from the machine hosting the Alveos or another one:

.. code-block:: sh

    xsct -nopdisp
    xsct% connect -xvc <host>:<port>
    xsct% targets

`host` and `port` are the hostname where the debug server was started, and the port of the XVC, typically 10200. 
Before exiting XSCT make a note of the identified Microblaze targets visible through the debug hub. 
You should see at least one Microblaze if your design includes the CCLO kernel.
To program a new firmware ELF file into the CCLO(s), run

.. code-block:: sh

    xsct -nodisp update_elf.tcl <host> <port> <ELF file> <target(s)>

Notice that multiple targets can be programmed with the provided ELF, e.g. if there are multiple CCLOs in a design.
You can now connect to the Microblaze target from a remote Vitis GUI for code debugging, or use the local
XSCT console to step through the code, set breakpoints, etc.