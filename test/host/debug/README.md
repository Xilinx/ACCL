# Debugging ACCL Hardware
This document describes the steps to program ACCL-enabled design into an Alveo board, access the design with debugging tools, inspect and update CCLO firmware, and debug ACCL hardware with Chipscope.

## Programming an ACCL design into an Alveo
Before performing any action on the Alveo, make sure it is functional. `xbutil` is the Alveo platform management utility. To check the cards available on your system, run:

```
xbutil examine
```
This will list all Alveo cards available as well as XRT and system information. Here is some example output:
```
Devices present
BDF             :  Shell                      Platform UUID  Device ID
[0000:81:00.1]  :  xilinx_u280_xdma_201920_3  0x5e278820     user(inst=129)
[0000:21:00.1]  :  xilinx_u250_xdma_201830_2  0x5d14fbe6     user(inst=128)
```
At least one card should have a shell matching the target shell of your design. Notice each card has a unique BDF string, e.g. `0000:81:00.1` for the Alveo U280 in this example. Make a note of this string as it is used to identify the card in all other commands.

Before programming the ACCL design, you may want to check the integrity of the target Alveo card with `xbutil validate -d <BDF>`. This programs the board with a test design and performs diagnostic tests.

Finally we can program our ACCL design, using the Vitis-generated XCLBIN. Run `xbutil program -d <BDF> -u <XCLBIN file>` and wait for completion. This should take a few seconds as the FPGA is being programmed.

To check the design is correctly programmed and visible to XRT, we can run `xbutil examine -d <BDF>` - notice we're now examining only our target board, and we'll get more detail of kernels and memories present in the design.

## Enable Debugging Access to the Board

To access the design for any form of debugging we require a virtual JTAG cable and debug server connected to it. Both of these can be started with one command:
```
debug_hw --xvc_pcie /dev/xfpga/xvc_pub.u<BDF number> --hw_server --hw_server_port 3121 --xvc_pcie_port 10200
```
For a BDF of the form `0000:B:D.F` the BDF number is `B*256 + D*8 + F`. Port numbers shown are default values and can be changed. This command will start a debug server with Xilinx Virtual Cable attached to the target board.

## Access and Update CCLO Firmware

You can inspect Microblaze targets inside the design with `xsct` either from the machine hosting the Alveos or another one:
```
xsct -nopdisp
xsct% connect -xvc <host>:<port>
xsct% targets
```
`host` and `port` are the hostname where the debug server was started, and the port of the XVC, typically 10200. Before exiting XSCT make a note of the identified Microblaze targets visible through the debug hub. You should see at least one Microblaze if your design includes the CCLO kernel.

To program a new firmware ELF file into the CCLO(s), run `xsct -nodisp update_elf.tcl <host> <port> <ELF file> <target(s)>`. Notice that multiple targets can be programmed with the provided ELF, e.g. if there are multiple CCLOs in a design.

You can now connect to the Microblaze target from a remote Vitis GUI for code debugging.
