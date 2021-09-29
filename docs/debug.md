# Debugging
The Microblaze debug interface is by default connected up to the debug bridge. 
Optionally, the `DEBUG` switch can be passed to make to enable logic analyzer insertion on the DMA logic.


## MicroBlaze code

1. Program the xclbin:

	``xbutil program -d <board_idx> -p <.xclbin path>`` 

	More info [sdaccel_doc xbutil program html doc](https://www.xilinx.com/html_docs/xilinx2019_1/sdaccel_doc/kim1536963234393.html)

2.  start a debug server which will be accessible via TCP. The debug server has to be active for the entire duration of the debug session.  You can open it using ``&``, using ``tmux`` or ``screen``. The suffix of the xvc_pub file is specific to your environment, just take the device file that is created on your system. 
This sets up a connection over pcie to the debug bridge in the device.

	-  ``tmux``

	-  ``debug_hw --xvc_pcie /dev/xfpga/xvc_pub.<driver_id> --hw_server --hw_server_port 3121 --xvc_pcie_port 10200``

More info [here](https://developer.xilinx.com/en/articles/debugging-your-applications-on-an-alveo-data-center-accelerator-.html)

Example:
````
(base) danielep@xirxlabs53:~$ debug_hw --hw_server --xvc_pcie /dev/xfpga/xvc_pub.u15100  --xvc_pcie_port 10202 --hw_server_port 3122
launching xvc_pcie...
/proj/xbuilds/SWIP/2020.2_1118_1232/installs/lin64/Vivado/2020.2/bin/xvc_pcie -d /dev/xfpga/xvc_pub.u15105.0 -s TCP::10202
launching hw_server...
/proj/xbuilds/SWIP/2020.2_1118_1232/installs/lin64/Vivado/2020.2/bin/hw_server -sTCP::3122

****************************
*** Press Ctrl-C to exit ***
****************************

````

NOTE: From pcie bus device function (bdf) we can obtain the path under /dev/xvc_pcie path under /dev.xfpga (mxxxxx.0 or uxxxxx.0) usign the simple formulae: (b x 256 + d x 8 + f)*8 TODO:check 

To obtain b-d-f use xbutil 

````
(base) danielep@xirxlabs53:~/mpi_bnn_pynq_demo/XCCL_Offload$ xbutil scan
INFO: Found total 3 card(s), 3 are usable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
System Configuration
OS name:        Linux
Release:        4.4.0-210-generic
Version:        #242-Ubuntu SMP Fri Apr 16 09:57:56 UTC 2021
Machine:        x86_64
Model:          Precision 7920 Rack
CPU cores:      32
Memory:         128585 MB
Glibc:          2.23
Distribution:   Ubuntu 16.04.5 LTS
Now:            Mon Apr 26 11:14:54 2021 GMT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XRT Information
Version:        2.9.317
Git Hash:       b0230e59e22351fb957dc46a6e68d7560e5f630c
Git Branch:     2020.2_PU1
Build Date:     2021-03-13 05:10:40
XOCL:           2.9.317,b0230e59e22351fb957dc46a6e68d7560e5f630c
XCLMGMT:        2.9.317,b0230e59e22351fb957dc46a6e68d7560e5f630c
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 [0] 0000:d8:00.1 xilinx_u250_xdma_201830_2(ID=0x5d14fbe6) user(inst=130)
 [1] 0000:af:00.1 xilinx_u280_xdma_201920_3(ID=0x5e278820) user(inst=129)
 [2] 0000:3b:00.1 xilinx_u250_xdma_201830_2(ID=0x5d14fbe6) user(inst=128) 
 ````
 so for board_index = 2 : bus = 0, device = 3b and function = 0.1 => path = (0*256 + h3b*d8 + 0.125)*8 = 3b00+1
 In our server:
- /dev/xfpga/xvc_pub.u55297.0 u250 device 0
- /dev/xfpga/xvc_pub.u44801.0 u280 device 1
- /dev/xfpga/xvc_pub.u15105.0 u250 device 2

now detach from ``tmux`` session or open another ``screen``

3. Now that the hw server is started we can connect to it using Xilinx® Software Command-Line tool  (``XSCT``). 
	0. start xsct:

	``xsct -interactive``

	1. connect to the hw_server instance that we have just started in the other screen using Xilinx Virtual Cable connection:

	``connect -xvc 127.0.0.1:10200``

	2. list all the available targets attached to the hw_server:

		``targets``

		Example:
		````
		xsct% targets
		1  PS TAP
			2  PMU
			3  PL
				4  MicroBlaze Debug Module at USER2
				5  MicroBlaze #0 (Running)
				6  MicroBlaze #1 (Running)
		7  PSU
			8  RPU (Reset)
				9  Cortex-R5 #0 (No Power)
			10  Cortex-R5 #1 (No Power)
			11  APU
			12  Cortex-A53 #0 (Running)
			13  Cortex-A53 #1 (Running)
			14  Cortex-A53 #2 (Running)
			15  Cortex-A53 #3 (Running)
		16  debug_bridge
			17  00000000
			18  Legacy Debug Hub
			19  MicroBlaze Debug Module at USER1.1.2.2
			20  MicroBlaze #0 (Running)
		
		````
		As you can see it includes all the targets available on the server, including cards, processors in the cards (e.g. Zynq devices). Each device as associated a numeric id that can be used in the next step.

	3. Connect to the target we want to debug:
	
		``targets <id_of_target>``

		````
		targets 12
		xsct% Info: Cortex-A53 #0 (target 12) Stopped at 0xffff0000 (Reset Catch)`
		````
	4. run your experiments:
		there are several things you can:
		1. read memory content:

			``mrd <address> ``

		2. write memory:

			``mwr <address> <value>``
		
		3. list all the available targets:

			``targets``
		
		3. switch to another target:

			``targets <id>``

		4. reset the target:

			``rst -proc``
		
		5. download the ``.elf`` executable on the target:

			``down <path to the executable>``

            e.g. load the most recent ccl_offload control software from its default build path  

            ``dow ../../kernel/vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf``

			**NOTE**: this gives the debugger access to the code that the MicroBlaze is executing. In this way it is possible to set breakpoints by function name and avoids using hexadecimal addresses. Note that debug symbols has to be included in the  executable.
		
		6. add a breakpoint:

			``bpadd main``

		7. stop code execution:

			``stop``
		
		8. list/enable/disable a breakpoint:

			``bplist``/``bpenable <bp_id>``/``bpdisable <bp_id>``
		
		9. step over the current line of code and takes you to the next line 

			``stp``

		10. resume execution from a breakpoint

			``con``
		
		11. print variable value

			``print <variable>``
		
    	12. you can partially automate this tasks writing instructions in ``.tcl`` files. Then to execute them simply type:

            ``source <path of .tcl script>``

            e.g. to update ccl_offload control software of a quad design type:

            ``source update_elf_quad.tcl``

            An example of ``.tcl`` file is [update_elf_quad.tcl](update_elf_quad.tcl)

            ````
            targets 17
            rst -proc
            dow ../../kernel/vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf
            con
            targets 19
            rst -proc
            dow ../../kernel/vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf
            con
            [...]

            ````

            More info [here](https://www.xilinx.com/html_docs/xilinx2018_1/SDK_Doc/xsct/use_cases/xsct_howtoruntclscriptfiles.html?hl=tcl)

            
	5. disconnect from hw_server

		``disconnect``

	6. exit from xsct

		``exit``
	
	7. go to the other screen and kill ``hwserver``

		``CTRL+C``

More info at [ Xilinx® Software Command-Line tool html doc](https://www.xilinx.com/html_docs/xilinx2018_1/SDK_Doc/xsct/intro/xsct_commands.html)


It is also possible to use the Vitis gui. 
After configuring a breakpoint in main, open Vitis gui, opening the microblaze project, and start a debug session which attaches to the running target. 
Since we added a breakpoint at main, after attaching, the Microblaze will still be in that stopped state, and you can then start a debug session.