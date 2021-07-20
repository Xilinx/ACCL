
## Setup the environment

0.  to do it in a fast way is better to create a .bashrc file the gets executed everytime bash is launched. So :

    ````$ nano ~/.bashrc````:

    And write the following: 

    0. adds platforms definitions
        
        ````export  PLATFORM_REPO_PATHS=/proj/xlabs_t3/users/yamanu/vitis_platforms/````

    0. adds the correct version of vitis (vitis includes vivado, the 2020.2 works) to the path

        ````source /proj/xbuilds/SWIP/2020.2_1118_1232/installs/lin64/Vitis/2020.2/settings64.sh````

    0. adds utils (e.g. xbutil ) and other programs

        ````source /opt/xilinx/xrt/setup.sh````
0. git submodule init
0. git submodule update

0. Synthesize the design: 
    0. look for the right platform as you'll need to explicitly pass the PLATFORM=... argument to ````kernel/make```` you can get from:` xbutil list`  or ` xbutil query -d0 `

    0. ````cd CXXL_OFFLOAD/kernel````

    0. ````nano Makefile````

    0. Platform should reflect what has been found by xbutil
    ````
    PLATFORM ?= xilinx_u250_xdma_201920_1
    XSA := $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))`
    ...
    ````
    0. Debug indicates whether to include chipscopes inside the design.
    ````
    DEBUG ?= none
     ...
    ````

    0. There are 5 modes (vnx|vnx_dual|loopback|dual|quad): 
    ````
    MODE ?= quad
    ...
    ````

        a. vnx      : uses 1 NIC with tcp stack
        b. vnx_dual : uses 2 NICs with tcp stack
        c. loopback : uses 1 XCCL_offload with axi packetizers to emaulte the stack with a single board
        c. dual     : uses 2 XCCL_offload with axi packetizers to emaulte the stack with a single board
        d. quad     : uses 4 XCCL_offload with axi packetizers to emaulte the stack with a single board 
    0. How does the makefile work?
        



