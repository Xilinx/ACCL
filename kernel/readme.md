
## Setup the environment

In Makefile 
1. Platform should reflect what has been found by xbutil
````
PLATFORM ?= xilinx_u280_xdma_201920_3
    ...
...
````
1. Debug indicates whether to include chipscopes logic analyzer inside the design (default none. Other values all,dma,pkt)
````
DEBUG ?= none
    ...
````

You are ready to go.

1. Source Vitis 2020.2
1. run make to build ``ccl_offload.xo`` under ``ccl_offload_ex/exports``
        