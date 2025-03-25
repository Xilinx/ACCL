# /*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# *******************************************************************************/

set command [lindex $argv 0]
set device [lindex $argv 1]

set do_sim 0
set do_syn 0
set do_export 0
set do_cosim 0

switch $command {
    "sim" {
        set do_sim 1
    }
    "syn" {
        set do_syn 1
    }
    "ip" {
        set do_syn 1
        set do_export 1
    }
    "cosim" {
        set do_syn 1
        set do_cosim 1
    }
    "all" {
        set do_sim 1
        set do_syn 1
        set do_export 1
        set do_cosim 1
    }
    default {
        puts "Unrecognized command"
        exit
    }
}

open_project build_dma_mover.${device}

add_files dma_mover.cpp -cflags "-std=c++14 -I[pwd]/../../../../driver/hls -I[pwd]/../eth_intf/ -I[pwd]/../../../../hlslib/include/hlslib/xilinx -I[pwd]/../segmenter -I[pwd]/../../fw/sw_apps/ccl_offload_control/src -DHLSLIB_SYNTHESIS"
add_files -tb tb_dma_mover.cpp -cflags "-std=c++14 -I[pwd]/../../../../driver/hls -I[pwd]/../eth_intf/ -I[pwd]/../../../../hlslib/include/hlslib/xilinx -I[pwd]/../segmenter -I[pwd]/../../fw/sw_apps/ccl_offload_control/src -DHLSLIB_SYNTHESIS"

set_top dma_mover
open_solution sol1
config_interface -m_axi_addr64=false 

if {$do_sim} {
    csim_design -clean
}

if {$do_syn} {
    set_part $device
    create_clock -period 4 -name default
    csynth_design
}

if {$do_export} {
    config_export -format ip_catalog
    export_design
}

if ${do_cosim} {
    cosim_design
}

exit