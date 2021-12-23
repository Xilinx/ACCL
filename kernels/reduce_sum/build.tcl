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
set dtype [lindex $argv 2]
set dwidth [lindex $argv 3]

set ipname reduce_sum_${dtype}

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


open_project build_${ipname}

add_files reduce_sum.cpp -cflags "-std=c++14 -DDATA_WIDTH=${dwidth} -DREDUCE_HALF_PRECISION -I[pwd]/ -I[pwd]/../cclo/hls -DACCL_SYNTHESIS"
add_files -tb tb.cpp -cflags "-std=c++14 -DDATA_WIDTH=${dwidth} -DREDUCE_HALF_PRECISION -I[pwd]/ -I[pwd]/../cclo/hls -DACCL_SYNTHESIS"

set_top ${ipname}

open_solution sol1
config_rtl -module_prefix ${dtype}_${dwidth}_
config_export -format xo -library ACCL -output [pwd]/${ipname}.xo

if {$do_sim} {
    csim_design -clean
}

if {$do_syn} {
    set_part $device
    create_clock -period 4 -name default
    csynth_design
}

if {$do_export} {
    export_design
}

if ${do_cosim} {
    cosim_design
}

exit
