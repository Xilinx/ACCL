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

set stacktype [lindex $::argv 0]
set en_dma [lindex $::argv 1]
set en_arith [lindex $::argv 2]
set en_compress [lindex $::argv 3]
set en_extkrnl [lindex $::argv 4]
set mb_debug_level [lindex $::argv 5]

# open project
open_project ./ccl_offload_ex/ccl_offload_ex.xpr

#run kernel packaging
reset_run synth_1
set extra_synth_options "-mode out_of_context -verilog_define AXILITE_ADR_BITS=13 "
if { $en_arith == 1 } { set extra_synth_options "$extra_synth_options -verilog_define ARITH_ENABLE " }
if { $en_compress == 1 } { set extra_synth_options "$extra_synth_options -verilog_define COMPRESSION_ENABLE " }
if { $en_dma == 1 } { 
    set extra_synth_options "$extra_synth_options -verilog_define DMA_ENABLE " 
}
if { $en_extkrnl == 1 } { set extra_synth_options "$extra_synth_options -verilog_define STREAM_ENABLE " }
if { $stacktype == "TCP" } { set extra_synth_options "$extra_synth_options -verilog_define TCP_ENABLE " }
if { $stacktype == "RDMA" } { set extra_synth_options "$extra_synth_options -verilog_define RDMA_ENABLE " }
if { $mb_debug_level > 0 } { set extra_synth_options "$extra_synth_options -verilog_define MB_DEBUG_ENABLE " }
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value $extra_synth_options -objects [get_runs synth_1]
launch_runs synth_1 -jobs 6
wait_on_run [get_runs synth_1]
open_run synth_1 -name synth_1
rename_ref -prefix_all ccl_offload_
refresh_meminit
write_checkpoint ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp
write_xdc ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.xdc
close_design

# close and exit
close_project
exit