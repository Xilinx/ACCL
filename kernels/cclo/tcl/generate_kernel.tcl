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
set fpgapart [lindex $::argv 0]
set hw_debug_level [lindex $::argv 1]
set xsafile [lindex $::argv 2]
set stacktype [lindex $::argv 3]
set en_dma [lindex $::argv 4]
set en_arith [lindex $::argv 5]
set en_compress [lindex $::argv 6]
set en_extkrnl [lindex $::argv 7]
set mb_debug_level [lindex $::argv 8]
set commit_hash [lindex $::argv 9]

# create project with correct target
create_project -force ccl_offload_ex ./ccl_offload_ex -part $fpgapart
set_property target_language verilog [current_project]
set_property simulator_language MIXED [current_project]
set_property coreContainer.enable false [current_project]
update_compile_order -fileset sources_1
create_bd_design ccl_offload_bd

# add our own ip to the repo
set_property  ip_repo_paths  {../hls/} [current_project]
update_ip_catalog

#rebuild bd
source -notrace ../tcl/rebuild_bd.tcl
create_root_design $stacktype $en_dma $en_arith $en_compress $en_extkrnl $mb_debug_level $commit_hash

#add debug if requested
if [string equal $hw_debug_level "dma"] {
  puts "Adding DMA debug to block design"
  source  -notrace ../tcl/debug_dma.tcl
} elseif [string equal $hw_debug_level "pkt"] {
  puts "Adding (de)packetizer debug to block design"
  source  -notrace ../tcl/debug_pkt.tcl
} elseif [string equal $hw_debug_level "arith"] {
  puts "Adding arithmetic debug to block design"
  source  -notrace ../tcl/debug_arith.tcl
} elseif [string equal $hw_debug_level "control"] {
  puts "Adding control debug to block design"
  source  -notrace ../tcl/debug_control.tcl
} elseif [string equal $hw_debug_level "all"] {
  puts "Adding all debug cores to block design"
  source  -notrace ../tcl/debug_dma.tcl
  source  -notrace ../tcl/debug_pkt.tcl
  source  -notrace ../tcl/debug_arith.tcl
  source  -notrace ../tcl/debug_control.tcl
}

# add wrapper
add_files -norecurse ../hdl/ccl_offload.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
generate_target all [get_files  ./ccl_offload_ex/ccl_offload_ex.srcs/sources_1/bd/ccl_offload_bd/ccl_offload_bd.bd]
#build a .xsa file handoff
write_hw_platform -fixed -force -file $xsafile


#create ila_top for commands on cclo interf
#create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_top
#set_property -dict [ list CONFIG.C_PROBE0_WIDTH {32} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE4_WIDTH {512} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE5_WIDTH {8} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE8_WIDTH {512} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE9_WIDTH {8} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE12_WIDTH {512} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE13_WIDTH {8} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE16_WIDTH {512} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE17_WIDTH {8} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE20_WIDTH {512} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE21_WIDTH {8} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE24_WIDTH {64} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE27_WIDTH {32} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE30_WIDTH {64} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE33_WIDTH {512} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_PROBE36_WIDTH {8} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_NUM_OF_PROBES {37} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_EN_STRG_QUAL {1} ] [get_ips ila_top]
#set_property -dict [ list CONFIG.C_DATA_DEPTH {1024} ] [get_ips ila_top]
# close and exit
close_project
exit
