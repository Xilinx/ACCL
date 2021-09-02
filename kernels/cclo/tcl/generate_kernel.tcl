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
set boardpart [lindex $::argv 1]
set debug [lindex $::argv 2]
set xsafile [lindex $::argv 3]

# create project with correct target
create_project -force ccl_offload_ex ./ccl_offload_ex -part $fpgapart
set_property board_part $boardpart [current_project]

#open up kernel project and its BD
update_compile_order -fileset sources_1
create_bd_design ccl_offload_bd

# add our own ip to the repo
set_property  ip_repo_paths  {./hls/} [current_project]
update_ip_catalog

#rebuild bd
source  -notrace tcl/rebuild_bd.tcl

#add debug if requested
if [string equal $debug "dma"] {
  puts "Adding DMA debug to block design"
  source  -notrace tcl/debug_dma.tcl
} elseif [string equal $debug "pkt"] {
  puts "Adding (de)packetizer debug to block design"
  source  -notrace tcl/debug_pkt.tcl
} elseif [string equal $debug "arith"] {
  puts "Adding arithmetic debug to block design"
  source  -notrace tcl/debug_arith.tcl
} elseif [string equal $debug "control"] {
  puts "Adding control debug to block design"
  source  -notrace tcl/debug_control.tcl
} elseif [string equal $debug "all"] {
  puts "Adding all debug cores to block design"
  source  -notrace tcl/debug_dma.tcl
  source  -notrace tcl/debug_pkt.tcl
  source  -notrace tcl/debug_arith.tcl
  source  -notrace tcl/debug_control.tcl
}

# add wrapper
add_files -norecurse ./hdl/ccl_offload.v
update_compile_order -fileset sources_1
generate_target all [get_files  ./ccl_offload_ex/ccl_offload_ex.srcs/sources_1/bd/ccl_offload_bd/ccl_offload_bd.bd]
#build a .xsa file handoff
write_hw_platform -fixed -force -file $xsafile

# close and exit
close_project
exit
