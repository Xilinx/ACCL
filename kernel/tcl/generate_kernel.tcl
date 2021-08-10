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
create_project -force rtl_kernel_gen ./rtl_kernel_gen -part $fpgapart
set_property board_part $boardpart [current_project]

# create RTL kernel for Vitis
create_ip -name rtl_kernel_wizard -vendor xilinx.com -library ip -version 1.0 -module_name ccl_offload
# configure general properties of RTL kernel
set_property -dict [list CONFIG.KERNEL_NAME {ccl_offload} CONFIG.KERNEL_TYPE {Block_Design} CONFIG.NUM_CLOCKS {1} CONFIG.NUM_RESETS {1} CONFIG.DEBUG_ENABLED {1}] [get_ips ccl_offload]
set_property -dict [list CONFIG.Component_Name {ccl_offload} CONFIG.KERNEL_NAME {ccl_offload} CONFIG.KERNEL_VENDOR {Xilinx} CONFIG.KERNEL_LIBRARY {ACCL}] [get_ips ccl_offload]
# configure number, names and types of scalar arguments
set_property -dict [list CONFIG.NUM_INPUT_ARGS {9}] [get_ips ccl_offload] 
set_property -dict [list CONFIG.ARG00_NAME {call_type} CONFIG.ARG00_TYPE {uint}] [get_ips ccl_offload]
set_property -dict [list CONFIG.ARG01_NAME {byte_count} CONFIG.ARG01_TYPE {uint}] [get_ips ccl_offload]
set_property -dict [list CONFIG.ARG02_NAME {comm} CONFIG.ARG02_TYPE {uint}] [get_ips ccl_offload]
set_property -dict [list CONFIG.ARG03_NAME {root_src_dst} CONFIG.ARG03_TYPE {uint}] [get_ips ccl_offload]
set_property -dict [list CONFIG.ARG04_NAME {reduce_op} CONFIG.ARG04_TYPE {uint}] [get_ips ccl_offload]
set_property -dict [list CONFIG.ARG05_NAME {tag} CONFIG.ARG05_TYPE {uint}] [get_ips ccl_offload]
set_property -dict [list CONFIG.ARG06_NAME {buf0_type} CONFIG.ARG06_TYPE {uint}] [get_ips ccl_offload]
set_property -dict [list CONFIG.ARG07_NAME {buf1_type} CONFIG.ARG07_TYPE {uint}] [get_ips ccl_offload]
set_property -dict [list CONFIG.ARG07_NAME {buf2_type} CONFIG.ARG08_TYPE {uint}] [get_ips ccl_offload]
# configure number and properties of AXI Masters
set_property -dict [list CONFIG.NUM_M_AXI {3}] [get_ips ccl_offload] 
set_property -dict [list CONFIG.M00_AXI_NAME {m_axi_0} CONFIG.M00_AXI_ARG00_INTF {m_axi_0} CONFIG.M00_AXI_ARG00_NAME {buf0_ptr}] [get_ips ccl_offload]
set_property -dict [list CONFIG.M01_AXI_NAME {m_axi_1} CONFIG.M01_AXI_ARG00_INTF {m_axi_1} CONFIG.M01_AXI_ARG00_NAME {buf1_ptr}] [get_ips ccl_offload]
set_property -dict [list CONFIG.M02_AXI_NAME {m_axi_2} CONFIG.M02_AXI_ARG00_INTF {m_axi_2} CONFIG.M02_AXI_ARG00_NAME {buf2_ptr}] [get_ips ccl_offload]
# configure number and properties of AXI Streams
set_property -dict [list CONFIG.NUM_AXIS {18}] [get_ips ccl_offload] 
set_property -dict [list CONFIG.AXIS00_NAME {s_axis_udp_rx_data} CONFIG.AXIS00_MODE {read_only} CONFIG.AXIS00_NUM_BYTES {64}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS01_NAME {m_axis_udp_tx_data} CONFIG.AXIS01_MODE {write_only} CONFIG.AXIS01_NUM_BYTES {64}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS02_NAME {s_axis_tcp_notification} CONFIG.AXIS02_MODE {read_only} CONFIG.AXIS02_NUM_BYTES {16}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS03_NAME {m_axis_tcp_read_pkg} CONFIG.AXIS03_MODE {write_only} CONFIG.AXIS03_NUM_BYTES {4}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS04_NAME {s_axis_tcp_rx_meta} CONFIG.AXIS04_MODE {read_only} CONFIG.AXIS04_NUM_BYTES {2}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS05_NAME {s_axis_tcp_rx_data} CONFIG.AXIS05_MODE {read_only} CONFIG.AXIS05_NUM_BYTES {64}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS06_NAME {m_axis_tcp_tx_meta} CONFIG.AXIS06_MODE {write_only} CONFIG.AXIS06_NUM_BYTES {4}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS07_NAME {m_axis_tcp_tx_data} CONFIG.AXIS07_MODE {write_only} CONFIG.AXIS07_NUM_BYTES {64}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS08_NAME {s_axis_tcp_tx_status} CONFIG.AXIS08_MODE {read_only} CONFIG.AXIS08_NUM_BYTES {8}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS09_NAME {m_axis_tcp_open_connection} CONFIG.AXIS09_MODE {write_only} CONFIG.AXIS09_NUM_BYTES {8}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS10_NAME {s_axis_tcp_open_status} CONFIG.AXIS10_MODE {read_only} CONFIG.AXIS10_NUM_BYTES {16}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS11_NAME {m_axis_tcp_listen_port} CONFIG.AXIS11_MODE {write_only} CONFIG.AXIS11_NUM_BYTES {2}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS12_NAME {s_axis_tcp_port_status} CONFIG.AXIS12_MODE {read_only} CONFIG.AXIS12_NUM_BYTES {1}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS13_NAME {s_axis_krnl} CONFIG.AXIS13_MODE {read_only} CONFIG.AXIS13_NUM_BYTES {64}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS14_NAME {m_axis_krnl} CONFIG.AXIS14_MODE {write_only} CONFIG.AXIS14_NUM_BYTES {64}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS15_NAME {s_axis_arith_res} CONFIG.AXIS15_MODE {read_only} CONFIG.AXIS15_NUM_BYTES {64}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS16_NAME {m_axis_arith_op0} CONFIG.AXIS16_MODE {write_only} CONFIG.AXIS16_NUM_BYTES {64}] [get_ips ccl_offload]
set_property -dict [list CONFIG.AXIS17_NAME {m_axis_arith_op1} CONFIG.AXIS17_MODE {write_only} CONFIG.AXIS17_NUM_BYTES {64}] [get_ips ccl_offload]

generate_target {instantiation_template} [get_files ./rtl_kernel_gen/rtl_kernel_gen.srcs/sources_1/ip/ccl_offload/ccl_offload.xci]
update_compile_order -fileset sources_1
open_example_project -force -in_process -dir . [get_ips ccl_offload]

#open up kernel project and its BD
update_compile_order -fileset sources_1
open_bd_design [get_files ./ccl_offload_ex/ccl_offload_ex.srcs/sources_1/bd/ccl_offload_bd/ccl_offload_bd.bd]

# remove default design
delete_bd_objs [get_bd_intf_nets] [get_bd_nets] [get_bd_intf_ports] [get_bd_ports] [get_bd_cells]

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
} elseif [string equal $debug "all"] {
  puts "Adding all debug cores to block design"
  source  -notrace tcl/debug_all.tcl
}
#generate final version of bd 
update_compile_order -fileset sources_1
generate_target all [get_files  ./ccl_offload_ex/ccl_offload_ex.srcs/sources_1/bd/ccl_offload_bd/ccl_offload_bd.bd]
#build a .xsa file handoff
write_hw_platform -fixed -force -file $xsafile

# close and exit
close_project
exit
