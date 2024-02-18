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
set num_dma 2

# create project with correct target
create_project -force external_dma ./external_dma -part $fpgapart
set_property target_language verilog [current_project]
set_property simulator_language MIXED [current_project]
set_property coreContainer.enable false [current_project]
update_compile_order -fileset sources_1
create_bd_design external_dma_bd
open_bd_design external_dma_bd

# Create ports
create_bd_port -dir I -type clk -freq_hz 250000000 ap_clk
create_bd_port -dir I -type rst ap_rst_n

create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_control
set_property -dict [ list \
  CONFIG.ADDR_WIDTH {16} \
  CONFIG.ARUSER_WIDTH {0} \
  CONFIG.AWUSER_WIDTH {0} \
  CONFIG.BUSER_WIDTH {0} \
  CONFIG.DATA_WIDTH {32} \
  CONFIG.FREQ_HZ {250000000} \
  CONFIG.HAS_BRESP {1} \
  CONFIG.HAS_BURST {0} \
  CONFIG.HAS_CACHE {0} \
  CONFIG.HAS_LOCK {0} \
  CONFIG.HAS_PROT {1} \
  CONFIG.HAS_QOS {0} \
  CONFIG.HAS_REGION {0} \
  CONFIG.HAS_RRESP {1} \
  CONFIG.HAS_WSTRB {1} \
  CONFIG.ID_WIDTH {0} \
  CONFIG.MAX_BURST_LENGTH {1} \
  CONFIG.NUM_READ_OUTSTANDING {1} \
  CONFIG.NUM_READ_THREADS {1} \
  CONFIG.NUM_WRITE_OUTSTANDING {1} \
  CONFIG.NUM_WRITE_THREADS {1} \
  CONFIG.PROTOCOL {AXI4LITE} \
  CONFIG.READ_WRITE_MODE {READ_WRITE} \
  CONFIG.RUSER_BITS_PER_BYTE {0} \
  CONFIG.RUSER_WIDTH {0} \
  CONFIG.SUPPORTS_NARROW_BURST {0} \
  CONFIG.WUSER_BITS_PER_BYTE {0} \
  CONFIG.WUSER_WIDTH {0} \
] [get_bd_intf_port s_axi_control]

create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_s2mm
set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \
                          CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \
                          CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {8} CONFIG.TID_WIDTH {0} \
                          CONFIG.TUSER_WIDTH {0} ] [get_bd_intf_port s_axis_s2mm]

create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_mm2s
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] [get_bd_intf_port m_axis_mm2s]

create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_mm2s_cmd
set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \
                          CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \
                          CONFIG.TDATA_NUM_BYTES {13} CONFIG.TDEST_WIDTH {8} CONFIG.TID_WIDTH {0} \
                          CONFIG.TUSER_WIDTH {0} ] [get_bd_intf_port s_axis_mm2s_cmd]

create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_mm2s_sts
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] [get_bd_intf_port m_axis_mm2s_sts]

create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_s2mm_cmd
set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} \
                          CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} \
                          CONFIG.TDATA_NUM_BYTES {13} CONFIG.TDEST_WIDTH {8} CONFIG.TID_WIDTH {0} \
                          CONFIG.TUSER_WIDTH {0} ] [get_bd_intf_port s_axis_s2mm_cmd]

create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_s2mm_sts
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] [get_bd_intf_port m_axis_s2mm_sts]

set interfaces "s_axi_control:s_axis_s2mm:m_axis_mm2s:s_axis_s2mm_cmd:m_axis_s2mm_sts:s_axis_mm2s_cmd:m_axis_mm2s_sts"

# Instantiate performance monitor
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_perf_mon:5.0 axi_perf_mon_0
set_property -dict [list \
  CONFIG.C_ENABLE_EVENT_LOG {0} \
  CONFIG.C_ENABLE_PROFILE {1} \
  CONFIG.C_ENABLE_TRACE {0} \
  CONFIG.C_EN_AXI_DEBUG {0} \
  CONFIG.C_NUM_MONITOR_SLOTS $num_dma \
  CONFIG.C_SLOT_0_AXI_PROTOCOL {AXI4} \
] [get_bd_cells axi_perf_mon_0]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_perf_mon_0/s_axi_aclk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_perf_mon_0/s_axi_aresetn]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_perf_mon_0/core_aclk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_perf_mon_0/core_aresetn]
connect_bd_intf_net [get_bd_intf_ports s_axi_control] [get_bd_intf_pins axi_perf_mon_0/S_AXI]
assign_bd_address

# DMA connections
for {set i 0} {$i < $num_dma} {incr i} {
  # Create aximm port, and set properties
  create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 m_axi_${i}
  set_property -dict [ list \
    CONFIG.ADDR_WIDTH {64} \
    CONFIG.DATA_WIDTH {512} \
    CONFIG.FREQ_HZ {250000000} \
    CONFIG.HAS_BRESP {0} \
    CONFIG.HAS_BURST {0} \
    CONFIG.HAS_CACHE {0} \
    CONFIG.HAS_LOCK {0} \
    CONFIG.HAS_PROT {0} \
    CONFIG.HAS_QOS {0} \
    CONFIG.HAS_REGION {0} \
    CONFIG.HAS_WSTRB {1} \
    CONFIG.NUM_READ_OUTSTANDING {1} \
    CONFIG.NUM_WRITE_OUTSTANDING {1} \
    CONFIG.PROTOCOL {AXI4} \
    CONFIG.READ_WRITE_MODE {READ_WRITE}
  ] [get_bd_intf_port m_axi_${i}]
  set interfaces "$interfaces:m_axi_${i}"
  # Create instance: dma_0, and set properties
  create_bd_cell -type ip -vlnv xilinx.com:ip:axi_datamover:5.1 dma_${i}
  set_property -dict [ list \
    CONFIG.c_addr_width {64} \
    CONFIG.c_dummy {0} \
    CONFIG.c_enable_mm2s {1} \
    CONFIG.c_include_mm2s {Full} \
    CONFIG.c_include_mm2s_stsfifo {true} \
    CONFIG.c_m_axi_mm2s_data_width {512} \
    CONFIG.c_m_axi_mm2s_id_width {0} \
    CONFIG.c_m_axi_s2mm_data_width {512} \
    CONFIG.c_m_axi_s2mm_id_width {0} \
    CONFIG.c_m_axis_mm2s_tdata_width {512} \
    CONFIG.c_mm2s_btt_used {23} \
    CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_mm2s_include_sf {true} \
    CONFIG.c_s2mm_btt_used {23} \
    CONFIG.c_s2mm_burst_size {64} \
    CONFIG.c_s2mm_support_indet_btt {true} \
    CONFIG.c_s_axis_s2mm_tdata_width {512} \
    CONFIG.c_include_mm2s_dre {true} \
    CONFIG.c_include_s2mm_dre {true} \
    CONFIG.c_single_interface {1} \
  ] [get_bd_cell dma_${i}]

  connect_bd_intf_net [get_bd_intf_ports m_axi_${i}] [get_bd_intf_pins dma_${i}/M_AXI]

  connect_bd_intf_net [get_bd_intf_pins axi_perf_mon_0/SLOT_${i}_AXI] [get_bd_intf_pins dma_${i}/M_AXI]
  connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_perf_mon_0/slot_${i}_axi_aclk]
  connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_perf_mon_0/slot_${i}_axi_aresetn]

  connect_bd_net [get_bd_ports ap_clk] \
                 [get_bd_pins dma_${i}/m_axi_mm2s_aclk] \
                 [get_bd_pins dma_${i}/m_axi_s2mm_aclk] \
                 [get_bd_pins dma_${i}/m_axis_s2mm_cmdsts_awclk] \
                 [get_bd_pins dma_${i}/m_axis_mm2s_cmdsts_aclk]

  connect_bd_net [get_bd_ports ap_rst_n] \
                 [get_bd_pins dma_${i}/m_axi_mm2s_aresetn] \
                 [get_bd_pins dma_${i}/m_axi_s2mm_aresetn] \
                 [get_bd_pins dma_${i}/m_axis_s2mm_cmdsts_aresetn] \
                 [get_bd_pins dma_${i}/m_axis_mm2s_cmdsts_aresetn]
}

if { $num_dma > 1 } {
  # create switches
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_s2mm_data
  set_property -dict [list \
    CONFIG.DECODER_REG {1} \
    CONFIG.HAS_TSTRB.VALUE_SRC USER \
    CONFIG.HAS_TSTRB {0} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.NUM_MI $num_dma \
    CONFIG.NUM_SI {1} \
    CONFIG.TDATA_NUM_BYTES {64} \
    CONFIG.TDEST_WIDTH.VALUE_SRC USER \
    CONFIG.ROUTING_MODE {0} \
    CONFIG.TDEST_WIDTH {8} \
  ] [get_bd_cells axis_switch_s2mm_data]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_mm2s_data
  set_property -dict [list \
    CONFIG.DECODER_REG {1} \
    CONFIG.HAS_TSTRB.VALUE_SRC USER \
    CONFIG.HAS_TSTRB {0} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI $num_dma \
    CONFIG.TDATA_NUM_BYTES {64} \
    CONFIG.ROUTING_MODE {0} \
    CONFIG.ARB_ON_TLAST {1} \
    CONFIG.ARB_ALGORITHM {3} \
    CONFIG.ARB_ON_MAX_XFERS {0} \
  ] [get_bd_cells axis_switch_mm2s_data]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_s2mm_cmd
  set_property -dict [list \
    CONFIG.DECODER_REG {1} \
    CONFIG.HAS_TSTRB.VALUE_SRC USER \
    CONFIG.HAS_TSTRB {0} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.NUM_MI $num_dma \
    CONFIG.NUM_SI {1} \
    CONFIG.TDATA_NUM_BYTES {13} \
    CONFIG.TDEST_WIDTH.VALUE_SRC USER \
    CONFIG.ROUTING_MODE {0} \
    CONFIG.TDEST_WIDTH {8} \
  ] [get_bd_cells axis_switch_s2mm_cmd]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_mm2s_cmd
  set_property -dict [list \
    CONFIG.DECODER_REG {1} \
    CONFIG.HAS_TSTRB.VALUE_SRC USER \
    CONFIG.HAS_TSTRB {0} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.NUM_MI $num_dma \
    CONFIG.NUM_SI {1} \
    CONFIG.TDATA_NUM_BYTES {13} \
    CONFIG.TDEST_WIDTH.VALUE_SRC USER \
    CONFIG.ROUTING_MODE {0} \
    CONFIG.TDEST_WIDTH {8} \
  ] [get_bd_cells axis_switch_mm2s_cmd]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_mm2s_sts
  set_property -dict [list \
    CONFIG.DECODER_REG {1} \
    CONFIG.HAS_TSTRB.VALUE_SRC USER \
    CONFIG.HAS_TSTRB {0} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI $num_dma \
    CONFIG.TDATA_NUM_BYTES {1} \
    CONFIG.ROUTING_MODE {0} \
    CONFIG.ARB_ON_TLAST {1} \
    CONFIG.ARB_ALGORITHM {3} \
    CONFIG.ARB_ON_MAX_XFERS {0} \
  ] [get_bd_cells axis_switch_mm2s_sts]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_s2mm_sts
  set_property -dict [list \
    CONFIG.DECODER_REG {1} \
    CONFIG.HAS_TSTRB.VALUE_SRC USER \
    CONFIG.HAS_TSTRB {0} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI $num_dma \
    CONFIG.TDATA_NUM_BYTES {4} \
    CONFIG.ROUTING_MODE {0} \
    CONFIG.ARB_ON_TLAST {1} \
    CONFIG.ARB_ALGORITHM {3} \
    CONFIG.ARB_ON_MAX_XFERS {0} \
  ] [get_bd_cells axis_switch_s2mm_sts]

  connect_bd_intf_net [get_bd_intf_pins s_axis_mm2s_cmd] [get_bd_intf_pins axis_switch_mm2s_cmd/S00_AXIS]
  connect_bd_intf_net [get_bd_intf_pins m_axis_mm2s_sts] [get_bd_intf_pins axis_switch_mm2s_sts/M00_AXIS]
  connect_bd_intf_net [get_bd_intf_pins s_axis_s2mm_cmd] [get_bd_intf_pins axis_switch_s2mm_cmd/S00_AXIS]
  connect_bd_intf_net [get_bd_intf_pins m_axis_s2mm_sts] [get_bd_intf_pins axis_switch_s2mm_sts/M00_AXIS]
  connect_bd_intf_net [get_bd_intf_pins s_axis_s2mm] [get_bd_intf_pins axis_switch_s2mm_data/S00_AXIS]
  connect_bd_intf_net [get_bd_intf_pins m_axis_mm2s] [get_bd_intf_pins axis_switch_mm2s_data/M00_AXIS]

  connect_bd_net [get_bd_ports ap_clk] \
                 [get_bd_pins axis_switch_mm2s_cmd/aclk] \
                 [get_bd_pins axis_switch_mm2s_sts/aclk] \
                 [get_bd_pins axis_switch_s2mm_cmd/aclk] \
                 [get_bd_pins axis_switch_s2mm_sts/aclk] \
                 [get_bd_pins axis_switch_s2mm_data/aclk] \
                 [get_bd_pins axis_switch_mm2s_data/aclk]

  connect_bd_net [get_bd_ports ap_rst_n] \
                 [get_bd_pins axis_switch_mm2s_cmd/aresetn] \
                 [get_bd_pins axis_switch_mm2s_sts/aresetn] \
                 [get_bd_pins axis_switch_s2mm_cmd/aresetn] \
                 [get_bd_pins axis_switch_s2mm_sts/aresetn] \
                 [get_bd_pins axis_switch_s2mm_data/aresetn] \
                 [get_bd_pins axis_switch_mm2s_data/aresetn]

  for {set i 0} {$i < $num_dma} {incr i} {
    connect_bd_intf_net [get_bd_intf_pins dma_${i}/S_AXIS_MM2S_CMD] [get_bd_intf_pins axis_switch_mm2s_cmd/M0${i}_AXIS]
    connect_bd_intf_net [get_bd_intf_pins dma_${i}/M_AXIS_MM2S_STS] [get_bd_intf_pins axis_switch_mm2s_sts/S0${i}_AXIS]
    connect_bd_intf_net [get_bd_intf_pins dma_${i}/S_AXIS_S2MM_CMD] [get_bd_intf_pins axis_switch_s2mm_cmd/M0${i}_AXIS]
    connect_bd_intf_net [get_bd_intf_pins dma_${i}/M_AXIS_S2MM_STS] [get_bd_intf_pins axis_switch_s2mm_sts/S0${i}_AXIS]
    connect_bd_intf_net [get_bd_intf_pins dma_${i}/S_AXIS_S2MM] [get_bd_intf_pins axis_switch_s2mm_data/M0${i}_AXIS]
    connect_bd_intf_net [get_bd_intf_pins dma_${i}/M_AXIS_MM2S] [get_bd_intf_pins axis_switch_mm2s_data/S0${i}_AXIS]
  }
} else {
  connect_bd_intf_net [get_bd_intf_pins s_axis_mm2s_cmd] [get_bd_intf_pins dma_0/S_AXIS_MM2S_CMD]
  connect_bd_intf_net [get_bd_intf_pins m_axis_mm2s_sts] [get_bd_intf_pins dma_0/M_AXIS_MM2S_STS]
  connect_bd_intf_net [get_bd_intf_pins s_axis_s2mm_cmd] [get_bd_intf_pins dma_0/S_AXIS_S2MM_CMD]
  connect_bd_intf_net [get_bd_intf_pins m_axis_s2mm_sts] [get_bd_intf_pins dma_0/M_AXIS_S2MM_STS]
  connect_bd_intf_net [get_bd_intf_pins s_axis_s2mm] [get_bd_intf_pins dma_0/S_AXIS_S2MM]
  connect_bd_intf_net [get_bd_intf_pins m_axis_mm2s] [get_bd_intf_pins dma_0/M_AXIS_MM2S]
}

# address segments for DMAs
for {set i 0} {$i < $num_dma} {incr i} {
  assign_bd_address -offset 0x00000000 -range 0x00010000000000000000 -target_address_space [get_bd_addr_spaces dma_${i}/Data] [get_bd_addr_segs m_axi_${i}/Reg] -force
}

set_property -dict [ list CONFIG.ASSOCIATED_BUSIF $interfaces ] [get_bd_ports ap_clk]

validate_bd_design
save_bd_design

add_files -norecurse ./external_dma.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

set bdfile [get_files ./external_dma/external_dma.srcs/sources_1/bd/external_dma_bd/external_dma_bd.bd]
generate_target all $bdfile
export_ip_user_files -of_objects $bdfile -no_script -sync -force -quiet
create_ip_run $bdfile
update_compile_order -fileset sources_1
set_property top external_dma [current_fileset]

# Package IP

ipx::package_project -root_dir ./packaged_kernel -vendor Xilinx -library ACCL -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core ./packaged_kernel/component.xml

ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory ./package ./packaged_kernel/component.xml
set_property core_revision 1 [ipx::current_core]

foreach up [ipx::get_user_parameters] {
  ipx::remove_user_parameter [get_property NAME $up] [ipx::current_core]
}

set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]

ipx::add_bus_interface ap_clk [ipx::current_core]
set clocksig [ipx::get_bus_interfaces ap_clk -of_objects [ipx::current_core]]
set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $clocksig
set_property bus_type_vlnv xilinx.com:signal:clock:1.0 $clocksig
ipx::add_port_map CLK $clocksig
set_property physical_name ap_clk [ipx::get_port_maps CLK -of_objects $clocksig]

ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axis_s2mm -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axis_mm2s -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axis_s2mm_cmd -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axis_s2mm_sts -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axis_mm2s_cmd -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axis_mm2s_sts -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axi_0 -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axi_1 -clock ap_clk [ipx::current_core]

set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} [ipx::current_core]
set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]

## Generate XO
if {[file exists "external_dma.xo"]} {
    file delete -force "external_dma.xo"
}

package_xo -xo_path external_dma.xo -kernel_name external_dma -ip_directory ./packaged_kernel -kernel_xml ./kernel.xml

close_project -delete
