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

# Hierarchical cell: dma
proc create_hier_cell_dma { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_dma() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 dma_aximm

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma_mm2s

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 dma_mm2s_cmd

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma_mm2s_sts

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 dma_s2mm

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 dma_s2mm_cmd

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma_s2mm_sts


  # Create pins
  create_bd_pin -dir I -type clk ap_clk
  create_bd_pin -dir I -type rst ap_rst_n

  # Create instance: axi_datamover_0, and set properties
  set axi_datamover_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_datamover:5.1 axi_datamover_0 ]
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
 ] $axi_datamover_0

  # Create instance: axis_dwidth_cnv_0, and set properties
  set axis_dwidth_cnv_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 axis_dwidth_cnv_0 ]
  set_property -dict [ list \
   CONFIG.HAS_MI_TKEEP {0} \
   CONFIG.HAS_TLAST {0} \
   CONFIG.M_TDATA_NUM_BYTES {16} \
   CONFIG.S_TDATA_NUM_BYTES {4} \
 ] $axis_dwidth_cnv_0

  # Create instance: axis_dwidth_cnv_1, and set properties
  set axis_dwidth_cnv_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 axis_dwidth_cnv_1 ]
  set_property -dict [ list \
   CONFIG.HAS_MI_TKEEP {0} \
   CONFIG.HAS_TLAST {0} \
   CONFIG.M_TDATA_NUM_BYTES {16} \
   CONFIG.S_TDATA_NUM_BYTES {4} \
 ] $axis_dwidth_cnv_1

  # Create instance: axis_subset_cnv_cmd_0, and set properties
  set axis_subset_cnv_cmd_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 axis_subset_cnv_cmd_0 ]
  set_property -dict [ list \
   CONFIG.M_TDATA_NUM_BYTES {13} \
   CONFIG.S_TDATA_NUM_BYTES {16} \
   CONFIG.TDATA_REMAP {tdata[103:0]} \
 ] $axis_subset_cnv_cmd_0

  # Create instance: axis_subset_cnv_cmd_1, and set properties
  set axis_subset_cnv_cmd_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 axis_subset_cnv_cmd_1 ]
  set_property -dict [ list \
   CONFIG.M_TDATA_NUM_BYTES {13} \
   CONFIG.S_TDATA_NUM_BYTES {16} \
   CONFIG.TDATA_REMAP {tdata[103:0]} \
 ] $axis_subset_cnv_cmd_1

  # Create instance: axis_subset_cnv_sts_0, and set properties
  set axis_subset_cnv_sts_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 axis_subset_cnv_sts_0 ]
  set_property -dict [ list \
   CONFIG.M_HAS_TKEEP {0} \
   CONFIG.M_TDATA_NUM_BYTES {4} \
   CONFIG.S_TDATA_NUM_BYTES {1} \
   CONFIG.TDATA_REMAP {24'b000000000000000000000000,tdata[7:0]} \
 ] $axis_subset_cnv_sts_0

  # Create instance: axis_subset_cnv_sts_1, and set properties
  set axis_subset_cnv_sts_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 axis_subset_cnv_sts_1 ]
  set_property -dict [ list \
   CONFIG.M_HAS_TKEEP {0} \
   CONFIG.M_TDATA_NUM_BYTES {4} \
   CONFIG.S_TDATA_NUM_BYTES {4} \
   CONFIG.TDATA_REMAP {tdata[31:0]} \
 ] $axis_subset_cnv_sts_1

  # Create interface connections
  connect_bd_intf_net -intf_net S_AXIS_1 [get_bd_intf_pins dma_s2mm_cmd] [get_bd_intf_pins axis_dwidth_cnv_1/S_AXIS]
  connect_bd_intf_net -intf_net axi_datamover_0_0_M_AXIS_MM2S [get_bd_intf_pins dma_mm2s] [get_bd_intf_pins axi_datamover_0/M_AXIS_MM2S]
  connect_bd_intf_net -intf_net axi_datamover_0_M_AXI [get_bd_intf_pins dma_aximm] [get_bd_intf_pins axi_datamover_0/M_AXI]
  connect_bd_intf_net -intf_net axi_datamover_0_M_AXIS_MM2S_STS [get_bd_intf_pins axi_datamover_0/M_AXIS_MM2S_STS] [get_bd_intf_pins axis_subset_cnv_sts_0/S_AXIS]
  connect_bd_intf_net -intf_net axi_datamover_0_M_AXIS_S2MM_STS [get_bd_intf_pins axi_datamover_0/M_AXIS_S2MM_STS] [get_bd_intf_pins axis_subset_cnv_sts_1/S_AXIS]
  connect_bd_intf_net -intf_net axis_dwidth_cnv_0_M_AXIS [get_bd_intf_pins axis_dwidth_cnv_0/M_AXIS] [get_bd_intf_pins axis_subset_cnv_cmd_0/S_AXIS]
  connect_bd_intf_net -intf_net axis_dwidth_cnv_0_S_AXIS [get_bd_intf_pins dma_mm2s_cmd] [get_bd_intf_pins axis_dwidth_cnv_0/S_AXIS]
  connect_bd_intf_net -intf_net axis_dwidth_cnv_1_M_AXIS [get_bd_intf_pins axis_dwidth_cnv_1/M_AXIS] [get_bd_intf_pins axis_subset_cnv_cmd_1/S_AXIS]
  connect_bd_intf_net -intf_net axis_subset_cnv_0_M_AXIS [get_bd_intf_pins dma_mm2s_sts] [get_bd_intf_pins axis_subset_cnv_sts_0/M_AXIS]
  connect_bd_intf_net -intf_net axis_subset_cnv_cmd_0_M_AXIS [get_bd_intf_pins axi_datamover_0/S_AXIS_MM2S_CMD] [get_bd_intf_pins axis_subset_cnv_cmd_0/M_AXIS]
  connect_bd_intf_net -intf_net axis_subset_cnv_cmd_1_M_AXIS [get_bd_intf_pins axi_datamover_0/S_AXIS_S2MM_CMD] [get_bd_intf_pins axis_subset_cnv_cmd_1/M_AXIS]
  connect_bd_intf_net -intf_net axis_subset_cnv_sts_1_M_AXIS [get_bd_intf_pins dma_s2mm_sts] [get_bd_intf_pins axis_subset_cnv_sts_1/M_AXIS]
  connect_bd_intf_net -intf_net s00_axis [get_bd_intf_pins dma_s2mm] [get_bd_intf_pins axi_datamover_0/S_AXIS_S2MM]

  # Create port connections
  connect_bd_net -net ap_clk [get_bd_pins ap_clk] [get_bd_pins axi_datamover_0/m_axi_mm2s_aclk] [get_bd_pins axi_datamover_0/m_axi_s2mm_aclk] [get_bd_pins axi_datamover_0/m_axis_mm2s_cmdsts_aclk] [get_bd_pins axi_datamover_0/m_axis_s2mm_cmdsts_awclk] [get_bd_pins axis_dwidth_cnv_0/aclk] [get_bd_pins axis_dwidth_cnv_1/aclk] [get_bd_pins axis_subset_cnv_cmd_0/aclk] [get_bd_pins axis_subset_cnv_cmd_1/aclk] [get_bd_pins axis_subset_cnv_sts_0/aclk] [get_bd_pins axis_subset_cnv_sts_1/aclk]
  connect_bd_net -net ap_rst_n [get_bd_pins ap_rst_n] [get_bd_pins axi_datamover_0/m_axi_mm2s_aresetn] [get_bd_pins axi_datamover_0/m_axi_s2mm_aresetn] [get_bd_pins axi_datamover_0/m_axis_mm2s_cmdsts_aresetn] [get_bd_pins axi_datamover_0/m_axis_s2mm_cmdsts_aresetn] [get_bd_pins axis_dwidth_cnv_0/aresetn] [get_bd_pins axis_dwidth_cnv_1/aresetn] [get_bd_pins axis_subset_cnv_cmd_0/aresetn] [get_bd_pins axis_subset_cnv_cmd_1/aresetn] [get_bd_pins axis_subset_cnv_sts_0/aresetn] [get_bd_pins axis_subset_cnv_sts_1/aresetn]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Create instances
create_hier_cell_dma [current_bd_instance .] dma_0
create_hier_cell_dma [current_bd_instance .] dma_1
create_hier_cell_dma [current_bd_instance .] dma_2
