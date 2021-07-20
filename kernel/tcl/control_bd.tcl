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

# Hierarchical cell: microblaze_0_local_memory
proc create_hier_cell_microblaze_0_local_memory { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_microblaze_0_local_memory() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode MirroredMaster -vlnv xilinx.com:interface:lmb_rtl:1.0 DLMB

  create_bd_intf_pin -mode MirroredMaster -vlnv xilinx.com:interface:lmb_rtl:1.0 ILMB


  # Create pins
  create_bd_pin -dir I -type clk LMB_Clk
  create_bd_pin -dir I -type rst SYS_Rst

  # Create instance: dlmb_bram_if_cntlr, and set properties
  set dlmb_bram_if_cntlr [ create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_bram_if_cntlr:4.0 dlmb_bram_if_cntlr ]
  set_property -dict [ list \
   CONFIG.C_ECC {0} \
 ] $dlmb_bram_if_cntlr

  # Create instance: dlmb_bram_if_cntlr_bram, and set properties
  set dlmb_bram_if_cntlr_bram [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 dlmb_bram_if_cntlr_bram ]
  set_property -dict [ list \
   CONFIG.Memory_Type {True_Dual_Port_RAM} \
 ] $dlmb_bram_if_cntlr_bram

  # Create instance: dlmb_v10, and set properties
  set dlmb_v10 [ create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_v10:3.0 dlmb_v10 ]
  set_property -dict [ list \
   CONFIG.C_LMB_NUM_SLAVES {2} \
 ] $dlmb_v10

  # Create instance: ilmb_bram_if_cntlr, and set properties
  set ilmb_bram_if_cntlr [ create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_bram_if_cntlr:4.0 ilmb_bram_if_cntlr ]
  set_property -dict [ list \
   CONFIG.C_ECC {0} \
 ] $ilmb_bram_if_cntlr

  # Create instance: ilmb_v10, and set properties
  set ilmb_v10 [ create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_v10:3.0 ilmb_v10 ]

  # Create interface connections
  connect_bd_intf_net -intf_net dlmb_bram_if_cntlr_BRAM_PORT [get_bd_intf_pins dlmb_bram_if_cntlr/BRAM_PORT] [get_bd_intf_pins dlmb_bram_if_cntlr_bram/BRAM_PORTA]
  connect_bd_intf_net -intf_net ilmb_bram_if_cntlr_BRAM_PORT [get_bd_intf_pins dlmb_bram_if_cntlr_bram/BRAM_PORTB] [get_bd_intf_pins ilmb_bram_if_cntlr/BRAM_PORT]
  connect_bd_intf_net -intf_net microblaze_0_dlmb [get_bd_intf_pins DLMB] [get_bd_intf_pins dlmb_v10/LMB_M]
  connect_bd_intf_net -intf_net microblaze_0_dlmb_bus [get_bd_intf_pins dlmb_bram_if_cntlr/SLMB] [get_bd_intf_pins dlmb_v10/LMB_Sl_0]
  connect_bd_intf_net -intf_net microblaze_0_ilmb [get_bd_intf_pins ILMB] [get_bd_intf_pins ilmb_v10/LMB_M]
  connect_bd_intf_net -intf_net microblaze_0_ilmb_bus [get_bd_intf_pins ilmb_bram_if_cntlr/SLMB] [get_bd_intf_pins ilmb_v10/LMB_Sl_0]

  # Create port connections
  connect_bd_net -net microblaze_0_Clk [get_bd_pins LMB_Clk] [get_bd_pins dlmb_bram_if_cntlr/LMB_Clk] [get_bd_pins dlmb_v10/LMB_Clk] [get_bd_pins ilmb_bram_if_cntlr/LMB_Clk] [get_bd_pins ilmb_v10/LMB_Clk]
  connect_bd_net -net microblaze_0_LMB_Rst [get_bd_pins SYS_Rst] [get_bd_pins dlmb_bram_if_cntlr/LMB_Rst] [get_bd_pins dlmb_v10/SYS_Rst] [get_bd_pins ilmb_bram_if_cntlr/LMB_Rst] [get_bd_pins ilmb_v10/SYS_Rst]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: microblaze_0_exchange_memory
proc create_hier_cell_microblaze_0_exchange_memory { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_microblaze_0_exchange_memory() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S00_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 host_cmd

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 host_sts


  # Create pins
  create_bd_pin -dir I -type rst ap_rst_n
  create_bd_pin -dir O -from 0 -to 0 encore_aresetn
  create_bd_pin -dir I -type clk s_axi_aclk

  # Create instance: axi_bram_ctrl_0, and set properties
  set axi_bram_ctrl_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0 ]
  set_property -dict [ list \
   CONFIG.ECC_TYPE {0} \
   CONFIG.PROTOCOL {AXI4LITE} \
   CONFIG.SINGLE_PORT_BRAM {1} \
   CONFIG.SUPPORTS_NARROW_BURST {0} \
 ] $axi_bram_ctrl_0

  # Create instance: axi_bram_ctrl_0_bram, and set properties
  set axi_bram_ctrl_0_bram [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 axi_bram_ctrl_0_bram ]

  # Create instance: axi_crossbar_0, and set properties
  set axi_crossbar_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_crossbar:2.1 axi_crossbar_0 ]
  set_property -dict [ list \
   CONFIG.NUM_MI {2} \
   CONFIG.NUM_SI {2} \
   CONFIG.R_REGISTER {1} \
 ] $axi_crossbar_0

  # Create instance: axi_crossbar_1, and set properties
  set axi_crossbar_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_crossbar:2.1 axi_crossbar_1 ]
  set_property -dict [ list \
   CONFIG.NUM_SI {1} \
 ] $axi_crossbar_1

  # Create instance: axi_gpio_0, and set properties
  set axi_gpio_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0 ]
  set_property -dict [ list \
   CONFIG.C_IS_DUAL {1} \
   CONFIG.C_ALL_INPUTS {0} \
   CONFIG.C_ALL_OUTPUTS {1} \
   CONFIG.C_ALL_INPUTS_2 {1} \
   CONFIG.C_GPIO2_WIDTH {32} \
   CONFIG.C_GPIO_WIDTH {2} \
   CONFIG.C_INTERRUPT_PRESENT {0} \
 ] $axi_gpio_0

  # Create instance: xlconstant_hwid, and set properties
  set xlconstant_hwid [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_hwid ]
  set_property -dict [ list \
   CONFIG.CONST_WIDTH {32} \
   CONFIG.CONST_VAL {0xcafebabe} \
 ] $xlconstant_hwid

  # Create instance: axi_register_slice_0, and set properties
  set axi_register_slice_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_0 ]
  set_property -dict [ list \
   CONFIG.REG_B {7} \
   CONFIG.REG_R {7} \
   CONFIG.REG_W {7} \
 ] $axi_register_slice_0

  # Create instance: hostctrl, and set properties
  set hostctrl [ create_bd_cell -type ip -vlnv xilinx.com:hls:hostctrl:1.0 hostctrl ]

  # Create instance: xlslice_encore_rstn, and set properties
  set xlslice_encore_rstn [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_encore_rstn ]
  set_property -dict [ list \
   CONFIG.DIN_FROM {1} \
   CONFIG.DIN_TO {1} \
   CONFIG.DIN_WIDTH {3} \
   CONFIG.DOUT_WIDTH {1} \
 ] $xlslice_encore_rstn

  # Create instance: xlslice_init_done, and set properties
  set xlslice_init_done [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_init_done ]
  set_property -dict [ list \
   CONFIG.DIN_FROM {0} \
   CONFIG.DIN_TO {0} \
   CONFIG.DIN_WIDTH {3} \
 ] $xlslice_init_done

  # Create interface connections
  connect_bd_intf_net -intf_net S00_AXI_1 [get_bd_intf_pins S00_AXI] [get_bd_intf_pins axi_crossbar_1/S00_AXI]
  connect_bd_intf_net -intf_net S_AXI_1 [get_bd_intf_pins S_AXI] [get_bd_intf_pins axi_register_slice_0/S_AXI]
  connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTA [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins axi_bram_ctrl_0_bram/BRAM_PORTA]
  connect_bd_intf_net -intf_net axi_crossbar_0_M00_AXI [get_bd_intf_pins axi_bram_ctrl_0/S_AXI] [get_bd_intf_pins axi_crossbar_0/M00_AXI]
  connect_bd_intf_net -intf_net axi_crossbar_0_M01_AXI [get_bd_intf_pins axi_crossbar_0/M01_AXI] [get_bd_intf_pins hostctrl/s_axi_control]
  connect_bd_intf_net -intf_net axi_crossbar_1_M00_AXI [get_bd_intf_pins axi_crossbar_0/S00_AXI] [get_bd_intf_pins axi_crossbar_1/M00_AXI]
  connect_bd_intf_net -intf_net axi_crossbar_1_M01_AXI [get_bd_intf_pins axi_crossbar_1/M01_AXI] [get_bd_intf_pins axi_gpio_0/S_AXI]
  connect_bd_intf_net -intf_net axi_register_slice_0_M_AXI [get_bd_intf_pins axi_crossbar_0/S01_AXI] [get_bd_intf_pins axi_register_slice_0/M_AXI]
  connect_bd_intf_net -intf_net host_cmd [get_bd_intf_pins host_cmd] [get_bd_intf_pins hostctrl/cmd_V]
  connect_bd_intf_net -intf_net host_sts [get_bd_intf_pins host_sts] [get_bd_intf_pins hostctrl/sts_V]

  # Create port connections
  connect_bd_net -net ap_rst_n_1 [get_bd_pins ap_rst_n] [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn] [get_bd_pins axi_crossbar_0/aresetn] [get_bd_pins axi_crossbar_1/aresetn] [get_bd_pins axi_gpio_0/s_axi_aresetn] [get_bd_pins hostctrl/ap_rst_n]
  connect_bd_net -net axi_gpio_0_gpio_io_o [get_bd_pins axi_gpio_0/gpio_io_o] [get_bd_pins xlslice_encore_rstn/Din] [get_bd_pins xlslice_init_done/Din]
  connect_bd_net -net s_axi_aclk_1 [get_bd_pins s_axi_aclk] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk] [get_bd_pins axi_crossbar_0/aclk] [get_bd_pins axi_crossbar_1/aclk] [get_bd_pins axi_gpio_0/s_axi_aclk] [get_bd_pins axi_register_slice_0/aclk] [get_bd_pins hostctrl/ap_clk]
  connect_bd_net -net xlslice_encore_rstn_Dout [get_bd_pins encore_aresetn] [get_bd_pins xlslice_encore_rstn/Dout]
  connect_bd_net -net xlslice_init_done [get_bd_pins axi_register_slice_0/aresetn] [get_bd_pins xlslice_init_done/Dout]
  connect_bd_net -net hwid [get_bd_pins xlconstant_hwid/dout] [get_bd_pins axi_gpio_0/gpio2_io_i]

  # Restore current instance
  current_bd_instance $oldCurInst
}

proc create_dma_infrastructure { dmaIndex mbStartIndex } {
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma${dmaIndex}_mm2s_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 dma${dmaIndex}_mm2s_sts

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma${dmaIndex}_s2mm_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 dma${dmaIndex}_s2mm_sts

  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma${dmaIndex}_mm2s_cmd
  set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {4}] [get_bd_cells fifo_dma${dmaIndex}_mm2s_cmd]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma${dmaIndex}_mm2s_sts
  set_property -dict [ list CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {4}] [get_bd_cells fifo_dma${dmaIndex}_mm2s_sts]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma${dmaIndex}_s2mm_cmd
  set_property -dict [ list CONFIG.HAS_WR_DATA_COUNT {1} CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {4}] [get_bd_cells fifo_dma${dmaIndex}_s2mm_cmd]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma${dmaIndex}_s2mm_sts
  set_property -dict [ list CONFIG.HAS_RD_DATA_COUNT {1} CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {4}] [get_bd_cells fifo_dma${dmaIndex}_s2mm_sts]

  connect_bd_intf_net [get_bd_intf_pins dma${dmaIndex}_mm2s_cmd] [get_bd_intf_pins fifo_dma${dmaIndex}_mm2s_cmd/M_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dma${dmaIndex}_mm2s_sts] [get_bd_intf_pins fifo_dma${dmaIndex}_mm2s_sts/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dma${dmaIndex}_s2mm_cmd] [get_bd_intf_pins fifo_dma${dmaIndex}_s2mm_cmd/M_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dma${dmaIndex}_s2mm_sts] [get_bd_intf_pins fifo_dma${dmaIndex}_s2mm_sts/S_AXIS]

  connect_bd_intf_net [get_bd_intf_pins fifo_dma${dmaIndex}_mm2s_sts/M_AXIS] [get_bd_intf_pins microblaze_0/S${mbStartIndex}_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma${dmaIndex}_s2mm_cmd/S_AXIS] [get_bd_intf_pins microblaze_0/M${mbStartIndex}_AXIS]
  incr mbStartIndex
  connect_bd_intf_net [get_bd_intf_pins fifo_dma${dmaIndex}_s2mm_sts/M_AXIS] [get_bd_intf_pins microblaze_0/S${mbStartIndex}_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma${dmaIndex}_mm2s_cmd/S_AXIS] [get_bd_intf_pins microblaze_0/M${mbStartIndex}_AXIS]

  connect_bd_net [get_bd_pins proc_sys_reset_1/peripheral_aresetn]  [get_bd_pins fifo_dma${dmaIndex}_mm2s_sts/s_axis_aresetn] \
                                                                    [get_bd_pins fifo_dma${dmaIndex}_mm2s_cmd/s_axis_aresetn] \
                                                                    [get_bd_pins fifo_dma${dmaIndex}_s2mm_sts/s_axis_aresetn] \
                                                                    [get_bd_pins fifo_dma${dmaIndex}_s2mm_cmd/s_axis_aresetn] 
  connect_bd_net [get_bd_pins ap_clk] [get_bd_pins fifo_dma${dmaIndex}_mm2s_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_dma${dmaIndex}_mm2s_cmd/s_axis_aclk] \
                                      [get_bd_pins fifo_dma${dmaIndex}_s2mm_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_dma${dmaIndex}_s2mm_cmd/s_axis_aclk]
}

# Hierarchical cell: control
proc create_hier_cell_control { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_control() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:mbtrace_rtl:2.0 TRACE
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:bscan_rtl:1.0 bscan_0

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 encore_control
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 host_control

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 udp_packetizer_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 udp_depacketizer_sts

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 tcp_packetizer_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 tcp_depacketizer_sts

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 tcp_opencon_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 tcp_opencon_sts

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 tcp_openport_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 tcp_openport_sts

  # Create pins
  create_bd_pin -dir I -type clk ap_clk
  create_bd_pin -dir I -type rst ap_rst_n
  create_bd_pin -dir O -from 0 -to 0 -type rst encore_aresetn

  # Create instance: compute_rx_udp_cmd_nonzero, and set properties
  set compute_rx_udp_cmd_nonzero [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_reduced_logic:2.0 compute_rx_udp_cmd_nonzero ]
  set_property -dict [ list \
   CONFIG.C_OPERATION {or} \
   CONFIG.C_SIZE {32} \
   CONFIG.LOGO_FILE {data/sym_orgate.png} \
 ] $compute_rx_udp_cmd_nonzero

  # Create instance: compute_rx_udp_cmd_zero, and set properties
  set compute_rx_udp_cmd_zero [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 compute_rx_udp_cmd_zero ]
  set_property -dict [ list \
   CONFIG.C_OPERATION {not} \
   CONFIG.C_SIZE {1} \
   CONFIG.LOGO_FILE {data/sym_notgate.png} \
 ] $compute_rx_udp_cmd_zero

  # Create instance: compute_rx_udp_sts_nonzero, and set properties
  set compute_rx_udp_sts_nonzero [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_reduced_logic:2.0 compute_rx_udp_sts_nonzero ]
  set_property -dict [ list \
   CONFIG.C_OPERATION {or} \
   CONFIG.C_SIZE {32} \
   CONFIG.LOGO_FILE {data/sym_orgate.png} \
 ] $compute_rx_udp_sts_nonzero


  # Create instance: compute_rx_tcp_cmd_nonzero, and set properties
  set compute_rx_tcp_cmd_nonzero [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_reduced_logic:2.0 compute_rx_tcp_cmd_nonzero ]
  set_property -dict [ list \
   CONFIG.C_OPERATION {or} \
   CONFIG.C_SIZE {32} \
   CONFIG.LOGO_FILE {data/sym_orgate.png} \
 ] $compute_rx_tcp_cmd_nonzero

  # Create instance: compute_rx_tcp_cmd_zero, and set properties
  set compute_rx_tcp_cmd_zero [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 compute_rx_tcp_cmd_zero ]
  set_property -dict [ list \
   CONFIG.C_OPERATION {not} \
   CONFIG.C_SIZE {1} \
   CONFIG.LOGO_FILE {data/sym_notgate.png} \
 ] $compute_rx_tcp_cmd_zero

  # Create instance: compute_rx_tcp_sts_nonzero, and set properties
  set compute_rx_tcp_sts_nonzero [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_reduced_logic:2.0 compute_rx_tcp_sts_nonzero ]
  set_property -dict [ list \
   CONFIG.C_OPERATION {or} \
   CONFIG.C_SIZE {32} \
   CONFIG.LOGO_FILE {data/sym_orgate.png} \
 ] $compute_rx_tcp_sts_nonzero

  # Create instance: fifo_depacketizer_sts, and set properties
  set fifo_depacketizer_sts [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_depacketizer_sts ]
  set_property -dict [ list \
   CONFIG.HAS_TLAST {1} \
   CONFIG.TDATA_NUM_BYTES {4} \
 ] $fifo_depacketizer_sts

  # Create instance: fifo_depacketizer_sts, and set properties
  set fifo_tcp_depacketizer_sts [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_tcp_depacketizer_sts ]
  set_property -dict [ list \
   CONFIG.HAS_TLAST {1} \
   CONFIG.TDATA_NUM_BYTES {4} \
 ] $fifo_tcp_depacketizer_sts

  # Create instance: fifo_openCon_cmd, and set properties
  set fifo_openCon_cmd [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_openCon_cmd ]
  set_property -dict [ list \
   CONFIG.HAS_TLAST {1} \
   CONFIG.TDATA_NUM_BYTES {4} \
 ] $fifo_openCon_cmd

  # Create instance: fifo_openCon_sts, and set properties
  set fifo_openCon_sts [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_openCon_sts ]
  set_property -dict [ list \
   CONFIG.HAS_TLAST {1} \
   CONFIG.TDATA_NUM_BYTES {4} \
 ] $fifo_openCon_sts

  # Create instance: fifo_openPort_cmd, and set properties
  set fifo_openPort_cmd [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_openPort_cmd ]
  set_property -dict [ list \
   CONFIG.HAS_TLAST {1} \
   CONFIG.TDATA_NUM_BYTES {4} \
 ] $fifo_openPort_cmd

  # Create instance: fifo_openPort_sts, and set properties
  set fifo_openPort_sts [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_openPort_sts ]
  set_property -dict [ list \
   CONFIG.HAS_TLAST {1} \
   CONFIG.TDATA_NUM_BYTES {4} \
 ] $fifo_openPort_sts

  # Create instance: fifo_packetizer_cmd, and set properties
  set fifo_packetizer_cmd [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_packetizer_cmd ]

  # Create instance: fifo_packetizer_cmd, and set properties
  set fifo_tcp_packetizer_cmd [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_tcp_packetizer_cmd ]

  # Create instance: mdm_1, and set properties
  set mdm_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:mdm:3.2 mdm_1 ]
  set_property -dict [ list \
   CONFIG.C_USE_BSCAN {2} \
 ] $mdm_1

  # Create instance: microblaze_0, and set properties
  set microblaze_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:microblaze:11.0 microblaze_0 ]
  set_property -dict [ list \
   CONFIG.C_BASE_VECTORS {0x0000000000010000} \
   CONFIG.C_D_AXI {1} \
   CONFIG.C_D_LMB {1} \
   CONFIG.C_FSL_LINKS {11} \
   CONFIG.C_I_LMB {1} \
   CONFIG.C_TRACE {1} \
   CONFIG.C_USE_EXTENDED_FSL_INSTR {1} \
   CONFIG.C_USE_MSR_INSTR {1} \
   CONFIG.C_USE_PCMP_INSTR {1} \
   CONFIG.C_DEBUG_EXTERNAL_TRACE.VALUE_SRC PROPAGATED \
   CONFIG.C_DEBUG_ENABLED {2} CONFIG.C_DEBUG_PROFILE_SIZE {16384} \
 ] $microblaze_0

  # Create instance: microblaze_0_axi_periph, and set properties
  set microblaze_0_axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 microblaze_0_axi_periph ]
  set_property -dict [ list \
   CONFIG.NUM_MI {4} \
 ] $microblaze_0_axi_periph

  # Create instance: microblaze_0_exchange_memory
  create_hier_cell_microblaze_0_exchange_memory $hier_obj microblaze_0_exchange_memory

  # Create instance: microblaze_0_local_memory
  create_hier_cell_microblaze_0_local_memory $hier_obj microblaze_0_local_memory

  # Create instance: proc_sys_reset_0, and set properties
  set proc_sys_reset_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0 ]

  # Create instance: proc_sys_reset_1, and set properties
  set proc_sys_reset_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_1 ]

  # Create DMAs
  create_dma_infrastructure 0 0
  create_dma_infrastructure 1 2
  create_dma_infrastructure 2 4
  # Create instance: proc_irq_control, and set properties
  set proc_irq_control [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 proc_irq_control]
  set_property -dict [list \
      CONFIG.C_KIND_OF_INTR.VALUE_SRC USER \
      CONFIG.C_KIND_OF_INTR {0xFFFFFFE0} \
  ] $proc_irq_control
  
  # Create instance: proc_irq_concat and set properties
  set proc_irq_concat [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 proc_irq_concat]
  set_property -dict [list \
   CONFIG.NUM_PORTS {5} \
  ] $proc_irq_concat

  # Create instance: axi_timer and set properties
  set axi_timer [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_timer:2.0 axi_timer ]
  
  # Create interface connections
  connect_bd_intf_net -intf_net bscan_0 [get_bd_intf_pins bscan_0] [get_bd_intf_pins mdm_1/BSCAN]
  connect_bd_intf_net -intf_net microblaze_0_TRACE [get_bd_intf_pins TRACE] [get_bd_intf_pins microblaze_0/TRACE]
  connect_bd_intf_net -intf_net microblaze_0_debug [get_bd_intf_pins mdm_1/MBDEBUG_0] [get_bd_intf_pins microblaze_0/DEBUG]

  connect_bd_intf_net -intf_net microblaze_0_dlmb_1 [get_bd_intf_pins microblaze_0/DLMB] [get_bd_intf_pins microblaze_0_local_memory/DLMB]
  connect_bd_intf_net -intf_net microblaze_0_ilmb_1 [get_bd_intf_pins microblaze_0/ILMB] [get_bd_intf_pins microblaze_0_local_memory/ILMB]

  connect_bd_intf_net -intf_net s_axi_control [get_bd_intf_pins host_control] [get_bd_intf_pins microblaze_0_exchange_memory/S_AXI]

  connect_bd_intf_net -intf_net c0 [get_bd_intf_pins microblaze_0_exchange_memory/S00_AXI] [get_bd_intf_pins microblaze_0_axi_periph/M00_AXI]
  connect_bd_intf_net -intf_net c1 [get_bd_intf_pins encore_control] [get_bd_intf_pins microblaze_0_axi_periph/M01_AXI]

  connect_bd_intf_net -intf_net udp_depacketizer_sts [get_bd_intf_pins udp_depacketizer_sts] [get_bd_intf_pins fifo_depacketizer_sts/S_AXIS]  
  connect_bd_intf_net -intf_net udp_packetizer_cmd [get_bd_intf_pins udp_packetizer_cmd] [get_bd_intf_pins fifo_packetizer_cmd/M_AXIS]

  connect_bd_intf_net -intf_net tcp_openport_cmd [get_bd_intf_pins fifo_openPort_cmd/M_AXIS] [get_bd_intf_pins tcp_openport_cmd]
  connect_bd_intf_net -intf_net tcp_opencon_cmd [get_bd_intf_pins fifo_openCon_cmd/M_AXIS] [get_bd_intf_pins tcp_opencon_cmd]
  connect_bd_intf_net -intf_net tcp_packetizer_cmd [get_bd_intf_pins fifo_tcp_packetizer_cmd/M_AXIS] [get_bd_intf_pins tcp_packetizer_cmd]
  connect_bd_intf_net -intf_net tcp_openport_sts [get_bd_intf_pins fifo_openPort_sts/S_AXIS] [get_bd_intf_pins tcp_openport_sts]
  connect_bd_intf_net -intf_net tcp_opencon_sts [get_bd_intf_pins fifo_openCon_sts/S_AXIS] [get_bd_intf_pins tcp_opencon_sts]
  connect_bd_intf_net -intf_net tcp_depacketizer_sts [get_bd_intf_pins fifo_tcp_depacketizer_sts/S_AXIS] [get_bd_intf_pins tcp_depacketizer_sts]

  connect_bd_intf_net -intf_net microblaze_0_M_AXI_DP [get_bd_intf_pins microblaze_0/M_AXI_DP] [get_bd_intf_pins microblaze_0_axi_periph/S00_AXI]

  connect_bd_intf_net [get_bd_intf_pins fifo_depacketizer_sts/M_AXIS] [get_bd_intf_pins microblaze_0/S6_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_packetizer_cmd/S_AXIS] [get_bd_intf_pins microblaze_0/M6_AXIS]

  connect_bd_intf_net [get_bd_intf_pins fifo_openPort_cmd/S_AXIS] [get_bd_intf_pins microblaze_0/M7_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_openCon_cmd/S_AXIS] [get_bd_intf_pins microblaze_0/M8_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_openPort_sts/M_AXIS] [get_bd_intf_pins microblaze_0/S7_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_openCon_sts/M_AXIS] [get_bd_intf_pins microblaze_0/S8_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_tcp_packetizer_cmd/S_AXIS] [get_bd_intf_pins microblaze_0/M9_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_tcp_depacketizer_sts/M_AXIS] [get_bd_intf_pins microblaze_0/S9_AXIS]

  connect_bd_intf_net [get_bd_intf_pins microblaze_0/S10_AXIS] [get_bd_intf_pins microblaze_0_exchange_memory/host_cmd]
  connect_bd_intf_net [get_bd_intf_pins microblaze_0/M10_AXIS] [get_bd_intf_pins microblaze_0_exchange_memory/host_sts]
  connect_bd_intf_net -intf_net microblaze_0_axi_periph_M05_AXI [get_bd_intf_pins axi_timer/S_AXI] [get_bd_intf_pins microblaze_0_axi_periph/M02_AXI]
  connect_bd_intf_net -intf_net microblaze_0_axi_periph_M06_AXI [get_bd_intf_pins proc_irq_control/s_axi] [get_bd_intf_pins microblaze_0_axi_periph/M03_AXI]
  connect_bd_intf_net -intf_net microblaze_0_irq [get_bd_intf_pins proc_irq_control/interrupt] [get_bd_intf_pins microblaze_0/INTERRUPT]
  # Clocks and resets
  connect_bd_net -net SYS_Rst_1 [get_bd_pins microblaze_0_local_memory/SYS_Rst] [get_bd_pins proc_sys_reset_0/peripheral_reset]
  connect_bd_net [get_bd_pins ap_clk] [get_bd_pins fifo_depacketizer_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_packetizer_cmd/s_axis_aclk] \
                                      [get_bd_pins fifo_openCon_cmd/s_axis_aclk] \
                                      [get_bd_pins fifo_openCon_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_openPort_cmd/s_axis_aclk] \
                                      [get_bd_pins fifo_openPort_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_tcp_depacketizer_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_tcp_packetizer_cmd/s_axis_aclk] \
                                      [get_bd_pins microblaze_0/Clk] \
                                      [get_bd_pins microblaze_0_axi_periph/ACLK] \
                                      [get_bd_pins microblaze_0_axi_periph/M00_ACLK] \
                                      [get_bd_pins microblaze_0_axi_periph/M01_ACLK] \
                                      [get_bd_pins microblaze_0_axi_periph/M02_ACLK] \
                                      [get_bd_pins microblaze_0_axi_periph/M03_ACLK] \
                                      [get_bd_pins microblaze_0_axi_periph/S00_ACLK] \
                                      [get_bd_pins microblaze_0_exchange_memory/s_axi_aclk] \
                                      [get_bd_pins microblaze_0_local_memory/LMB_Clk] \
                                      [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
                                      [get_bd_pins proc_sys_reset_1/slowest_sync_clk] \
                                      [get_bd_pins proc_irq_control/s_axi_aclk] \
                                      [get_bd_pins axi_timer/s_axi_aclk]
                                      
  connect_bd_net [get_bd_pins ap_rst_n] [get_bd_pins proc_sys_reset_0/ext_reset_in]
  connect_bd_net  [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins microblaze_0_axi_periph/ARESETN] \
                                                                    [get_bd_pins microblaze_0_axi_periph/M00_ARESETN] \
                                                                    [get_bd_pins microblaze_0_axi_periph/S00_ARESETN] \
                                                                    [get_bd_pins microblaze_0_exchange_memory/ap_rst_n] \
                                                                    [get_bd_pins proc_irq_control/s_axi_aresetn] \
                                                                    [get_bd_pins axi_timer/s_axi_aresetn] \
                                                                    [get_bd_pins microblaze_0_axi_periph/M02_ARESETN] \
                                                                    [get_bd_pins microblaze_0_axi_periph/M03_ARESETN] 
  connect_bd_net -net mdm_1_Debug_SYS_Rst [get_bd_pins mdm_1/Debug_SYS_Rst] [get_bd_pins proc_sys_reset_0/mb_debug_sys_rst] [get_bd_pins proc_sys_reset_1/mb_debug_sys_rst]
  connect_bd_net -net microblaze_0_exchange_memory_Dout [get_bd_pins microblaze_0_exchange_memory/encore_aresetn] [get_bd_pins proc_sys_reset_1/ext_reset_in]
  connect_bd_net [get_bd_pins microblaze_0/Reset] [get_bd_pins proc_sys_reset_0/mb_reset]
  connect_bd_net [get_bd_pins proc_sys_reset_1/peripheral_aresetn] [get_bd_pins encore_aresetn] \
                                                                   [get_bd_pins microblaze_0_axi_periph/M01_ARESETN] \
                                                                   [get_bd_pins fifo_depacketizer_sts/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_packetizer_cmd/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_tcp_depacketizer_sts/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_tcp_packetizer_cmd/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_openCon_cmd/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_openCon_sts/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_openPort_cmd/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_openPort_sts/s_axis_aresetn]

  # interrupt generation
  connect_bd_net -net rx_udp_cmd_count [get_bd_pins compute_rx_udp_cmd_nonzero/Op1] [get_bd_pins fifo_dma0_s2mm_cmd/axis_wr_data_count]
  connect_bd_net -net rx_udp_cmd_nonzero [get_bd_pins compute_rx_udp_cmd_nonzero/Res] [get_bd_pins compute_rx_udp_cmd_zero/Op1]
  connect_bd_net -net rx_udp_cmd_zero [get_bd_pins proc_irq_concat/In1] [get_bd_pins compute_rx_udp_cmd_zero/Res]
  connect_bd_net -net rx_udp_sts_count [get_bd_pins compute_rx_udp_sts_nonzero/Op1] [get_bd_pins fifo_dma0_s2mm_sts/axis_rd_data_count]
  connect_bd_net -net rx_udp_sts_nonzero [get_bd_pins proc_irq_concat/In2] [get_bd_pins compute_rx_udp_sts_nonzero/Res]

  connect_bd_net -net rx_tcp_cmd_count [get_bd_pins compute_rx_tcp_cmd_nonzero/Op1] [get_bd_pins fifo_dma2_s2mm_cmd/axis_wr_data_count]
  connect_bd_net -net rx_tcp_cmd_nonzero [get_bd_pins compute_rx_tcp_cmd_nonzero/Res] [get_bd_pins compute_rx_tcp_cmd_zero/Op1]
  connect_bd_net -net rx_tcp_cmd_zero [get_bd_pins proc_irq_concat/In3] [get_bd_pins compute_rx_tcp_cmd_zero/Res]
  connect_bd_net -net rx_tcp_sts_count [get_bd_pins compute_rx_tcp_sts_nonzero/Op1] [get_bd_pins fifo_dma2_s2mm_sts/axis_rd_data_count]
  connect_bd_net -net rx_tcp_sts_nonzero [get_bd_pins proc_irq_concat/In4] [get_bd_pins compute_rx_tcp_sts_nonzero/Res]

  connect_bd_net -net axi_timer_irq [get_bd_pins proc_irq_concat/In0] [get_bd_pins axi_timer/interrupt]
  connect_bd_net -net concat_irq [get_bd_pins proc_irq_concat/dout] [get_bd_pins proc_irq_control/intr]
  # Restore current instance
  current_bd_instance $oldCurInst
}

# Create instance: control
create_hier_cell_control [current_bd_instance .] control
