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

proc create_hier_cell_udp_tx_subsystem { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_tx_subsystem() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_command
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_data
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_data
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_control
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_sts
  # Create pins
  create_bd_pin -dir I -type clk ap_clk
  create_bd_pin -dir I -type rst ap_rst_n

  # Create FIFO instances, and set properties
  set tx_fifo [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 tx_fifo ]
  set_property -dict [ list \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.IS_ACLK_ASYNC {0} \
   CONFIG.TDATA_NUM_BYTES {64} \
 ] $tx_fifo

  # Create TX instances and set properties
  set vnx_packetizer_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:vnx_packetizer:1.0 vnx_packetizer_0 ]

  # Create interface connections
  connect_bd_intf_net -intf_net status [get_bd_intf_pins m_axis_sts] [get_bd_intf_pins vnx_packetizer_0/sts_V]
  connect_bd_intf_net -intf_net control [get_bd_intf_pins s_axi_control] [get_bd_intf_pins vnx_packetizer_0/s_axi_control]
  connect_bd_intf_net -intf_net command [get_bd_intf_pins s_axis_command] [get_bd_intf_pins vnx_packetizer_0/cmd_V]
  connect_bd_intf_net -intf_net pkt2fifo [get_bd_intf_pins m_axis_data] [get_bd_intf_pins tx_fifo/M_AXIS]
  connect_bd_intf_net -intf_net in2pkt [get_bd_intf_pins s_axis_data] [get_bd_intf_pins vnx_packetizer_0/in_r]
  connect_bd_intf_net -intf_net fifo2out [get_bd_intf_pins tx_fifo/S_AXIS] [get_bd_intf_pins vnx_packetizer_0/out_r]

  # Create port connections
  connect_bd_net -net ap_clk [get_bd_pins ap_clk]  [get_bd_pins tx_fifo/s_axis_aclk] \
                                                   [get_bd_pins vnx_packetizer_0/ap_clk]
  connect_bd_net -net ap_rst_n [get_bd_pins ap_rst_n] [get_bd_pins tx_fifo/s_axis_aresetn] \
                                                      [get_bd_pins vnx_packetizer_0/ap_rst_n]

  # Restore current instance
  current_bd_instance $oldCurInst
}

proc create_hier_cell_tcp_tx_subsystem { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_tx_subsystem() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_pktcmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_control
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_tcp_open_status
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_tcp_open_connection
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_tcp_tx_data
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_tcp_tx_data
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_tcp_tx_status
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_tcp_tx_meta
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_opencon_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_opencon_sts
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_tcp_packetizer_sts

  # Create pins
  create_bd_pin -dir I -type clk ap_clk
  create_bd_pin -dir I -type rst ap_rst_n

  # Create FIFO instances, and set properties
  set tx_fifo [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 tx_fifo ]
  set_property -dict [ list \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.IS_ACLK_ASYNC {0} \
   CONFIG.TDATA_NUM_BYTES {64} \
 ] $tx_fifo

  # Create TX instances and set properties
  set tcp_openConReq_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:tcp_openConReq:1.0 tcp_openConReq_0 ]
  set tcp_openConResp_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:tcp_openConResp:1.0 tcp_openConResp_0 ]
  set tcp_packetizer_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:tcp_packetizer:1.0 tcp_packetizer_0 ]
  set tcp_txHandler_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:tcp_txHandler:1.0 tcp_txHandler_0 ]

  # Create interface connections
  connect_bd_intf_net -intf_net control [get_bd_intf_pins s_axi_control] [get_bd_intf_pins tcp_packetizer_0/s_axi_control]
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins s_axis_tcp_open_status] [get_bd_intf_pins tcp_openConResp_0/s_axis_tcp_open_status]
  connect_bd_intf_net -intf_net Conn2 [get_bd_intf_pins m_axis_tcp_open_connection] [get_bd_intf_pins tcp_openConReq_0/m_axis_tcp_open_connection]
  connect_bd_intf_net -intf_net Conn3 [get_bd_intf_pins m_axis_tcp_tx_data] [get_bd_intf_pins tx_fifo/M_AXIS]
  connect_bd_intf_net -intf_net cmd_V1_1 [get_bd_intf_pins s_axis_opencon_cmd] [get_bd_intf_pins tcp_openConReq_0/cmd_V]
  connect_bd_intf_net -intf_net cmd_V_1 [get_bd_intf_pins s_axis_pktcmd] [get_bd_intf_pins tcp_packetizer_0/cmd_V]
  connect_bd_intf_net -intf_net in_r_1 [get_bd_intf_pins s_axis_tcp_tx_data] [get_bd_intf_pins tcp_packetizer_0/in_r]
  connect_bd_intf_net -intf_net status [get_bd_intf_pins m_axis_tcp_packetizer_sts] [get_bd_intf_pins tcp_packetizer_0/sts_V]
  connect_bd_intf_net -intf_net s_axis_tcp_tx_status_1 [get_bd_intf_pins s_axis_tcp_tx_status] [get_bd_intf_pins tcp_txHandler_0/s_axis_tcp_tx_status]
  connect_bd_intf_net -intf_net tcp_openConResp_0_sts_V [get_bd_intf_pins m_axis_opencon_sts] [get_bd_intf_pins tcp_openConResp_0/sts_V]
  connect_bd_intf_net -intf_net tcp_packetizer_0_cmd_txHandler_V [get_bd_intf_pins tcp_packetizer_0/cmd_txHandler_V] [get_bd_intf_pins tcp_txHandler_0/cmd_txHandler_V]
  connect_bd_intf_net -intf_net tcp_packetizer_0_out_r [get_bd_intf_pins tcp_packetizer_0/out_r] [get_bd_intf_pins tcp_txHandler_0/s_data_in]
  connect_bd_intf_net -intf_net tcp_txHandler_0_m_axis_tcp_tx_data [get_bd_intf_pins tcp_txHandler_0/m_axis_tcp_tx_data] [get_bd_intf_pins tx_fifo/S_AXIS]
  connect_bd_intf_net -intf_net tcp_txHandler_0_m_axis_tcp_tx_meta [get_bd_intf_pins m_axis_tcp_tx_meta] [get_bd_intf_pins tcp_txHandler_0/m_axis_tcp_tx_meta]

  # Create port connections
  connect_bd_net -net ap_clk [get_bd_pins ap_clk]  [get_bd_pins tcp_openConReq_0/ap_clk] \
                                                   [get_bd_pins tcp_openConResp_0/ap_clk] \
                                                   [get_bd_pins tcp_packetizer_0/ap_clk] \
                                                   [get_bd_pins tcp_txHandler_0/ap_clk] \
                                                   [get_bd_pins tx_fifo/s_axis_aclk]
  connect_bd_net -net ap_rst_n [get_bd_pins ap_rst_n] [get_bd_pins tcp_openConReq_0/ap_rst_n] \
                                                      [get_bd_pins tcp_openConResp_0/ap_rst_n] \
                                                      [get_bd_pins tcp_packetizer_0/ap_rst_n] \
                                                      [get_bd_pins tcp_txHandler_0/ap_rst_n] \
                                                      [get_bd_pins tx_fifo/s_axis_aresetn]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Create instance: tx_subsystem
create_hier_cell_udp_tx_subsystem [current_bd_instance .] udp_tx_subsystem
create_hier_cell_tcp_tx_subsystem [current_bd_instance .] tcp_tx_subsystem
