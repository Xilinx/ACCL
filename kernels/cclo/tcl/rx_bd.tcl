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

proc create_udp_rx_subsystem { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_rx_subsystem() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_data
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_data
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_control
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_status
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_notification

  # Create pins
  create_bd_pin -dir I -type clk ap_clk
  create_bd_pin -dir I -type rst ap_rst_n

  # Create rx_fifos, and set properties
  set rx_fifo [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 rx_fifo ]
  set_property -dict [ list \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.IS_ACLK_ASYNC {0} \
   CONFIG.TDATA_NUM_BYTES {64} \
 ] $rx_fifo

  set dpkt_fifo [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 dpkt_fifo ]
  set_property -dict [ list \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.IS_ACLK_ASYNC {0} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.FIFO_DEPTH {32} \
   CONFIG.FIFO_MEMORY_TYPE {distributed} \
 ] $dpkt_fifo

  # Create instance: udp_depacketizer_0, and set properties
  set udp_depacketizer_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:udp_depacketizer:1.0 udp_depacketizer_0 ]

  # Create interface connections
  connect_bd_intf_net [get_bd_intf_pins s_axi_control] [get_bd_intf_pins udp_depacketizer_0/s_axi_control]
  connect_bd_intf_net [get_bd_intf_pins s_axis_data] [get_bd_intf_pins rx_fifo/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins rx_fifo/M_AXIS] [get_bd_intf_pins udp_depacketizer_0/in_r]
  connect_bd_intf_net [get_bd_intf_pins udp_depacketizer_0/out_r] [get_bd_intf_pins dpkt_fifo/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dpkt_fifo/M_AXIS] [get_bd_intf_pins m_axis_data]
  connect_bd_intf_net [get_bd_intf_pins udp_depacketizer_0/sts] [get_bd_intf_pins m_axis_status] 
  connect_bd_intf_net [get_bd_intf_pins udp_depacketizer_0/notif_out] [get_bd_intf_pins m_axis_notification]

  # Create port connections
  connect_bd_net -net ap_clk [get_bd_pins ap_clk] [get_bd_pins rx_fifo/s_axis_aclk] [get_bd_pins dpkt_fifo/s_axis_aclk] [get_bd_pins udp_depacketizer_0/ap_clk]
  connect_bd_net -net ap_rst_n [get_bd_pins ap_rst_n] [get_bd_pins rx_fifo/s_axis_aresetn] [get_bd_pins dpkt_fifo/s_axis_aresetn] [get_bd_pins udp_depacketizer_0/ap_rst_n]

  # Restore current instance
  current_bd_instance $oldCurInst
}


proc create_tcp_rx_subsystem { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_rx_subsystem() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_control
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_pktsts
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_rx_data
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_rx_data
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_notification
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_notification
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_rx_meta
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_read_pkg

  # Create pins
  create_bd_pin -dir I -type clk ap_clk
  create_bd_pin -dir I -type rst ap_rst_n

  set rx_fifo [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 rx_fifo ]
  set_property -dict [ list \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.IS_ACLK_ASYNC {0} \
   CONFIG.TDATA_NUM_BYTES {64} \
 ] $rx_fifo

  set dpkt_fifo [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 dpkt_fifo]
  set_property -dict [ list \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.TDATA_NUM_BYTES {64} \
    CONFIG.FIFO_DEPTH {64} \
    CONFIG.FIFO_MEMORY_TYPE {distributed} \
  ] $dpkt_fifo
  
  # Create instances of TCP blocks
  set tcp_depacketizer_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:tcp_depacketizer:1.0 tcp_depacketizer_0 ]
  set tcp_rxHandler_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:tcp_rxHandler:1.0 tcp_rxHandler_0 ]

  # Create interface connections
  connect_bd_intf_net -intf_net control [get_bd_intf_pins s_axi_control] [get_bd_intf_pins tcp_depacketizer_0/s_axi_control]

  # various metadata
  connect_bd_intf_net [get_bd_intf_pins s_axis_notification] [get_bd_intf_pins tcp_rxHandler_0/s_axis_tcp_notification]
  connect_bd_intf_net [get_bd_intf_pins s_axis_rx_meta] [get_bd_intf_pins tcp_rxHandler_0/s_axis_tcp_rx_meta]
  connect_bd_intf_net [get_bd_intf_pins m_axis_pktsts] [get_bd_intf_pins tcp_depacketizer_0/sts]
  connect_bd_intf_net [get_bd_intf_pins m_axis_read_pkg] [get_bd_intf_pins tcp_rxHandler_0/m_axis_tcp_read_pkg]

  # main data path through FIFO, RX handler, RX depacketizer
  connect_bd_intf_net [get_bd_intf_pins s_axis_rx_data] [get_bd_intf_pins rx_fifo/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins rx_fifo/M_AXIS] [get_bd_intf_pins tcp_rxHandler_0/s_axis_tcp_rx_data]
  connect_bd_intf_net [get_bd_intf_pins tcp_depacketizer_0/in_r] [get_bd_intf_pins tcp_rxHandler_0/m_data_out]
  connect_bd_intf_net [get_bd_intf_pins tcp_depacketizer_0/notif_in] [get_bd_intf_pins tcp_rxHandler_0/m_notif_out]
  connect_bd_intf_net [get_bd_intf_pins tcp_depacketizer_0/notif_out] [get_bd_intf_pins m_axis_notification]
  connect_bd_intf_net [get_bd_intf_pins tcp_depacketizer_0/out_r] [get_bd_intf_pins dpkt_fifo/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dpkt_fifo/M_AXIS] [get_bd_intf_pins m_axis_rx_data]

  # Create port connections
  connect_bd_net -net ap_clk [get_bd_pins ap_clk]  [get_bd_pins tcp_depacketizer_0/ap_clk] \
                                                   [get_bd_pins tcp_rxHandler_0/ap_clk] \
                                                   [get_bd_pins dpkt_fifo/s_axis_aclk] \
                                                   [get_bd_pins rx_fifo/s_axis_aclk]
  connect_bd_net -net ap_rst_n [get_bd_pins ap_rst_n] [get_bd_pins tcp_depacketizer_0/ap_rst_n] \
                                                      [get_bd_pins tcp_rxHandler_0/ap_rst_n] \
                                                      [get_bd_pins dpkt_fifo/s_axis_aresetn] \
                                                      [get_bd_pins rx_fifo/s_axis_aresetn]

  # Restore current instance
  current_bd_instance $oldCurInst
}

proc create_rdma_rx_subsystem { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_rx_subsystem() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_control
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_pktsts
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_rx_data
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_rx_data
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_notification
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_notification
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_ub_rq

  # Create pins
  create_bd_pin -dir I -type clk ap_clk
  create_bd_pin -dir I -type rst ap_rst_n

  set rx_fifo [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 rx_fifo ]
  set_property -dict [ list \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.IS_ACLK_ASYNC {0} \
   CONFIG.TDATA_NUM_BYTES {64} \
 ] $rx_fifo

  set dpkt_fifo [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 dpkt_fifo]
  set_property -dict [ list \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.TDATA_NUM_BYTES {64} \
    CONFIG.FIFO_DEPTH {64} \
    CONFIG.FIFO_MEMORY_TYPE {distributed} \
  ] $dpkt_fifo
  
  # Create instances of RDMA blocks
  set rdma_depacketizer_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:rdma_depacketizer:1.0 rdma_depacketizer_0 ]

  # Create FIFO for RQ forward to Microblaze
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 ub_notif_fifo 
  set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER] [get_bd_cells ub_notif_fifo]
  set_property -dict [list CONFIG.FIFO_DEPTH {64} CONFIG.FIFO_MEMORY_TYPE {distributed} CONFIG.HAS_TLAST {1}] [get_bd_cells ub_notif_fifo]

  # Create interface connections
  connect_bd_intf_net -intf_net control [get_bd_intf_pins s_axi_control] [get_bd_intf_pins rdma_depacketizer_0/s_axi_control]

  # various metadata
  connect_bd_intf_net [get_bd_intf_pins s_axis_notification] [get_bd_intf_pins rdma_depacketizer_0/notif_in]
  connect_bd_intf_net [get_bd_intf_pins m_axis_pktsts] [get_bd_intf_pins rdma_depacketizer_0/sts]
	connect_bd_intf_net [get_bd_intf_pins rdma_depacketizer_0/notif_out] [get_bd_intf_pins m_axis_notification]

  # main data path through FIFO, RX depacketizer
  connect_bd_intf_net [get_bd_intf_pins s_axis_rx_data] [get_bd_intf_pins rx_fifo/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins rx_fifo/M_AXIS] [get_bd_intf_pins rdma_depacketizer_0/in_r]
  connect_bd_intf_net [get_bd_intf_pins rdma_depacketizer_0/out_r] [get_bd_intf_pins dpkt_fifo/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dpkt_fifo/M_AXIS] [get_bd_intf_pins m_axis_rx_data]

  connect_bd_intf_net [get_bd_intf_pins rdma_depacketizer_0/ub_notif_out] [get_bd_intf_pins ub_notif_fifo/S_AXIS] 
  connect_bd_intf_net [get_bd_intf_pins ub_notif_fifo/M_AXIS] [get_bd_intf_pins m_axis_ub_rq] 

  # Create port connections
  connect_bd_net -net ap_clk [get_bd_pins ap_clk]  [get_bd_pins rdma_depacketizer_0/ap_clk] \
                                                   [get_bd_pins dpkt_fifo/s_axis_aclk] \
                                                   [get_bd_pins ub_notif_fifo/s_axis_aclk] \
                                                   [get_bd_pins rx_fifo/s_axis_aclk]
  connect_bd_net -net ap_rst_n [get_bd_pins ap_rst_n] [get_bd_pins rdma_depacketizer_0/ap_rst_n] \
                                                      [get_bd_pins dpkt_fifo/s_axis_aresetn] \
                                                      [get_bd_pins ub_notif_fifo/s_axis_aresetn] \
                                                      [get_bd_pins rx_fifo/s_axis_aresetn]

  # Restore current instance
  current_bd_instance $oldCurInst
}