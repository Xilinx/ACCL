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

# VNX Packetizer
create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 ila_packetizer
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {3} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells ila_packetizer]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_packetizer]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_packetizer]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_packetizer]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ila_packetizer/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ila_packetizer/resetn]
connect_bd_intf_net [get_bd_intf_pins ila_packetizer/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins udp_tx_subsystem/s_axis_command]
connect_bd_intf_net [get_bd_intf_pins ila_packetizer/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins udp_tx_subsystem/s_axis_data]
connect_bd_intf_net [get_bd_intf_pins ila_packetizer/SLOT_2_AXIS] -boundary_type upper [get_bd_intf_pins udp_tx_subsystem/m_axis_data]

# VNX Depacketizer
create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 ila_depacketizer
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {2} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells ila_depacketizer]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_depacketizer]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_depacketizer]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ila_depacketizer/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ila_depacketizer/resetn]
connect_bd_intf_net [get_bd_intf_pins ila_depacketizer/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins udp_rx_subsystem/s_axis_data]
connect_bd_intf_net [get_bd_intf_pins ila_depacketizer/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins udp_rx_subsystem/m_axis_status]

# TODO: add tcp packetizer/depacketizer


save_bd_design
validate_bd_design
