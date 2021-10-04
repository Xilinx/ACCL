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

# Add arith debug
create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 ila_arith
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {2} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells ila_arith]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_arith]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_arith]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ila_arith/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ila_arith/resetn]
connect_bd_intf_net [get_bd_intf_pins ila_arith/SLOT_0_AXIS] [get_bd_intf_pins m_axis_arith_op]
connect_bd_intf_net [get_bd_intf_pins ila_arith/SLOT_1_AXIS] [get_bd_intf_pins s_axis_arith_res]

# Add streaming kernel debug
create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 ila_krnl
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {2} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells ila_krnl]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_krnl]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_krnl]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ila_krnl/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ila_krnl/resetn]
connect_bd_intf_net [get_bd_intf_pins ila_krnl/SLOT_0_AXIS] [get_bd_intf_pins m_axis_krnl]
connect_bd_intf_net [get_bd_intf_pins ila_krnl/SLOT_1_AXIS] [get_bd_intf_pins s_axis_krnl]

# Add streaming kernel debug
create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 ila_compression
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {6} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells ila_compression]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_compression]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_compression]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_compression]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_compression]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_compression]
set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_compression]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ila_compression/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ila_compression/resetn]
connect_bd_intf_net [get_bd_intf_pins ila_compression/SLOT_0_AXIS] [get_bd_intf_pins m_axis_compression0]
connect_bd_intf_net [get_bd_intf_pins ila_compression/SLOT_1_AXIS] [get_bd_intf_pins s_axis_compression0]
connect_bd_intf_net [get_bd_intf_pins ila_compression/SLOT_2_AXIS] [get_bd_intf_pins m_axis_compression1]
connect_bd_intf_net [get_bd_intf_pins ila_compression/SLOT_3_AXIS] [get_bd_intf_pins s_axis_compression1]
connect_bd_intf_net [get_bd_intf_pins ila_compression/SLOT_4_AXIS] [get_bd_intf_pins m_axis_compression2]
connect_bd_intf_net [get_bd_intf_pins ila_compression/SLOT_5_AXIS] [get_bd_intf_pins s_axis_compression2]

save_bd_design
validate_bd_design
