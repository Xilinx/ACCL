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

# Add aximm debug
create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 ila_control
set_property -dict [list CONFIG.C_MON_TYPE {MIX} CONFIG.C_NUM_MONITOR_SLOTS {7} CONFIG.C_NUM_OF_PROBES {2} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells ila_control]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells ila_control]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells ila_control]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells ila_control]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells ila_control]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells ila_control]
set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells ila_control]
set_property -dict [list CONFIG.C_SLOT_6_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells ila_control]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ila_control/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ila_control/resetn]
connect_bd_intf_net [get_bd_intf_pins ila_control/SLOT_0_AXI] [get_bd_intf_pins control_xbar/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ila_control/SLOT_1_AXI] [get_bd_intf_pins control_xbar/M00_AXI]
connect_bd_intf_net [get_bd_intf_pins ila_control/SLOT_2_AXI] [get_bd_intf_pins control_xbar/M01_AXI]
connect_bd_intf_net [get_bd_intf_pins ila_control/SLOT_3_AXI] [get_bd_intf_pins control_xbar/M02_AXI]
connect_bd_intf_net [get_bd_intf_pins ila_control/SLOT_4_AXI] [get_bd_intf_pins control_xbar/M03_AXI]
connect_bd_intf_net [get_bd_intf_pins ila_control/SLOT_5_AXI] [get_bd_intf_pins control_xbar/M04_AXI]
connect_bd_intf_net [get_bd_intf_pins ila_control/SLOT_6_AXI] [get_bd_intf_pins control_xbar/M05_AXI]
connect_bd_net [get_bd_pins ila_control/probe0] [get_bd_pins axi_gpio_tdest/gpio_io_o]
connect_bd_net [get_bd_pins ila_control/probe1] [get_bd_pins axi_gpio_tdest/gpio2_io_o]

#Add Microblaze debug
create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 control/ila_mb
set_property -dict [list CONFIG.C_MON_TYPE {MIX} CONFIG.C_NUM_MONITOR_SLOTS {7} CONFIG.C_NUM_OF_PROBES {2} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells control/ila_mb]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells control/ila_mb]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells control/ila_mb]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells control/ila_mb]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells control/ila_mb]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells control/ila_mb]
set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells control/ila_mb]
set_property -dict [list CONFIG.C_SLOT_6_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells control/ila_mb]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins control/ila_mb/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins control/ila_mb/resetn]
connect_bd_intf_net [get_bd_intf_pins control/ila_mb/SLOT_0_AXI] [get_bd_intf_pins control/microblaze_0_exchange_memory/S_AXI]
connect_bd_intf_net [get_bd_intf_pins control/ila_mb/SLOT_1_AXI] [get_bd_intf_pins control/microblaze_0_axi_periph/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control/ila_mb/SLOT_2_AXI] [get_bd_intf_pins control/microblaze_0_axi_periph/M00_AXI]
connect_bd_intf_net [get_bd_intf_pins control/ila_mb/SLOT_3_AXI] [get_bd_intf_pins control/microblaze_0_axi_periph/M02_AXI]
connect_bd_intf_net [get_bd_intf_pins control/ila_mb/SLOT_4_AXI] [get_bd_intf_pins control/microblaze_0_axi_periph/M03_AXI]
connect_bd_intf_net [get_bd_intf_pins control/ila_mb/SLOT_5_AXIS] [get_bd_intf_pins control/microblaze_0_exchange_memory/host_cmd]
connect_bd_intf_net [get_bd_intf_pins control/ila_mb/SLOT_6_AXIS] [get_bd_intf_pins control/microblaze_0_exchange_memory/host_sts]
connect_bd_net [get_bd_pins control/ila_mb/probe0] [get_bd_pins control/microblaze_0_exchange_memory/encore_aresetn]
connect_bd_net [get_bd_pins control/ila_mb/probe1] [get_bd_pins control/proc_irq_concat/dout]

save_bd_design
validate_bd_design
