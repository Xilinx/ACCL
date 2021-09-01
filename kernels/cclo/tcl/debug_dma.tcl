
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

# Add DMA debug
proc dma_debug {idx} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 ila_dma${idx}
    set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {6} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells ila_dma${idx}]
    set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_dma${idx}]
    set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_dma${idx}]
    set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_dma${idx}]
    set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_dma${idx}]
    set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_dma${idx}]
    set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells ila_dma${idx}]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ila_dma${idx}/clk]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ila_dma${idx}/resetn]
    connect_bd_intf_net [get_bd_intf_pins ila_dma${idx}/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_mm2s_cmd]
    connect_bd_intf_net [get_bd_intf_pins ila_dma${idx}/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins udp_rx_subsystem/m_axis_data]
    connect_bd_intf_net [get_bd_intf_pins ila_dma${idx}/SLOT_2_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_s2mm_cmd]
    connect_bd_intf_net [get_bd_intf_pins ila_dma${idx}/SLOT_3_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_s2mm_sts]
    connect_bd_intf_net [get_bd_intf_pins ila_dma${idx}/SLOT_4_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_mm2s]
    connect_bd_intf_net [get_bd_intf_pins ila_dma${idx}/SLOT_5_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_mm2s_sts]
}

dma_debug 0
dma_debug 1
dma_debug 2

save_bd_design
validate_bd_design
