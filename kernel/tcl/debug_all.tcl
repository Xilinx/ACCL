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
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {reduce_arith_0_out_r}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {arith_switch_0_M00_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {arith_switch_0_M01_AXIS}]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_arith
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {3} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells system_ila_arith]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_arith]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_arith]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_arith]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_arith/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_arith/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_arith/SLOT_0_AXIS] [get_bd_intf_pins reduce_arith_0/out_r]
connect_bd_intf_net [get_bd_intf_pins system_ila_arith/SLOT_1_AXIS] [get_bd_intf_pins reduce_arith_0/in1]
connect_bd_intf_net [get_bd_intf_pins system_ila_arith/SLOT_2_AXIS] [get_bd_intf_pins reduce_arith_0/in2]


# Add DMA0 debug
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma0_rx}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma0_tx}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma0_rx_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma0_tx_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma0_rx_sts}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma0_tx_sts}]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_dma0
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {6} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells system_ila_dma0]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma0]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma0]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma0]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma0]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma0]
set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma0]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_dma0/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_dma0/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma0/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_mm2s_cmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma0/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins udp_rx_subsystem/m_axis_data]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma0/SLOT_2_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_s2mm_cmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma0/SLOT_3_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_s2mm_sts]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma0/SLOT_4_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_mm2s]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma0/SLOT_5_AXIS] -boundary_type upper [get_bd_intf_pins dma_0/dma_mm2s_sts]


# Add DMA1 debug
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma1_rx}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma1_tx}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma1_rx_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma1_tx_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma1_rx_sts}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma1_tx_sts}]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_dma1
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {6} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells system_ila_dma1]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma1]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma1]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma1]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma1]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma1]
set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma1]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_dma1/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_dma1/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma1/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins dma_1/dma_mm2s_cmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma1/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins dma_1/dma_s2mm]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma1/SLOT_2_AXIS] -boundary_type upper [get_bd_intf_pins dma_1/dma_s2mm_cmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma1/SLOT_3_AXIS] -boundary_type upper [get_bd_intf_pins dma_1/dma_mm2s]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma1/SLOT_4_AXIS] -boundary_type upper [get_bd_intf_pins dma_1/dma_mm2s_sts]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma1/SLOT_5_AXIS] -boundary_type upper [get_bd_intf_pins dma_1/dma_s2mm_sts]

# Add DMA2 debug
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma2_rx}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma2_tx}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma2_rx_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma2_tx_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma2_rx_sts}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma2_tx_sts}]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_dma2
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {6} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells system_ila_dma2]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma2]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma2]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma2]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma2]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma2]
set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_dma2]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_dma2/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_dma2/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma2/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins dma_2/dma_mm2s_cmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma2/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins dma_2/dma_s2mm]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma2/SLOT_2_AXIS] -boundary_type upper [get_bd_intf_pins dma_2/dma_s2mm_cmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma2/SLOT_3_AXIS] -boundary_type upper [get_bd_intf_pins dma_2/dma_mm2s]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma2/SLOT_4_AXIS] -boundary_type upper [get_bd_intf_pins dma_2/dma_mm2s_sts]
connect_bd_intf_net [get_bd_intf_pins system_ila_dma2/SLOT_5_AXIS] -boundary_type upper [get_bd_intf_pins dma_2/dma_s2mm_sts]

# VNX Packetizer
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {vnx_packetizer_in}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {vnx_packetizer_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {vnx_packetizer_out}]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_packetizer
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {3} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells system_ila_packetizer]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_packetizer]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_packetizer]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_packetizer]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_packetizer/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_packetizer/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_packetizer/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins udp_tx_subsystem/s_axis_command]
connect_bd_intf_net [get_bd_intf_pins system_ila_packetizer/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins udp_tx_subsystem/s_axis_data]
connect_bd_intf_net [get_bd_intf_pins system_ila_packetizer/SLOT_2_AXIS] -boundary_type upper [get_bd_intf_pins udp_tx_subsystem/m_axis_data]

# VNX Depacketizer
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {vnx_depacketizer_in}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {vnx_depacketizer_sts}]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_depacketizer
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {2} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells system_ila_depacketizer]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_depacketizer]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_depacketizer]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_depacketizer/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_depacketizer/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_depacketizer/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins udp_rx_subsystem/s_axis_data]
connect_bd_intf_net [get_bd_intf_pins system_ila_depacketizer/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins udp_rx_subsystem/m_axis_status]

# TCP Packetizer
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_pktcmd_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_tcp_open_status_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axis_switch_0_M01_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_tcp_tx_status_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_opencon_cmd_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {tcp_tx_subsystem_m_axis_tcp_tx_data}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {tcp_tx_subsystem_m_axis_tcp_tx_meta}]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_tcp_packetizer
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {7} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells system_ila_tcp_packetizer]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_packetizer]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_packetizer]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_packetizer]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_packetizer]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_packetizer]
set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_packetizer]
set_property -dict [list CONFIG.C_SLOT_6_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_packetizer]

connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_tcp_packetizer/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_tcp_packetizer/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_packetizer/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins tcp_tx_subsystem/s_axis_pktcmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_packetizer/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins tcp_tx_subsystem/s_axis_tcp_open_status]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_packetizer/SLOT_2_AXIS] -boundary_type upper [get_bd_intf_pins tcp_tx_subsystem/s_axis_tcp_tx_data]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_packetizer/SLOT_3_AXIS] -boundary_type upper [get_bd_intf_pins tcp_tx_subsystem/s_axis_tcp_tx_status]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_packetizer/SLOT_4_AXIS] -boundary_type upper [get_bd_intf_pins tcp_tx_subsystem/s_axis_opencon_cmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_packetizer/SLOT_5_AXIS] -boundary_type upper [get_bd_intf_pins tcp_tx_subsystem/m_axis_tcp_tx_data]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_packetizer/SLOT_6_AXIS] -boundary_type upper [get_bd_intf_pins tcp_tx_subsystem/m_axis_tcp_tx_meta]

# TCP Depacketizer
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_tcp_port_status_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_tcp_rx_data_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_openport_cmd_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_tcp_notification_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axis_tcp_rx_meta_1}]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_tcp_depacketizer
set_property -dict [list CONFIG.C_NUM_MONITOR_SLOTS {5} CONFIG.C_DATA_DEPTH {1024}] [get_bd_cells system_ila_tcp_depacketizer]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_depacketizer]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_depacketizer]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_depacketizer]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_depacketizer]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_tcp_depacketizer]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_tcp_depacketizer/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_tcp_depacketizer/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_depacketizer/SLOT_0_AXIS] -boundary_type upper [get_bd_intf_pins tcp_rx_subsystem/s_axis_tcp_port_status]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_depacketizer/SLOT_1_AXIS] -boundary_type upper [get_bd_intf_pins tcp_rx_subsystem/s_axis_tcp_rx_data]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_depacketizer/SLOT_2_AXIS] -boundary_type upper [get_bd_intf_pins tcp_rx_subsystem/s_axis_openport_cmd]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_depacketizer/SLOT_3_AXIS] -boundary_type upper [get_bd_intf_pins tcp_rx_subsystem/s_axis_tcp_notification]
connect_bd_intf_net [get_bd_intf_pins system_ila_tcp_depacketizer/SLOT_4_AXIS] -boundary_type upper [get_bd_intf_pins tcp_rx_subsystem/s_axis_tcp_rx_meta]


# Add aximm debug
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {switch_control}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {arith_control}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {udp_depacketizer_control}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {udp_packetizer_control}]

#Debug inside the control hierarchy, close to the Microblaze
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {host_control}]
#set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/microblaze_0/INTERRUPT }]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/rx_udp_cmd_zero }]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/rx_udp_sts_nonzero }]
#set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control/axis_fifo_sts_3_M_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control/microblaze_0_M0_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/rx_udp_cmd_nonzero }]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/rx_udp_cmd_count }]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/rx_udp_sts_count }]

create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_control
set_property -dict [list CONFIG.C_MON_TYPE {MIX} CONFIG.C_NUM_MONITOR_SLOTS {9} CONFIG.C_NUM_OF_PROBES {6} CONFIG.C_DATA_DEPTH {4096}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:aximm rtl:1.0}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_6_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_7_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_control]
set_property -dict [list CONFIG.C_SLOT_8_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0}] [get_bd_cells system_ila_control]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins system_ila_control/clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins system_ila_control/resetn]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_0_AXI] [get_bd_intf_pins reduce_arith_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_1_AXI] -boundary_type upper [get_bd_intf_pins udp_tx_subsystem/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_2_AXI] -boundary_type upper [get_bd_intf_pins udp_rx_subsystem/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_3_AXI] -boundary_type upper [get_bd_intf_pins control/encore_control]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_4_AXI] -boundary_type upper [get_bd_intf_pins control/host_control]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_5_AXIS] [get_bd_intf_pins control/microblaze_0/S3_AXIS]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_6_AXIS] [get_bd_intf_pins control/microblaze_0/M0_AXIS]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_7_AXIS] [get_bd_intf_pins control/microblaze_0/S5_AXIS]
connect_bd_intf_net [get_bd_intf_pins system_ila_control/SLOT_8_AXIS] [get_bd_intf_pins control/microblaze_0/M5_AXIS]
connect_bd_net [get_bd_pins system_ila_control/probe0] [get_bd_pins control/fifo_dma0_s2mm_cmd/axis_wr_data_count]
connect_bd_net [get_bd_pins system_ila_control/probe1] [get_bd_pins control/fifo_dma0_s2mm_sts/axis_rd_data_count]
connect_bd_net [get_bd_pins system_ila_control/probe2] [get_bd_pins control/compute_rx_udp_sts_nonzero/Res]
connect_bd_net [get_bd_pins system_ila_control/probe3] [get_bd_pins control/compute_rx_udp_cmd_nonzero/Res]
connect_bd_net [get_bd_pins system_ila_control/probe4] [get_bd_pins control/compute_rx_udp_cmd_zero/Res]
connect_bd_net [get_bd_pins system_ila_control/probe5] [get_bd_pins control/axi_timer/interrupt]

save_bd_design
validate_bd_design



