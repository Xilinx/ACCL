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

source [file dirname [file normalize [info script]]]/rewire.tcl

# Break existing UDP connections and redo them through an AXI switch
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 udp_axis_switch
set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells udp_axis_switch]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {3} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {16} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells udp_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells udp_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB {1} CONFIG.HAS_TKEEP {1}] [get_bd_cells udp_axis_switch]
set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells udp_axis_switch]
connect_clk_rst udp_axis_switch/aclk udp_axis_switch/aresetn 1

delete_bd_objs [get_bd_intf_nets ccl_offload_0_m_axis_udp_tx_data]
delete_bd_objs [get_bd_intf_nets ccl_offload_1_m_axis_udp_tx_data]
delete_bd_objs [get_bd_intf_nets ccl_offload_2_m_axis_udp_tx_data]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_1/m_axis_udp_tx_data] [get_bd_intf_pins udp_axis_switch/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_2/m_axis_udp_tx_data] [get_bd_intf_pins udp_axis_switch/S02_AXIS]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_udp_tx_data] [get_bd_intf_pins udp_axis_switch/S00_AXIS]
connect_bd_intf_net [get_bd_intf_pins udp_axis_switch/M00_AXIS] [get_bd_intf_pins ccl_offload_0/s_axis_udp_rx_data]
connect_bd_intf_net [get_bd_intf_pins udp_axis_switch/M01_AXIS] [get_bd_intf_pins ccl_offload_1/s_axis_udp_rx_data]
connect_bd_intf_net [get_bd_intf_pins udp_axis_switch/M02_AXIS] [get_bd_intf_pins ccl_offload_2/s_axis_udp_rx_data]

# Break existing TCP connections and redo them through an AXI switch
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 tcp_axis_switch
set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells tcp_axis_switch]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {3} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {16} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells tcp_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells tcp_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB {1} CONFIG.HAS_TKEEP {1}] [get_bd_cells tcp_axis_switch]
set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells tcp_axis_switch]
connect_clk_rst tcp_axis_switch/aclk tcp_axis_switch/aresetn 1

delete_bd_objs [get_bd_intf_nets network_krnl_2_net_tx]
delete_bd_objs [get_bd_intf_nets network_krnl_0_net_tx]
delete_bd_objs [get_bd_intf_nets network_krnl_1_net_tx]
connect_bd_intf_net [get_bd_intf_pins network_krnl_0/net_tx] [get_bd_intf_pins tcp_axis_switch/S00_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M00_AXIS] [get_bd_intf_pins network_krnl_0/net_rx]
connect_bd_intf_net [get_bd_intf_pins network_krnl_1/net_tx] [get_bd_intf_pins tcp_axis_switch/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M01_AXIS] [get_bd_intf_pins network_krnl_1/net_rx]
connect_bd_intf_net [get_bd_intf_pins network_krnl_2/net_tx] [get_bd_intf_pins tcp_axis_switch/S02_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M02_AXIS] [get_bd_intf_pins network_krnl_2/net_rx]

rewire_cast 0
rewire_cast 1
rewire_cast 2
rewire_sum 0
rewire_sum 1
rewire_sum 2