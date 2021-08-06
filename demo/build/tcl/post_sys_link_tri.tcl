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

# Break existing UDP connections and redo them through an AXI switch
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 udp_axis_switch
set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells udp_axis_switch]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {3} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {16} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells udp_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells udp_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB {1} CONFIG.HAS_TKEEP {1}] [get_bd_cells udp_axis_switch]
set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells udp_axis_switch]
if {![catch { connect_bd_net [get_bd_ports clkwiz_kernel_clk_out1] [get_bd_pins udp_axis_switch/aclk] } ]} {
    puts "Inferred shell xilinx_u280_xdma_201920_3"
    connect_bd_net [get_bd_pins udp_axis_switch/aresetn] [get_bd_pins slr0/peripheral_aresetn]
} 
if {![catch { connect_bd_net [get_bd_pins udp_axis_switch/aclk] [get_bd_pins slr1/clkwiz_kernel_clk_out_gen] } ]} {
    puts "Inferred shell xilinx_u250_xdma_201830_2"
    connect_bd_net [get_bd_pins udp_axis_switch/aresetn] [get_bd_pins slr0/peripheral_aresetn]
}

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
if {![catch { connect_bd_net [get_bd_ports clkwiz_kernel_clk_out1] [get_bd_pins tcp_axis_switch/aclk] } ]} {
    puts "Inferred shell xilinx_u280_xdma_201920_3"
    connect_bd_net [get_bd_pins tcp_axis_switch/aresetn] [get_bd_pins slr0/peripheral_aresetn]
} 
if {![catch { connect_bd_net [get_bd_pins tcp_axis_switch/aclk] [get_bd_pins slr1/clkwiz_kernel_clk_out_gen] } ]} {
    puts "Inferred shell xilinx_u250_xdma_201830_2"
    connect_bd_net [get_bd_pins tcp_axis_switch/aresetn] [get_bd_pins slr0/peripheral_aresetn]
}

delete_bd_objs [get_bd_intf_nets network_krnl_2_net_tx]
delete_bd_objs [get_bd_intf_nets network_krnl_0_net_tx]
delete_bd_objs [get_bd_intf_nets network_krnl_1_net_tx]
connect_bd_intf_net [get_bd_intf_pins network_krnl_0/net_tx] [get_bd_intf_pins tcp_axis_switch/S00_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M00_AXIS] [get_bd_intf_pins network_krnl_0/net_rx]
connect_bd_intf_net [get_bd_intf_pins network_krnl_1/net_tx] [get_bd_intf_pins tcp_axis_switch/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M01_AXIS] [get_bd_intf_pins network_krnl_1/net_rx]
connect_bd_intf_net [get_bd_intf_pins network_krnl_2/net_tx] [get_bd_intf_pins tcp_axis_switch/S02_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M02_AXIS] [get_bd_intf_pins network_krnl_2/net_rx]

# Break casting kernel connections and redo them through a switch for each of the CCLO instances
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 cclo0_krnl_axis_switch
set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells cclo0_krnl_axis_switch]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {3} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {4} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells cclo0_krnl_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells cclo0_krnl_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB {1} CONFIG.HAS_TKEEP {1}] [get_bd_cells cclo0_krnl_axis_switch]
set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells cclo0_krnl_axis_switch]
set_property -dict [list CONFIG.M01_S01_CONNECTIVITY {0} CONFIG.M01_S02_CONNECTIVITY {0} CONFIG.M02_S01_CONNECTIVITY {0} CONFIG.M02_S02_CONNECTIVITY {0}] [get_bd_cells cclo0_krnl_axis_switch]
if {![catch { connect_bd_net [get_bd_ports clkwiz_kernel_clk_out1] [get_bd_pins cclo0_krnl_axis_switch/aclk] } ]} {
    puts "Inferred shell xilinx_u280_xdma_201920_3"
    connect_bd_net [get_bd_pins cclo0_krnl_axis_switch/aresetn] [get_bd_pins slr0/peripheral_aresetn]
} 
if {![catch { connect_bd_net [get_bd_pins cclo0_krnl_axis_switch/aclk] [get_bd_pins slr1/clkwiz_kernel_clk_out_gen] } ]} {
    puts "Inferred shell xilinx_u250_xdma_201830_2"
    connect_bd_net [get_bd_pins cclo0_krnl_axis_switch/aresetn] [get_bd_pins slr0/peripheral_aresetn]
}

delete_bd_objs [get_bd_intf_nets upcast_0_out_r]
delete_bd_objs [get_bd_intf_nets ccl_offload_0_m_axis_krnl]
delete_bd_objs [get_bd_intf_nets downcast_0_out_r]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_krnl] [get_bd_intf_pins cclo0_krnl_axis_switch/S00_AXIS]
connect_bd_intf_net [get_bd_intf_pins cclo0_krnl_axis_switch/M00_AXIS] [get_bd_intf_pins ccl_offload_0/s_axis_krnl]
connect_bd_intf_net [get_bd_intf_pins cclo0_krnl_axis_switch/M01_AXIS] [get_bd_intf_pins downcast_0/in_r]
connect_bd_intf_net [get_bd_intf_pins downcast_0/out_r] [get_bd_intf_pins cclo0_krnl_axis_switch/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins cclo0_krnl_axis_switch/M02_AXIS] [get_bd_intf_pins upcast_0/in_r]
connect_bd_intf_net [get_bd_intf_pins upcast_0/out_r] [get_bd_intf_pins cclo0_krnl_axis_switch/S02_AXIS]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 cclo1_krnl_axis_switch
set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells cclo1_krnl_axis_switch]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {3} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {4} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells cclo1_krnl_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells cclo1_krnl_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB {1} CONFIG.HAS_TKEEP {1}] [get_bd_cells cclo1_krnl_axis_switch]
set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells cclo1_krnl_axis_switch]
set_property -dict [list CONFIG.M01_S01_CONNECTIVITY {0} CONFIG.M01_S02_CONNECTIVITY {0} CONFIG.M02_S01_CONNECTIVITY {0} CONFIG.M02_S02_CONNECTIVITY {0}] [get_bd_cells cclo1_krnl_axis_switch]
if {![catch { connect_bd_net [get_bd_ports clkwiz_kernel_clk_out1] [get_bd_pins cclo1_krnl_axis_switch/aclk] } ]} {
    puts "Inferred shell xilinx_u280_xdma_201920_3"
    connect_bd_net [get_bd_pins cclo1_krnl_axis_switch/aresetn] [get_bd_pins slr1/peripheral_aresetn]
} 
if {![catch { connect_bd_net [get_bd_pins cclo1_krnl_axis_switch/aclk] [get_bd_pins slr1/clkwiz_kernel_clk_out_gen] } ]} {
    puts "Inferred shell xilinx_u250_xdma_201830_2"
    connect_bd_net [get_bd_pins cclo1_krnl_axis_switch/aresetn] [get_bd_pins slr1/peripheral_aresetn]
}

delete_bd_objs [get_bd_intf_nets upcast_1_out_r]
delete_bd_objs [get_bd_intf_nets ccl_offload_1_m_axis_krnl]
delete_bd_objs [get_bd_intf_nets downcast_1_out_r]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_1/m_axis_krnl] [get_bd_intf_pins cclo1_krnl_axis_switch/S00_AXIS]
connect_bd_intf_net [get_bd_intf_pins cclo1_krnl_axis_switch/M00_AXIS] [get_bd_intf_pins ccl_offload_1/s_axis_krnl]
connect_bd_intf_net [get_bd_intf_pins cclo1_krnl_axis_switch/M01_AXIS] [get_bd_intf_pins downcast_1/in_r]
connect_bd_intf_net [get_bd_intf_pins downcast_1/out_r] [get_bd_intf_pins cclo1_krnl_axis_switch/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins cclo1_krnl_axis_switch/M02_AXIS] [get_bd_intf_pins upcast_1/in_r]
connect_bd_intf_net [get_bd_intf_pins upcast_1/out_r] [get_bd_intf_pins cclo1_krnl_axis_switch/S02_AXIS]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 cclo2_krnl_axis_switch
set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells cclo2_krnl_axis_switch]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {3} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {4} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells cclo2_krnl_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells cclo2_krnl_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB {1} CONFIG.HAS_TKEEP {1}] [get_bd_cells cclo2_krnl_axis_switch]
set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells cclo2_krnl_axis_switch]
set_property -dict [list CONFIG.M01_S01_CONNECTIVITY {0} CONFIG.M01_S02_CONNECTIVITY {0} CONFIG.M02_S01_CONNECTIVITY {0} CONFIG.M02_S02_CONNECTIVITY {0}] [get_bd_cells cclo2_krnl_axis_switch]
if {![catch { connect_bd_net [get_bd_ports clkwiz_kernel_clk_out1] [get_bd_pins cclo2_krnl_axis_switch/aclk] } ]} {
    puts "Inferred shell xilinx_u280_xdma_201920_3"
    connect_bd_net [get_bd_pins cclo2_krnl_axis_switch/aresetn] [get_bd_pins slr2/peripheral_aresetn]
} 
if {![catch { connect_bd_net [get_bd_pins cclo2_krnl_axis_switch/aclk] [get_bd_pins slr2/clkwiz_kernel_clk_out_gen] } ]} {
    puts "Inferred shell xilinx_u250_xdma_201830_2"
    connect_bd_net [get_bd_pins cclo2_krnl_axis_switch/aresetn] [get_bd_pins slr0/peripheral_aresetn]
}

delete_bd_objs [get_bd_intf_nets upcast_2_out_r]
delete_bd_objs [get_bd_intf_nets ccl_offload_2_m_axis_krnl]
delete_bd_objs [get_bd_intf_nets downcast_2_out_r]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_2/m_axis_krnl] [get_bd_intf_pins cclo2_krnl_axis_switch/S00_AXIS]
connect_bd_intf_net [get_bd_intf_pins cclo2_krnl_axis_switch/M00_AXIS] [get_bd_intf_pins ccl_offload_2/s_axis_krnl]
connect_bd_intf_net [get_bd_intf_pins cclo2_krnl_axis_switch/M01_AXIS] [get_bd_intf_pins downcast_2/in_r]
connect_bd_intf_net [get_bd_intf_pins downcast_2/out_r] [get_bd_intf_pins cclo2_krnl_axis_switch/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins cclo2_krnl_axis_switch/M02_AXIS] [get_bd_intf_pins upcast_2/in_r]
connect_bd_intf_net [get_bd_intf_pins upcast_2/out_r] [get_bd_intf_pins cclo2_krnl_axis_switch/S02_AXIS]