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

proc connect_clk_rst {clksig rstsig rstslr} {
    if {![catch { connect_bd_net [get_bd_ports clkwiz_kernel_clk_out1] [get_bd_pins $clksig] } ]} {
        puts "Inferred shell xilinx_u280_xdma_201920_3"
        connect_bd_net [get_bd_pins slr${rstslr}/peripheral_aresetn] [get_bd_pins $rstsig]
    }
    if {![catch { connect_bd_net [get_bd_pins slr1/clkwiz_kernel_clk_out_gen] [get_bd_pins $clksig] } ]} {
        puts "Inferred shell xilinx_u250_xdma_201830_2"
        connect_bd_net [get_bd_pins slr${rstslr}/peripheral_aresetn] [get_bd_pins $rstsig]
    }
    if {![catch { connect_bd_net [get_bd_pins ss_ucs/aclk_kernel_00] [get_bd_pins $clksig] } ]} {
        puts "Inferred shell xilinx_u250_gen3x16_xdma_3_1_202020_1"
        connect_bd_net [get_bd_pins ip_psr_aresetn_kernel_00_slr${rstslr}/peripheral_aresetn] [get_bd_pins $rstsig]
    }
    if {![catch { connect_bd_net [get_bd_pins ulp_ucs/aclk_kernel_00] [get_bd_pins $clksig] } ]} {
        puts "Inferred shell xilinx_u55c_gen3x16_xdma_2_202110_1 or xilinx_u55c_gen3x16_xdma_3_202210_1"
        connect_bd_net [get_bd_pins proc_sys_reset_kernel_slr${rstslr}/peripheral_aresetn] [get_bd_pins $rstsig]
    }
}

# check for hardware emulation
set hw_emu 0
set sim_clk_gens [get_bd_cells -filter "VLNV=~xilinx.com:ip:sim_clk_gen:*"]
if {[llength ${sim_clk_gens}] > 0} {
    puts "Hardware emulation detected."
    set hw_emu 1
}

# Break existing TCP connections and redo them through an AXI switch
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 tcp_axis_switch
set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells tcp_axis_switch]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {3} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {16} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells tcp_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells tcp_axis_switch]
set_property -dict [list CONFIG.HAS_TSTRB {1} CONFIG.HAS_TKEEP {1}] [get_bd_cells tcp_axis_switch]
set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells tcp_axis_switch]
if {[expr $hw_emu == 0]} {
    connect_clk_rst tcp_axis_switch/aclk tcp_axis_switch/aresetn 1
} else {
    connect_bd_net [get_bd_pins kernel_clk/clk] [get_bd_pins tcp_axis_switch/aclk]
    connect_bd_net [get_bd_pins psr_kernel_clk_0/peripheral_aresetn] [get_bd_pins tcp_axis_switch/aresetn]
}

delete_bd_objs [get_bd_intf_nets poe_0_net_tx]
delete_bd_objs [get_bd_intf_nets poe_1_net_tx]
delete_bd_objs [get_bd_intf_nets poe_2_net_tx]
connect_bd_intf_net [get_bd_intf_pins poe_0/net_tx] [get_bd_intf_pins tcp_axis_switch/S00_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M00_AXIS] [get_bd_intf_pins poe_0/net_rx]
connect_bd_intf_net [get_bd_intf_pins poe_1/net_tx] [get_bd_intf_pins tcp_axis_switch/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M01_AXIS] [get_bd_intf_pins poe_1/net_rx]
connect_bd_intf_net [get_bd_intf_pins poe_2/net_tx] [get_bd_intf_pins tcp_axis_switch/S02_AXIS]
connect_bd_intf_net [get_bd_intf_pins tcp_axis_switch/M02_AXIS] [get_bd_intf_pins poe_2/net_rx]
