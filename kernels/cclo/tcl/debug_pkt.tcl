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

set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_M03_AXI}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {vnx_depacketizer_0_out_r}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {vnx_depacketizer_0_sts_V}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {net_rx_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_M4_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axis_switch_0_M00_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_M04_AXI}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axis_data_fifo_1_M_AXIS}]

set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma_0_foo}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_m00_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {s_axi_control}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control/microblaze_0_M0_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control/axis_fifo_sts_3_M_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/util_reduced_logic_1_Res }]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/util_reduced_logic_0_Res }]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {control/util_vector_logic_0_Res }]

apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
    [get_bd_intf_nets axis_data_fifo_1_M_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets axis_switch_0_M00_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets control_M4_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets control_M03_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets control_M04_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets net_rx_1] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets vnx_depacketizer_0_out_r] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets vnx_depacketizer_0_sts_V] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets control_m00_cmd] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets dma_0_foo] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets control/microblaze_0_M0_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_intf_nets control/axis_fifo_sts_3_M_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
    [get_bd_nets control/util_reduced_logic_0_Res] {PROBE_TYPE "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" } \
    [get_bd_nets control/util_reduced_logic_1_Res] {PROBE_TYPE "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" } \
    [get_bd_nets control/util_vector_logic_0_Res] {PROBE_TYPE "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" } \
    [get_bd_intf_nets s_axi_control] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
]
save_bd_design
validate_bd_design
