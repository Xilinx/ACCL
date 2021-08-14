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

# Add DMA/switch debug
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_m01_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axis_switch_0_M01_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axis_switch_0_M02_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axis_switch_0_M03_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_M3_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma_1_m00_axis}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma_1_m00_sts}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma_1_M_AXIS}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {S_AXIS_1}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_m00_cmd}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma_0_foo}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma_0_m00_axis}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {dma_0_m00_sts}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_M00_AXI}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {control_M02_AXI}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {reduce_arith_0_out_r}]


apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
  [get_bd_intf_nets axis_switch_0_M01_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets axis_switch_0_M02_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets axis_switch_0_M03_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets control_M3_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets control_M00_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets control_M02_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets control_m00_cmd] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets control_m01_cmd] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets dma_0_foo] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets dma_0_m00_axis] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets dma_0_m00_sts] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets dma_1_m00_axis] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets dma_1_m00_sts] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets dma_1_M_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets S_AXIS_1] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \
  [get_bd_intf_nets reduce_arith_0_out_r] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/ap_clk" SYSTEM_ILA "Auto" APC_EN "0" } \

]
save_bd_design
validate_bd_design


