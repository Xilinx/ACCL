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
}

# Break casting kernel connections and redo them through a switch for each of the CCLO instances
proc rewire_compression {idx} {
    rewire_compression_lane $idx decompress 0 upcast_0
    rewire_compression_lane $idx decompress 1 upcast_1
    rewire_compression_lane $idx compress 0 downcast_0
}

proc rewire_compression_lane {cclo_idx clane_type clane_idx ip_name} {
    set swname cclo${cclo_idx}_${clane_type}${clane_idx}_sw
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 ${swname}
    set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells ${swname}]
    set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {2} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {4} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells ${swname}]
    set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells ${swname}]
    set_property -dict [list CONFIG.HAS_TSTRB {0} CONFIG.HAS_TKEEP {1}] [get_bd_cells ${swname}]
    set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells ${swname}]
    set_property -dict [list CONFIG.M01_S01_CONNECTIVITY {0}] [get_bd_cells ${swname}]
    connect_clk_rst ${swname}/aclk ${swname}/aresetn $cclo_idx

    delete_bd_objs [get_bd_intf_nets ${ip_name}_out_r]
    delete_bd_objs [get_bd_intf_nets ccl_offload_${cclo_idx}_m_axis_${clane_type}${clane_idx}]
    connect_bd_intf_net [get_bd_intf_pins ccl_offload_${cclo_idx}/m_axis_${clane_type}${clane_idx}] [get_bd_intf_pins ${swname}/S00_AXIS]
    connect_bd_intf_net [get_bd_intf_pins ${swname}/M00_AXIS] [get_bd_intf_pins ccl_offload_${cclo_idx}/s_axis_${clane_type}${clane_idx}]
    connect_bd_intf_net [get_bd_intf_pins ${swname}/M01_AXIS] [get_bd_intf_pins ${ip_name}/in_r]
    connect_bd_intf_net [get_bd_intf_pins ${ip_name}/out_r] [get_bd_intf_pins ${swname}/S01_AXIS]
}

# break reduction kernel connections and redo them through switches
proc rewire_reduction {idx} {
    # remove existing stream infrastructure
    delete_bd_objs [get_bd_intf_nets arith_hp_${idx}_out_r] [get_bd_intf_nets dwc_arith_hp_${idx}_out_r_M_AXIS] [get_bd_cells dwc_arith_hp_${idx}_out_r]
    delete_bd_objs [get_bd_intf_nets arith_fp_${idx}_out_r] [get_bd_intf_nets dwc_arith_fp_${idx}_out_r_M_AXIS] [get_bd_cells dwc_arith_fp_${idx}_out_r]
    delete_bd_objs [get_bd_intf_nets arith_dp_${idx}_out_r] [get_bd_intf_nets dwc_arith_dp_${idx}_out_r_M_AXIS] [get_bd_cells dwc_arith_dp_${idx}_out_r]
    delete_bd_objs [get_bd_intf_nets arith_i32_${idx}_out_r] [get_bd_intf_nets dwc_arith_i32_${idx}_out_r_M_AXIS] [get_bd_cells dwc_arith_i32_${idx}_out_r]
    delete_bd_objs [get_bd_intf_nets arith_i64_${idx}_out_r]
    delete_bd_objs [get_bd_intf_nets ccl_offload_${idx}_m_axis_arith_op]

    # create switch for operand
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 cclo${idx}_reduce_op_sw
    set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells cclo${idx}_reduce_op_sw]
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {5} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {4} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells cclo${idx}_reduce_op_sw]
    set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells cclo${idx}_reduce_op_sw]
    set_property -dict [list CONFIG.HAS_TSTRB {0} CONFIG.HAS_TKEEP {1}] [get_bd_cells cclo${idx}_reduce_op_sw]
    set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells cclo${idx}_reduce_op_sw]
    connect_clk_rst cclo${idx}_reduce_op_sw/aclk cclo${idx}_reduce_op_sw/aresetn $idx

    # create switch for result
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 cclo${idx}_reduce_res_sw
    set_property -dict [list CONFIG.HAS_TLAST.VALUE_SRC USER CONFIG.TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells cclo${idx}_reduce_res_sw]
    set_property -dict [list CONFIG.NUM_SI {5} CONFIG.NUM_MI {1} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {0} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.ARB_ON_TLAST {1} CONFIG.DECODER_REG {1}] [get_bd_cells cclo${idx}_reduce_res_sw]
    set_property -dict [list CONFIG.HAS_TSTRB.VALUE_SRC USER CONFIG.HAS_TKEEP.VALUE_SRC USER] [get_bd_cells cclo${idx}_reduce_res_sw]
    set_property -dict [list CONFIG.HAS_TSTRB {0} CONFIG.HAS_TKEEP {1}] [get_bd_cells cclo${idx}_reduce_res_sw]
    set_property -dict [list CONFIG.ARB_ALGORITHM {3}] [get_bd_cells cclo${idx}_reduce_res_sw]
    connect_clk_rst cclo${idx}_reduce_res_sw/aclk cclo${idx}_reduce_res_sw/aresetn $idx

    # connect IPs to switches
    connect_bd_intf_net [get_bd_intf_pins ccl_offload_${idx}/m_axis_arith_op] [get_bd_intf_pins cclo${idx}_reduce_op_sw/S00_AXIS]
    connect_bd_intf_net [get_bd_intf_pins cclo${idx}_reduce_op_sw/M00_AXIS] [get_bd_intf_pins arith_fp_${idx}/in_r]
    connect_bd_intf_net [get_bd_intf_pins cclo${idx}_reduce_op_sw/M01_AXIS] [get_bd_intf_pins arith_dp_${idx}/in_r]
    connect_bd_intf_net [get_bd_intf_pins cclo${idx}_reduce_op_sw/M02_AXIS] [get_bd_intf_pins arith_i32_${idx}/in_r]
    connect_bd_intf_net [get_bd_intf_pins cclo${idx}_reduce_op_sw/M03_AXIS] [get_bd_intf_pins arith_i64_${idx}/in_r]
    connect_bd_intf_net [get_bd_intf_pins cclo${idx}_reduce_op_sw/M04_AXIS] [get_bd_intf_pins arith_hp_${idx}/in_r]

    connect_bd_intf_net [get_bd_intf_pins arith_fp_${idx}/out_r] [get_bd_intf_pins cclo${idx}_reduce_res_sw/S00_AXIS]
    connect_bd_intf_net [get_bd_intf_pins arith_dp_${idx}/out_r] [get_bd_intf_pins cclo${idx}_reduce_res_sw/S01_AXIS]
    connect_bd_intf_net [get_bd_intf_pins arith_i32_${idx}/out_r] [get_bd_intf_pins cclo${idx}_reduce_res_sw/S02_AXIS]
    connect_bd_intf_net [get_bd_intf_pins arith_i64_${idx}/out_r] [get_bd_intf_pins cclo${idx}_reduce_res_sw/S03_AXIS]
    connect_bd_intf_net [get_bd_intf_pins arith_hp_${idx}/out_r] [get_bd_intf_pins cclo${idx}_reduce_res_sw/S04_AXIS]
    connect_bd_intf_net [get_bd_intf_pins cclo${idx}_reduce_res_sw/M00_AXIS] [get_bd_intf_pins ccl_offload_${idx}/s_axis_arith_res]

}