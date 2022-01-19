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

if { $::argc eq 0 } {
    set memsize "256K"
} else {
    set memsize [lindex $::argv 0]
}

# add plugins to the catalog
set_property ip_repo_paths { ./hls ./../plugins } [current_project]
update_ip_catalog

# remove some ports which we don't need (to be replaced by logic)
catch { delete_bd_objs [get_bd_intf_ports m_axi_0] }
catch { delete_bd_objs [get_bd_intf_ports m_axi_1] }
catch { delete_bd_objs [get_bd_intf_ports m_axis_arith_op] }
catch { delete_bd_objs [get_bd_intf_ports s_axis_arith_res] }
catch { delete_bd_objs [get_bd_intf_ports s_axis_compression0] }
catch { delete_bd_objs [get_bd_intf_ports s_axis_compression1] }
catch { delete_bd_objs [get_bd_intf_ports s_axis_compression2] }
catch { delete_bd_objs [get_bd_intf_ports m_axis_compression0] }
catch { delete_bd_objs [get_bd_intf_ports m_axis_compression1] }
catch { delete_bd_objs [get_bd_intf_ports m_axis_compression2] }
catch { delete_bd_objs [get_bd_intf_ports m_axis_krnl] }
catch { delete_bd_objs [get_bd_intf_ports s_axis_krnl] }

create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 blk_mem_gen_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0
set_property -dict [list CONFIG.SINGLE_PORT_BRAM {1} CONFIG.DATA_WIDTH {512} CONFIG.ECC_TYPE {0}] [get_bd_cells axi_bram_ctrl_0]
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTA]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_crossbar:2.1 axi_crossbar_0
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells axi_crossbar_0]
connect_bd_intf_net [get_bd_intf_pins axi_crossbar_0/M00_AXI] [get_bd_intf_pins axi_bram_ctrl_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_crossbar_0/S00_AXI] -boundary_type upper [get_bd_intf_pins cclo/m_axi_0]
connect_bd_intf_net [get_bd_intf_pins axi_crossbar_0/S01_AXI] -boundary_type upper [get_bd_intf_pins cclo/m_axi_1]
make_bd_intf_pins_external  [get_bd_intf_pins axi_crossbar_0/S02_AXI]
set_property NAME s_axi_data [get_bd_intf_ports /S02_AXI_0]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_crossbar_0/aclk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_crossbar_0/aresetn]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn]


# loopback between streaming kernel interfaces
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 krnl_loopback
connect_bd_intf_net [get_bd_intf_pins cclo/m_axis_krnl] [get_bd_intf_pins krnl_loopback/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins krnl_loopback/M_AXIS] [get_bd_intf_pins cclo/s_axis_krnl]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins krnl_loopback/s_axis_aresetn]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins krnl_loopback/s_axis_aclk]

#assign addresses and set ranges
assign_bd_address
set_property range $memsize [get_bd_addr_segs {cclo/dma_0/Data/SEG_axi_bram_ctrl_0_Mem0}]
set_property range $memsize [get_bd_addr_segs {cclo/dma_1/Data/SEG_axi_bram_ctrl_0_Mem0}]
set_property range $memsize [get_bd_addr_segs {s_axi_data/SEG_axi_bram_ctrl_0_Mem0}]

group_bd_cells external_memory [get_bd_cells axi_bram_ctrl_0] [get_bd_cells axi_crossbar_0] [get_bd_cells blk_mem_gen_0]

# connect arithmetic plugins
create_bd_cell -type ip -vlnv xilinx.com:ACCL:reduce_sum_double:1.0 reduce_sum_double_0
create_bd_cell -type ip -vlnv xilinx.com:ACCL:reduce_sum_float:1.0 reduce_sum_float_0
create_bd_cell -type ip -vlnv xilinx.com:ACCL:reduce_sum_half:1.0 reduce_sum_half_0
create_bd_cell -type ip -vlnv xilinx.com:ACCL:reduce_sum_int32_t:1.0 reduce_sum_int32_t_0
create_bd_cell -type ip -vlnv xilinx.com:ACCL:reduce_sum_int64_t:1.0 reduce_sum_int64_t_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_interconnect:2.1 axis_ic_arith_op
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {5} CONFIG.ARB_ON_TLAST {0}] [get_bd_cells axis_ic_arith_op]

connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_op/S00_AXIS] [get_bd_intf_pins cclo/m_axis_arith_op]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_op/M00_AXIS] [get_bd_intf_pins reduce_sum_float_0/in_r]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_op/M01_AXIS] [get_bd_intf_pins reduce_sum_double_0/in_r]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_op/M02_AXIS] [get_bd_intf_pins reduce_sum_half_0/in_r]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_op/M03_AXIS] [get_bd_intf_pins reduce_sum_int32_t_0/in_r]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_op/M04_AXIS] [get_bd_intf_pins reduce_sum_int64_t_0/in_r]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_op/ACLK]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_op/ARESETN]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_op/S00_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_op/M00_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_op/M01_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_op/M02_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_op/M03_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_op/M04_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_op/M04_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_op/M03_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_op/M02_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_op/M01_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_op/M00_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_op/S00_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins reduce_sum_float_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins reduce_sum_double_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins reduce_sum_int32_t_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins reduce_sum_int64_t_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins reduce_sum_half_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins reduce_sum_int32_t_0/ap_clk]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins reduce_sum_int64_t_0/ap_clk]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins reduce_sum_float_0/ap_clk]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins reduce_sum_double_0/ap_clk]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins reduce_sum_half_0/ap_clk]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_interconnect:2.1 axis_ic_arith_res
set_property -dict [list CONFIG.NUM_SI {5} CONFIG.NUM_MI {1} CONFIG.ARB_ON_TLAST {1}] [get_bd_cells axis_ic_arith_res]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_res/M00_AXIS] [get_bd_intf_pins cclo/s_axis_arith_res]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_res/S00_AXIS] [get_bd_intf_pins reduce_sum_float_0/out_r]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_res/S01_AXIS] [get_bd_intf_pins reduce_sum_double_0/out_r]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_res/S02_AXIS] [get_bd_intf_pins reduce_sum_half_0/out_r]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_res/S03_AXIS] [get_bd_intf_pins reduce_sum_int32_t_0/out_r]
connect_bd_intf_net [get_bd_intf_pins axis_ic_arith_res/S04_AXIS] [get_bd_intf_pins reduce_sum_int64_t_0/out_r]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_res/ACLK]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_res/ARESETN]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_res/M00_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_res/S00_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_res/S01_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_res/S02_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_res/S03_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_arith_res/S04_AXIS_ACLK]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_res/S04_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_res/S03_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_res/S02_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_res/S01_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_res/S00_AXIS_ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_arith_res/M00_AXIS_ARESETN]

group_bd_cells reduce_arith [get_bd_cells reduce_sum_double_0] \
                            [get_bd_cells reduce_sum_int32_t_0] \
                            [get_bd_cells reduce_sum_half_0] \
                            [get_bd_cells reduce_sum_float_0] \
                            [get_bd_cells reduce_sum_int64_t_0] \
                            [get_bd_cells axis_interconnect_0] \
                            [get_bd_cells axis_ic_arith_res]

# implement compression lanes

proc assemble_clane { idx } {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_interconnect:2.1 axis_ic_clane${idx}_op
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2} CONFIG.ARB_ON_TLAST {0}] [get_bd_cells axis_ic_clane${idx}_op]
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_interconnect:2.1 axis_ic_clane${idx}_res
    set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1} CONFIG.ARB_ON_TLAST {1}] [get_bd_cells axis_ic_clane${idx}_res]
    create_bd_cell -type ip -vlnv xilinx.com:ACCL:fp_hp_stream_conv:1.0 fp_hp_stream_conv_${idx}
    create_bd_cell -type ip -vlnv xilinx.com:ACCL:hp_fp_stream_conv:1.0 hp_fp_stream_conv_${idx}
    connect_bd_intf_net [get_bd_intf_pins axis_ic_clane${idx}_res/M00_AXIS] [get_bd_intf_pins cclo/s_axis_compression${idx}]
    connect_bd_intf_net [get_bd_intf_pins axis_ic_clane${idx}_res/S00_AXIS] [get_bd_intf_pins fp_hp_stream_conv_${idx}/out_r]
    connect_bd_intf_net [get_bd_intf_pins axis_ic_clane${idx}_res/S01_AXIS] [get_bd_intf_pins hp_fp_stream_conv_${idx}/out_r]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_clane${idx}_res/ACLK]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_clane${idx}_res/ARESETN]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_clane${idx}_res/M00_AXIS_ACLK]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_clane${idx}_res/S00_AXIS_ACLK]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_clane${idx}_res/S01_AXIS_ACLK]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_clane${idx}_res/M00_AXIS_ARESETN]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_clane${idx}_res/S00_AXIS_ARESETN]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_clane${idx}_res/S01_AXIS_ARESETN]

    connect_bd_intf_net [get_bd_intf_pins axis_ic_clane${idx}_op/S00_AXIS] [get_bd_intf_pins cclo/m_axis_compression${idx}]
    connect_bd_intf_net [get_bd_intf_pins axis_ic_clane${idx}_op/M00_AXIS] [get_bd_intf_pins fp_hp_stream_conv_${idx}/in_r]
    connect_bd_intf_net [get_bd_intf_pins axis_ic_clane${idx}_op/M01_AXIS] [get_bd_intf_pins hp_fp_stream_conv_${idx}/in_r]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_clane${idx}_op/ACLK]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_clane${idx}_op/ARESETN]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_clane${idx}_op/S00_AXIS_ACLK]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_clane${idx}_op/M00_AXIS_ACLK]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axis_ic_clane${idx}_op/M01_AXIS_ACLK]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_clane${idx}_op/S00_AXIS_ARESETN]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_clane${idx}_op/M00_AXIS_ARESETN]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axis_ic_clane${idx}_op/M01_AXIS_ARESETN]

    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins hp_fp_stream_conv_${idx}/ap_rst_n]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins fp_hp_stream_conv_${idx}/ap_rst_n]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins hp_fp_stream_conv_${idx}/ap_clk]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins fp_hp_stream_conv_${idx}/ap_clk]

    group_bd_cells compression_lane_${idx}  [get_bd_cells fp_hp_stream_conv_${idx}] \
                                            [get_bd_cells hp_fp_stream_conv_${idx}] \
                                            [get_bd_cells axis_ic_clane${idx}_op] \
                                            [get_bd_cells axis_ic_clane${idx}_res]
}

assemble_clane 0
assemble_clane 1
assemble_clane 2

group_bd_cells compression  [get_bd_cells compression_lane_0] \
                            [get_bd_cells compression_lane_1] \
                            [get_bd_cells compression_lane_2]

validate_bd_design
