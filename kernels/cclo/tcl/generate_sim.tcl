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

# netStackType - UDP or TCP - type of POE attachment generated
# enableDMA - 0/1 - enables DMAs, providing support for send/recv from/to memory, and collectives
# enableArithmetic - 0/1 - enables arithmetic, providing support for reduction collectives and combine primitive
# enableCompression - 0/1 - enables compression feature
# enableExtKrnlStream - 0/1 - enables PL stream attachments, providing support for non-memory send/recv
# debugLevel - 0/1/2 - enables DEBUG/TRACE support for the control microblaze
set stacktype [lindex $::argv 0]
set en_dma [lindex $::argv 1]
set en_arith [lindex $::argv 2]
set en_compress [lindex $::argv 3]
set en_extkrnl [lindex $::argv 4]
set memsize [lindex $::argv 5]
puts "$stacktype $en_dma $memsize"

# open project
open_project ./ccl_offload_ex/ccl_offload_ex.xpr
update_compile_order -fileset sim_1

# add plugins to the catalog
set_property ip_repo_paths { ./hls ./../plugins } [current_project]
update_ip_catalog

# open the block design
open_bd_design {./ccl_offload_ex/ccl_offload_ex.srcs/sources_1/bd/ccl_offload_bd/ccl_offload_bd.bd}

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
catch { delete_bd_objs [get_bd_intf_ports bscan_0] }

if { $en_dma != 0 } {
    create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 blk_mem_gen_0
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0
    set_property -dict [list CONFIG.SINGLE_PORT_BRAM {1} CONFIG.DATA_WIDTH {512} CONFIG.ECC_TYPE {0}] [get_bd_cells axi_bram_ctrl_0]
    connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTA]

    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_crossbar:2.1 axi_crossbar_0
    set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells axi_crossbar_0]
    connect_bd_intf_net [get_bd_intf_pins axi_crossbar_0/M00_AXI] [get_bd_intf_pins axi_bram_ctrl_0/S_AXI]
    connect_bd_intf_net [get_bd_intf_pins axi_crossbar_0/S00_AXI] -boundary_type upper [get_bd_intf_pins cclo/m_axi_0]
    connect_bd_intf_net [get_bd_intf_pins axi_crossbar_0/S01_AXI] -boundary_type upper [get_bd_intf_pins cclo/m_axi_1]

    set s_axi [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_data ]
    set_property -dict [ list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.FREQ_HZ {250000000} CONFIG.HAS_BRESP {0} CONFIG.HAS_BURST {0} CONFIG.HAS_CACHE {0} CONFIG.HAS_LOCK {0} CONFIG.HAS_PROT {0} CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.HAS_WSTRB {1} CONFIG.NUM_READ_OUTSTANDING {1} CONFIG.NUM_WRITE_OUTSTANDING {1} CONFIG.PROTOCOL {AXI4} CONFIG.READ_WRITE_MODE {READ_WRITE} ] $s_axi
    connect_bd_intf_net [get_bd_intf_ports s_axi_data] [get_bd_intf_pins axi_crossbar_0/S02_AXI]

    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_crossbar_0/aclk]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_crossbar_0/aresetn]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn]

    #assign addresses and set ranges
    assign_bd_address
    set_property range $memsize [get_bd_addr_segs {cclo/dma_0/Data/SEG_axi_bram_ctrl_0_Mem0}]
    set_property range $memsize [get_bd_addr_segs {cclo/dma_1/Data/SEG_axi_bram_ctrl_0_Mem0}]
    set_property range $memsize [get_bd_addr_segs {s_axi_data/SEG_axi_bram_ctrl_0_Mem0}]
    set_property offset 0x0000000000000000 [get_bd_addr_segs {s_axi_data/SEG_axi_bram_ctrl_0_Mem0}]
    set_property offset 0x0000000000000000 [get_bd_addr_segs {cclo/dma_1/Data/SEG_axi_bram_ctrl_0_Mem0}]
    set_property offset 0x0000000000000000 [get_bd_addr_segs {cclo/dma_0/Data/SEG_axi_bram_ctrl_0_Mem0}]

    group_bd_cells external_memory [get_bd_cells axi_bram_ctrl_0] [get_bd_cells blk_mem_gen_0] [get_bd_cells axi_crossbar_0]

}

if { $stacktype == "TCP" } {
    create_bd_cell -type ip -vlnv xilinx.com:hls:network_krnl:1.0 dummy_tcp_stack
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins dummy_tcp_stack/ap_clk]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins dummy_tcp_stack/ap_rst_n]
    #replace direct connections to ports with connections via dummy tcp stack
    delete_bd_objs [get_bd_intf_ports s_axis_eth_rx_data]
    make_bd_intf_pins_external  [get_bd_intf_pins dummy_tcp_stack/net_rx]
    set_property name s_axis_eth_rx_data [get_bd_intf_ports net_rx_0]
    delete_bd_objs [get_bd_intf_ports m_axis_eth_tx_data]
    make_bd_intf_pins_external  [get_bd_intf_pins dummy_tcp_stack/net_tx]
    set_property name m_axis_eth_tx_data [get_bd_intf_ports net_tx_0]
    #make remaining connections
    delete_bd_objs [get_bd_intf_ports s_axis_eth_open_status]
    delete_bd_objs [get_bd_intf_ports s_axis_eth_port_status]
    delete_bd_objs [get_bd_intf_ports s_axis_eth_notification]
    delete_bd_objs [get_bd_intf_ports s_axis_eth_rx_meta]
    delete_bd_objs [get_bd_intf_ports s_axis_eth_tx_status]
    delete_bd_objs [get_bd_intf_ports m_axis_eth_open_connection]
    delete_bd_objs [get_bd_intf_ports m_axis_eth_listen_port]
    delete_bd_objs [get_bd_intf_ports m_axis_eth_read_pkg]
    delete_bd_objs [get_bd_intf_ports m_axis_eth_tx_meta]
    
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/m_axis_tcp_open_status] [get_bd_intf_pins cclo/s_axis_eth_open_status]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/m_axis_tcp_port_status] [get_bd_intf_pins cclo/s_axis_eth_port_status]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/m_axis_tcp_notification] [get_bd_intf_pins cclo/s_axis_eth_notification]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/m_axis_tcp_rx_meta] [get_bd_intf_pins cclo/s_axis_eth_rx_meta]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/m_axis_tcp_tx_status] [get_bd_intf_pins cclo/s_axis_eth_tx_status]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/s_axis_tcp_open_connection] [get_bd_intf_pins cclo/m_axis_eth_open_connection]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/s_axis_tcp_listen_port] [get_bd_intf_pins cclo/m_axis_eth_listen_port]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/s_axis_tcp_read_pkg] [get_bd_intf_pins cclo/m_axis_eth_read_pkg]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/s_axis_tcp_tx_meta] [get_bd_intf_pins cclo/m_axis_eth_tx_meta]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/s_axis_tcp_tx_data] [get_bd_intf_pins cclo/m_axis_eth_tx_data]
    connect_bd_intf_net [get_bd_intf_pins dummy_tcp_stack/m_axis_tcp_rx_data] [get_bd_intf_pins cclo/s_axis_eth_rx_data]
}

# loopback between streaming kernel interfaces
if { $en_extkrnl != 0 } {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 krnl_loopback
    connect_bd_intf_net [get_bd_intf_pins cclo/m_axis_krnl] [get_bd_intf_pins krnl_loopback/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins krnl_loopback/M_AXIS] [get_bd_intf_pins cclo/s_axis_krnl]
    connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins krnl_loopback/s_axis_aresetn]
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins krnl_loopback/s_axis_aclk]
}

# connect arithmetic plugins
if { $en_arith != 0 } {
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
                                [get_bd_cells axis_ic_arith_res] \
                                [get_bd_cells axis_ic_arith_op]
}

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

if { $en_compress != 0 } {
    assemble_clane 0
    assemble_clane 1
    assemble_clane 2

    group_bd_cells compression  [get_bd_cells compression_lane_0] \
                                [get_bd_cells compression_lane_1] \
                                [get_bd_cells compression_lane_2]
}

save_bd_design
validate_bd_design

set extra_sim_options ""
if { $en_dma == 1 } { set extra_sim_options "$extra_sim_options -d AXI_DATA_ACCESS " }
set_property -name {xsim.compile.xvlog.more_options} -value $extra_sim_options -objects [get_filesets sim_1]
set_property -name {xsim.elaborate.xelab.more_options} -value {-dll} -objects [get_filesets sim_1]
set_property generate_scripts_only 1 [current_fileset -simset]
launch_simulation -scripts_only

# close and exit
close_project
exit