# /*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
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

set nettype [lindex $::argv 0]
set build_dir [lindex $::argv 1]
open_project "$build_dir/lynx/lynx.xpr"
update_compile_order -fileset sources_1
create_bd_design "accl_bd"
update_compile_order -fileset sources_1
set_property  ip_repo_paths  "$build_dir ../../kernels" [current_project]
update_ip_catalog

create_bd_cell -type ip -vlnv Xilinx:ACCL:ccl_offload:1.0 ccl_offload_0
create_bd_cell -type ip -vlnv xilinx.com:ACCL:cyt_dma_adapter:1.0 cyt_dma_adapter_0

connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma0_mm2s_cmd] [get_bd_intf_pins cyt_dma_adapter_0/dma0_mm2s_cmd]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma1_mm2s_cmd] [get_bd_intf_pins cyt_dma_adapter_0/dma1_mm2s_cmd]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma1_s2mm_cmd] [get_bd_intf_pins cyt_dma_adapter_0/dma1_s2mm_cmd]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma0_s2mm_cmd] [get_bd_intf_pins cyt_dma_adapter_0/dma0_s2mm_cmd]
connect_bd_intf_net [get_bd_intf_pins cyt_dma_adapter_0/dma0_s2mm_sts] [get_bd_intf_pins ccl_offload_0/s_axis_dma0_s2mm_sts]
connect_bd_intf_net [get_bd_intf_pins cyt_dma_adapter_0/dma1_s2mm_sts] [get_bd_intf_pins ccl_offload_0/s_axis_dma1_s2mm_sts]
connect_bd_intf_net [get_bd_intf_pins cyt_dma_adapter_0/dma0_mm2s_sts] [get_bd_intf_pins ccl_offload_0/s_axis_dma0_mm2s_sts]
connect_bd_intf_net [get_bd_intf_pins cyt_dma_adapter_0/dma1_mm2s_sts] [get_bd_intf_pins ccl_offload_0/s_axis_dma1_mm2s_sts]
make_bd_pins_external  [get_bd_pins ccl_offload_0/ap_clk]
make_bd_pins_external  [get_bd_pins ccl_offload_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins cyt_dma_adapter_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins cyt_dma_adapter_0/ap_rst_n]
make_bd_intf_pins_external  [get_bd_intf_pins cyt_dma_adapter_0/cyt_byp_wr_sts]
make_bd_intf_pins_external  [get_bd_intf_pins cyt_dma_adapter_0/cyt_byp_rd_sts]
make_bd_intf_pins_external  [get_bd_intf_pins cyt_dma_adapter_0/cyt_byp_wr_cmd]
make_bd_intf_pins_external  [get_bd_intf_pins cyt_dma_adapter_0/cyt_byp_rd_cmd]

create_bd_cell -type ip -vlnv xilinx.com:ACCL:reduce_ops:1.0 reduce_ops_0
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_arith_op0] [get_bd_intf_pins reduce_ops_0/in0]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_arith_op1] [get_bd_intf_pins reduce_ops_0/in1]
connect_bd_intf_net [get_bd_intf_pins reduce_ops_0/out_r] [get_bd_intf_pins ccl_offload_0/s_axis_arith_res]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins reduce_ops_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins reduce_ops_0/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ACCL:hostctrl:1.0 hostctrl_0
connect_bd_intf_net [get_bd_intf_pins hostctrl_0/cmd] [get_bd_intf_pins ccl_offload_0/s_axis_call_req]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_call_ack] [get_bd_intf_pins hostctrl_0/sts]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins hostctrl_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins hostctrl_0/ap_rst_n]

# direct loopback for compression and kernel streams
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_krnl] [get_bd_intf_pins ccl_offload_0/s_axis_krnl]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_compression0] [get_bd_intf_pins ccl_offload_0/s_axis_compression0]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_compression1] [get_bd_intf_pins ccl_offload_0/s_axis_compression1]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_compression2] [get_bd_intf_pins ccl_offload_0/s_axis_compression2]

# create axis switch
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_2_to_1_inst_0
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.TDATA_NUM_BYTES {64} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.ARB_ON_TLAST {1} CONFIG.NUM_MI {1} CONFIG.DECODER_REG {0} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.Component_Name {axis_switch_2_to_1_inst_0}] [get_bd_cells axis_switch_2_to_1_inst_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_2_to_1_inst_1
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.TDATA_NUM_BYTES {64} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.ARB_ON_TLAST {1} CONFIG.NUM_MI {1} CONFIG.DECODER_REG {0} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.Component_Name {axis_switch_2_to_1_inst_1}] [get_bd_cells axis_switch_2_to_1_inst_1]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_1_to_2_inst_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2} CONFIG.TDATA_NUM_BYTES {64} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {8} CONFIG.DECODER_REG {1} CONFIG.Component_Name {axis_switch_1_to_2_inst_0}] [get_bd_cells axis_switch_1_to_2_inst_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_1_to_2_inst_1
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2} CONFIG.TDATA_NUM_BYTES {64} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {8} CONFIG.DECODER_REG {1} CONFIG.Component_Name {axis_switch_1_to_2_inst_1}] [get_bd_cells axis_switch_1_to_2_inst_1]

switch $nettype {
    "TCP" {
        # externalize TCP streams
        make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/s_axis_eth_rx_data]
        make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/m_axis_eth_tx_data]
        make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/s_axis_eth_tx_status]
        make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/m_axis_eth_read_pkg]
        make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/s_axis_eth_rx_meta]
        make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/s_axis_eth_notification]
        make_bd_intf_pins_external [get_bd_intf_pins ccl_offload_0/m_axis_eth_tx_meta]
    }
    "RDMA" {
        # externalize RDMA streams
        # data streams
        set m_axis_eth_tx_data [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_eth_tx_data ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_eth_tx_data
        set s_axis_eth_rx_data [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_eth_rx_data ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {1} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {8} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_eth_rx_data

        # RDMA sq and rq
        set m_axis_rdma_sq [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_rdma_sq ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_rdma_sq

        # RDMA extra pair of host/card streams
        set m_axis_host_2 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_2 ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_host_2
        set m_axis_card_2 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_2 ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_card_2
        set s_axis_host_2 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_2 ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_host_2
        set s_axis_card_2 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_2 ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_card_2

        # RDMA wr_req and rd_req
        set s_axis_rdma_wr_req [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_rdma_wr_req ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {12} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_rdma_wr_req
        set s_axis_rdma_rd_req [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_rdma_rd_req ]
        set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {12} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_rdma_rd_req

        # connections for rdma_arbiter and the axi 1-to-2 switch
        create_bd_cell -type ip -vlnv xilinx.com:ACCL:cyt_rdma_arbiter:1.0 cyt_rdma_arbiter_0
        create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_1_to_2_inst_2
        set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2} CONFIG.TDATA_NUM_BYTES {64} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.TDEST_WIDTH {8} CONFIG.DECODER_REG {1} CONFIG.Component_Name {axis_switch_1_to_2_inst_2}] [get_bd_cells axis_switch_1_to_2_inst_2]
        connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins cyt_rdma_arbiter_0/ap_clk]
        connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins cyt_rdma_arbiter_0/ap_rst_n]
        connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_1_to_2_inst_2/aclk]
        connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_1_to_2_inst_2/aresetn]

        connect_bd_intf_net [get_bd_intf_ports s_axis_eth_rx_data] [get_bd_intf_pins cyt_rdma_arbiter_0/s_axis]
        connect_bd_intf_net [get_bd_intf_ports s_axis_rdma_wr_req] [get_bd_intf_pins cyt_rdma_arbiter_0/s_meta]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_arbiter_0/m_meta_0] [get_bd_intf_pins ccl_offload_0/s_axis_eth_notification]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_arbiter_0/m_axis_0] [get_bd_intf_pins ccl_offload_0/s_axis_eth_rx_data]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_arbiter_0/m_meta_1] [get_bd_intf_pins cyt_dma_adapter_0/rdma_wr_req]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_arbiter_0/m_axis_1] [get_bd_intf_pins axis_switch_1_to_2_inst_2/S00_AXIS]
        connect_bd_intf_net [get_bd_intf_pins axis_switch_1_to_2_inst_2/M00_AXIS] [get_bd_intf_ports m_axis_card_2]
        connect_bd_intf_net [get_bd_intf_pins axis_switch_1_to_2_inst_2/M01_AXIS] [get_bd_intf_ports m_axis_host_2]

        # connections for rdma_mux and the axi 2-to-1 switch
        create_bd_cell -type ip -vlnv xilinx.com:ACCL:cyt_rdma_mux:1.0 cyt_rdma_mux_0
        create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_2_to_1_inst_2
        set_property -dict [list CONFIG.NUM_SI {2} CONFIG.TDATA_NUM_BYTES {64} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.ARB_ON_TLAST {1} CONFIG.NUM_MI {1} CONFIG.DECODER_REG {0} CONFIG.ARB_ON_MAX_XFERS {0} CONFIG.Component_Name {axis_switch_2_to_1_inst_2}] [get_bd_cells axis_switch_2_to_1_inst_2]
        connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins cyt_rdma_mux_0/ap_clk]
        connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins cyt_rdma_mux_0/ap_rst_n]
        connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_2_to_1_inst_2/aclk]
        connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_2_to_1_inst_2/aresetn]
        create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_2
        set_property -dict [list CONFIG.CONST_WIDTH {2}] [get_bd_cells xlconstant_2]
        set_property -dict [list CONFIG.CONST_VAL {0}] [get_bd_cells xlconstant_2]
        connect_bd_net [get_bd_pins xlconstant_2/dout] [get_bd_pins axis_switch_2_to_1_inst_2/s_req_suppress]

        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/s_meta_0] [get_bd_intf_pins ccl_offload_0/m_axis_rdma_sq]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/s_axis_0] [get_bd_intf_pins ccl_offload_0/m_axis_eth_tx_data]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/s_meta_1] [get_bd_intf_ports s_axis_rdma_rd_req]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/s_axis_1] [get_bd_intf_pins axis_switch_2_to_1_inst_2/M00_AXIS]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/m_meta_0] [get_bd_intf_ports m_axis_rdma_sq]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/m_meta_1] [get_bd_intf_pins cyt_dma_adapter_0/rdma_rd_req]
        connect_bd_intf_net [get_bd_intf_pins cyt_rdma_mux_0/m_axis] [get_bd_intf_ports m_axis_eth_tx_data]
        connect_bd_intf_net [get_bd_intf_ports s_axis_host_2] [get_bd_intf_pins axis_switch_2_to_1_inst_2/S00_AXIS]
        connect_bd_intf_net [get_bd_intf_ports s_axis_card_2] [get_bd_intf_pins axis_switch_2_to_1_inst_2/S01_AXIS]


    }
    default {
        puts "Unrecognized network backend"
        exit
    }
}


# externalize DMA data streams

set m_axis_host_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_0 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_host_0
set m_axis_host_1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_1 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_host_1
set m_axis_card_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_0 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_card_0
set m_axis_card_1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_1 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_card_1

set s_axis_host_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_0 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_host_0
set s_axis_host_1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_1 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_host_1
set s_axis_card_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_0 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_card_0
set s_axis_card_1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_1 ]
set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_card_1


# s_axis_host_0 and s_axis_card_0 multiplexed to single s_axis_dma0_mm2s stream, round-robin by tlast
connect_bd_intf_net [get_bd_intf_ports s_axis_host_0] [get_bd_intf_pins axis_switch_2_to_1_inst_0/S00_AXIS]
connect_bd_intf_net [get_bd_intf_ports s_axis_card_0] [get_bd_intf_pins axis_switch_2_to_1_inst_0/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_switch_2_to_1_inst_0/M00_AXIS] [get_bd_intf_pins ccl_offload_0/s_axis_dma0_mm2s]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_2_to_1_inst_0/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_2_to_1_inst_0/aresetn]

create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0
set_property -dict [list CONFIG.CONST_WIDTH {2}] [get_bd_cells xlconstant_0]
set_property -dict [list CONFIG.CONST_VAL {0}] [get_bd_cells xlconstant_0]
connect_bd_net [get_bd_pins xlconstant_0/dout] [get_bd_pins axis_switch_2_to_1_inst_0/s_req_suppress]

# s_axis_host_1 and s_axis_card_1 multiplexed to single s_axis_dma1_mm2s stream, round-robin by tlast
connect_bd_intf_net [get_bd_intf_ports s_axis_host_1] [get_bd_intf_pins axis_switch_2_to_1_inst_1/S00_AXIS]
connect_bd_intf_net [get_bd_intf_ports s_axis_card_1] [get_bd_intf_pins axis_switch_2_to_1_inst_1/S01_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_switch_2_to_1_inst_1/M00_AXIS] [get_bd_intf_pins ccl_offload_0/s_axis_dma1_mm2s]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_2_to_1_inst_1/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_2_to_1_inst_1/aresetn]

create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_1
set_property -dict [list CONFIG.CONST_WIDTH {2}] [get_bd_cells xlconstant_1]
set_property -dict [list CONFIG.CONST_VAL {0}] [get_bd_cells xlconstant_1]
connect_bd_net [get_bd_pins xlconstant_1/dout] [get_bd_pins axis_switch_2_to_1_inst_1/s_req_suppress]

# m_axis_dma0_s2mm multiplex to m_axis_host_0 and m_axis_card_0 according to the strm flag encoded in m_axis_dma0_s2mm tdest
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma0_s2mm] [get_bd_intf_pins axis_switch_1_to_2_inst_0/S00_AXIS]
connect_bd_intf_net [get_bd_intf_ports m_axis_card_0] [get_bd_intf_pins axis_switch_1_to_2_inst_0/M00_AXIS]
connect_bd_intf_net [get_bd_intf_ports m_axis_host_0] [get_bd_intf_pins axis_switch_1_to_2_inst_0/M01_AXIS]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_1_to_2_inst_0/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_1_to_2_inst_0/aresetn]

# m_axis_dma1_s2mm multiplex to m_axis_host_1 and m_axis_card_1 according to the strm flag encoded in m_axis_dma1_s2mm tdest
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/m_axis_dma1_s2mm] [get_bd_intf_pins axis_switch_1_to_2_inst_1/S00_AXIS]
connect_bd_intf_net [get_bd_intf_ports m_axis_card_1] [get_bd_intf_pins axis_switch_1_to_2_inst_1/M00_AXIS]
connect_bd_intf_net [get_bd_intf_ports m_axis_host_1] [get_bd_intf_pins axis_switch_1_to_2_inst_1/M01_AXIS]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins axis_switch_1_to_2_inst_1/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins axis_switch_1_to_2_inst_1/aresetn]



# connect up AXI lite
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_MI {2} CONFIG.NUM_SI {1}] [get_bd_cells smartconnect_0]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins hostctrl_0/s_axi_control] [get_bd_intf_pins smartconnect_0/M00_AXI]
connect_bd_intf_net [get_bd_intf_pins ccl_offload_0/s_axi_control] [get_bd_intf_pins smartconnect_0/M01_AXI]
make_bd_intf_pins_external  [get_bd_intf_pins smartconnect_0/S00_AXI]
set_property -dict [list CONFIG.ADDR_WIDTH {16}] [get_bd_intf_ports S00_AXI_0]

# Create address segments
assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces S00_AXI_0] [get_bd_addr_segs ccl_offload_0/s_axi_control/reg0] -force
assign_bd_address -offset 0x00002000 -range 0x00002000 -target_address_space [get_bd_addr_spaces S00_AXI_0] [get_bd_addr_segs hostctrl_0/s_axi_control/Reg] -force

set_property CONFIG.PROTOCOL AXI4LITE [get_bd_intf_ports /S00_AXI_0]
set_property -dict [list CONFIG.HAS_BURST {0} CONFIG.HAS_CACHE {0} CONFIG.HAS_LOCK {0} CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0}] [get_bd_intf_ports S00_AXI_0]
validate_bd_design
save_bd_design

make_wrapper -files [get_files "$build_dir/lynx/lynx.srcs/sources_1/bd/accl_bd/accl_bd.bd"] -top
add_files -norecurse "$build_dir/lynx/lynx.gen/sources_1/bd/accl_bd/hdl/accl_bd_wrapper.v"
update_compile_order -fileset sources_1
exit
