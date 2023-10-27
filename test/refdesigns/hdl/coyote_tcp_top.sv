/*******************************************************************************
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

`timescale 1ns / 1ps

import lynxTypes::*;

`include "axi_macros.svh"
`include "lynx_macros.svh"

/**
 * User logic
 * 
 */
module design_user_logic_c0_0 (
    // AXI4L CONTROL
    AXI4L.s                     axi_ctrl,

    // DESCRIPTOR BYPASS
    metaIntf.m			        bpss_rd_req,
    metaIntf.m			        bpss_wr_req,
    metaIntf.s                  bpss_rd_done,
    metaIntf.s                  bpss_wr_done,

    // AXI4S HOST STREAMS
    AXI4SR.s                    axis_host_0_sink,
    AXI4SR.m                    axis_host_0_src,
    AXI4SR.s                    axis_host_1_sink,
    AXI4SR.m                    axis_host_1_src,

    // AXI4S CARD STREAMS
    AXI4SR.s                    axis_card_0_sink,
    AXI4SR.m                    axis_card_0_src,
    AXI4SR.s                    axis_card_1_sink,
    AXI4SR.m                    axis_card_1_src,
    
    // TCP/IP QSFP0 CMD
    metaIntf.s			        tcp_0_notify,
    metaIntf.m			        tcp_0_rd_pkg,
    metaIntf.s			        tcp_0_rx_meta,
    metaIntf.m			        tcp_0_tx_meta,
    metaIntf.s			        tcp_0_tx_stat,

    // AXI4S TCP/IP QSFP0 STREAMS
    AXI4SR.s                    axis_tcp_0_sink,
    AXI4SR.m                    axis_tcp_0_src,

    // Clock and reset
    input  wire                 aclk,
    input  wire[0:0]            aresetn
);

/* -- Tie-off unused interfaces and signals ----------------------------- */
// always_comb axis_host_0_sink.tie_off_s();
// always_comb axis_host_0_src_s.tie_off_m();
// always_comb axis_card_0_sink.tie_off_s();
// always_comb axis_card_0_src_s.tie_off_m();
// always_comb axis_host_1_sink.tie_off_s();
// always_comb axis_host_1_src.tie_off_m();
// always_comb axis_card_1_sink.tie_off_s();
// always_comb axis_card_1_src_s.tie_off_m();

/* -- USER LOGIC -------------------------------------------------------- */

// Constants
localparam integer COYOTE_AXIL_ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer COYOTE_AXIL_ADDR_MSB = 16;

// Master Data Stream
AXI4SR axis_host_0_src_s ();
AXI4SR axis_host_1_src_s ();
AXI4SR axis_card_0_src_s ();
AXI4SR axis_card_1_src_s ();

// register slices
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_0_src_s),  .m_axis(axis_host_0_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_1_src_s),  .m_axis(axis_host_1_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_0_src_s),  .m_axis(axis_card_0_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_1_src_s),  .m_axis(axis_card_1_src));

// Slave Data Stream
AXI4SR axis_host_0_sink_s ();
AXI4SR axis_host_1_sink_s ();
AXI4SR axis_card_0_sink_s ();
AXI4SR axis_card_1_sink_s ();

// register slices
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_0_sink),  .m_axis(axis_host_0_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_1_sink),  .m_axis(axis_host_1_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_0_sink),  .m_axis(axis_card_0_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_1_sink),  .m_axis(axis_card_1_sink_s));


// ACCL Block Design
accl_bd_wrapper accl_system(
    .ap_clk_0(aclk),
    .ap_rst_n_0(aresetn),

    .S00_AXI_0_araddr(axi_ctrl.araddr[COYOTE_AXIL_ADDR_MSB-1:1]),
    .S00_AXI_0_arprot(axi_ctrl.arprot),
    .S00_AXI_0_arready(axi_ctrl.arready),
    .S00_AXI_0_arvalid(axi_ctrl.arvalid),
    .S00_AXI_0_awaddr(axi_ctrl.awaddr[COYOTE_AXIL_ADDR_MSB-1:1]),
    .S00_AXI_0_awprot(axi_ctrl.awprot),
    .S00_AXI_0_awready(axi_ctrl.awready),
    .S00_AXI_0_awvalid(axi_ctrl.awvalid),
    .S00_AXI_0_bready(axi_ctrl.bready),
    .S00_AXI_0_bresp(axi_ctrl.bresp),
    .S00_AXI_0_bvalid(axi_ctrl.bvalid),
    .S00_AXI_0_rdata(axi_ctrl.rdata),
    .S00_AXI_0_rready(axi_ctrl.rready),
    .S00_AXI_0_rresp(axi_ctrl.rresp),
    .S00_AXI_0_rvalid(axi_ctrl.rvalid),
    .S00_AXI_0_wdata(axi_ctrl.wdata),
    .S00_AXI_0_wready(axi_ctrl.wready),
    .S00_AXI_0_wstrb(axi_ctrl.wstrb),
    .S00_AXI_0_wvalid(axi_ctrl.wvalid),

    .cyt_byp_rd_cmd_0_tdata(bpss_rd_req.data),
    .cyt_byp_rd_cmd_0_tready(bpss_rd_req.ready),
    .cyt_byp_rd_cmd_0_tvalid(bpss_rd_req.valid),

    .cyt_byp_rd_sts_0_tdata(bpss_rd_done.data),
    .cyt_byp_rd_sts_0_tready(bpss_rd_done.ready),
    .cyt_byp_rd_sts_0_tvalid(bpss_rd_done.valid),

    .cyt_byp_wr_cmd_0_tdata(bpss_wr_req.data),
    .cyt_byp_wr_cmd_0_tready(bpss_wr_req.ready),
    .cyt_byp_wr_cmd_0_tvalid(bpss_wr_req.valid),

    .cyt_byp_wr_sts_0_tdata(bpss_wr_done.data),
    .cyt_byp_wr_sts_0_tready(bpss_wr_done.ready),
    .cyt_byp_wr_sts_0_tvalid(bpss_wr_done.valid),

    .m_axis_host_0_tdata(axis_host_0_src_s.tdata),
    .m_axis_host_0_tkeep(axis_host_0_src_s.tkeep),
    .m_axis_host_0_tlast(axis_host_0_src_s.tlast),
    .m_axis_host_0_tready(axis_host_0_src_s.tready),
    .m_axis_host_0_tvalid(axis_host_0_src_s.tvalid),
    .m_axis_host_0_tdest(),

    .m_axis_host_1_tdata(axis_host_1_src_s.tdata),
    .m_axis_host_1_tkeep(axis_host_1_src_s.tkeep),
    .m_axis_host_1_tlast(axis_host_1_src_s.tlast),
    .m_axis_host_1_tready(axis_host_1_src_s.tready),
    .m_axis_host_1_tvalid(axis_host_1_src_s.tvalid),
    .m_axis_host_1_tdest(),

    .m_axis_card_0_tdata(axis_card_0_src_s.tdata),
    .m_axis_card_0_tkeep(axis_card_0_src_s.tkeep),
    .m_axis_card_0_tlast(axis_card_0_src_s.tlast),
    .m_axis_card_0_tready(axis_card_0_src_s.tready),
    .m_axis_card_0_tvalid(axis_card_0_src_s.tvalid),
    .m_axis_card_0_tdest(),

    .m_axis_card_1_tdata(axis_card_1_src_s.tdata),
    .m_axis_card_1_tkeep(axis_card_1_src_s.tkeep),
    .m_axis_card_1_tlast(axis_card_1_src_s.tlast),
    .m_axis_card_1_tready(axis_card_1_src_s.tready),
    .m_axis_card_1_tvalid(axis_card_1_src_s.tvalid),
    .m_axis_card_1_tdest(),


    .m_axis_eth_read_pkg_0_tdata(tcp_0_rd_pkg.data),
    .m_axis_eth_read_pkg_0_tkeep(),
    .m_axis_eth_read_pkg_0_tlast(),
    .m_axis_eth_read_pkg_0_tready(tcp_0_rd_pkg.ready),
    .m_axis_eth_read_pkg_0_tstrb(),
    .m_axis_eth_read_pkg_0_tvalid(tcp_0_rd_pkg.valid),

    .m_axis_eth_tx_data_0_tdata(axis_tcp_0_src.tdata),
    .m_axis_eth_tx_data_0_tdest(axis_tcp_0_src.tid), // not driven, default 0
    .m_axis_eth_tx_data_0_tkeep(axis_tcp_0_src.tkeep),
    .m_axis_eth_tx_data_0_tlast(axis_tcp_0_src.tlast),
    .m_axis_eth_tx_data_0_tready(axis_tcp_0_src.tready),
    .m_axis_eth_tx_data_0_tvalid(axis_tcp_0_src.tvalid),

    .m_axis_eth_tx_meta_0_tdata(tcp_0_tx_meta.data),
    .m_axis_eth_tx_meta_0_tkeep(),
    .m_axis_eth_tx_meta_0_tlast(),
    .m_axis_eth_tx_meta_0_tready(tcp_0_tx_meta.ready),
    .m_axis_eth_tx_meta_0_tstrb(),
    .m_axis_eth_tx_meta_0_tvalid(tcp_0_tx_meta.valid),

    .s_axis_host_0_tdata(axis_host_0_sink_s.tdata),
    .s_axis_host_0_tkeep(axis_host_0_sink_s.tkeep),
    .s_axis_host_0_tlast(axis_host_0_sink_s.tlast),
    .s_axis_host_0_tready(axis_host_0_sink_s.tready),
    .s_axis_host_0_tvalid(axis_host_0_sink_s.tvalid),

    .s_axis_host_1_tdata(axis_host_1_sink_s.tdata),
    .s_axis_host_1_tkeep(axis_host_1_sink_s.tkeep),
    .s_axis_host_1_tlast(axis_host_1_sink_s.tlast),
    .s_axis_host_1_tready(axis_host_1_sink_s.tready),
    .s_axis_host_1_tvalid(axis_host_1_sink_s.tvalid),

    .s_axis_card_0_tdata(axis_card_0_sink_s.tdata),
    .s_axis_card_0_tkeep(axis_card_0_sink_s.tkeep),
    .s_axis_card_0_tlast(axis_card_0_sink_s.tlast),
    .s_axis_card_0_tready(axis_card_0_sink_s.tready),
    .s_axis_card_0_tvalid(axis_card_0_sink_s.tvalid),

    .s_axis_card_1_tdata(axis_card_1_sink_s.tdata),
    .s_axis_card_1_tkeep(axis_card_1_sink_s.tkeep),
    .s_axis_card_1_tlast(axis_card_1_sink_s.tlast),
    .s_axis_card_1_tready(axis_card_1_sink_s.tready),
    .s_axis_card_1_tvalid(axis_card_1_sink_s.tvalid),

    .s_axis_eth_notification_0_tdata(tcp_0_notify.data),
    .s_axis_eth_notification_0_tkeep(),
    .s_axis_eth_notification_0_tlast(),
    .s_axis_eth_notification_0_tready(tcp_0_notify.ready),
    .s_axis_eth_notification_0_tstrb(),
    .s_axis_eth_notification_0_tvalid(tcp_0_notify.valid),

    .s_axis_eth_rx_data_0_tdata(axis_tcp_0_sink.tdata),
    .s_axis_eth_rx_data_0_tdest(axis_tcp_0_sink.tid), // not parsed within CCLO
    .s_axis_eth_rx_data_0_tkeep(axis_tcp_0_sink.tkeep),
    .s_axis_eth_rx_data_0_tlast(axis_tcp_0_sink.tlast),
    .s_axis_eth_rx_data_0_tready(axis_tcp_0_sink.tready),
    .s_axis_eth_rx_data_0_tvalid(axis_tcp_0_sink.tvalid),

    .s_axis_eth_rx_meta_0_tdata(tcp_0_rx_meta.data),
    .s_axis_eth_rx_meta_0_tkeep(),
    .s_axis_eth_rx_meta_0_tlast(),
    .s_axis_eth_rx_meta_0_tready(tcp_0_rx_meta.ready),
    .s_axis_eth_rx_meta_0_tstrb(),
    .s_axis_eth_rx_meta_0_tvalid(tcp_0_rx_meta.valid),

    .s_axis_eth_tx_status_0_tdata(tcp_0_tx_stat.data),
    .s_axis_eth_tx_status_0_tkeep(),
    .s_axis_eth_tx_status_0_tlast(),
    .s_axis_eth_tx_status_0_tready(tcp_0_tx_stat.ready),
    .s_axis_eth_tx_status_0_tstrb(),
    .s_axis_eth_tx_status_0_tvalid(tcp_0_tx_stat.valid)
);


assign axis_host_0_src_s.tid = 0;
assign axis_host_1_src_s.tid = 0;
assign axis_card_0_src_s.tid = 0;
assign axis_card_1_src_s.tid = 0;

endmodule