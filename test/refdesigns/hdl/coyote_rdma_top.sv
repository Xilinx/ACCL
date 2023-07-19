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
    AXI4SR.s                    axis_host_2_sink,
    AXI4SR.m                    axis_host_2_src,

    // AXI4S CARD STREAMS
    AXI4SR.s                    axis_card_0_sink,
    AXI4SR.m                    axis_card_0_src,
    AXI4SR.s                    axis_card_1_sink,
    AXI4SR.m                    axis_card_1_src,
    AXI4SR.s                    axis_card_2_sink,
    AXI4SR.m                    axis_card_2_src,
    
    // RDMA QSFP0 CMD
    metaIntf.s			        rdma_0_rd_req,
    metaIntf.s 			        rdma_0_wr_req,

    // AXI4S RDMA QSFP0 STREAMS
    AXI4SR.s                    axis_rdma_0_sink,
    AXI4SR.m                    axis_rdma_0_src,

    // RDMA QSFP0 SQ and RQ
    metaIntf.m 			        rdma_0_sq,
    metaIntf.s 			        rdma_0_rq,
    metaIntf.s                  rdma_0_ack,

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
AXI4SR axis_host_2_src_s ();
AXI4SR axis_card_0_src_s ();
AXI4SR axis_card_1_src_s ();
AXI4SR axis_card_2_src_s ();

// register slices
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_0_src_s),  .m_axis(axis_host_0_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_1_src_s),  .m_axis(axis_host_1_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_2_src_s),  .m_axis(axis_host_2_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_0_src_s),  .m_axis(axis_card_0_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_1_src_s),  .m_axis(axis_card_1_src));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_2_src_s),  .m_axis(axis_card_2_src));

// Slave Data Stream
AXI4SR axis_host_0_sink_s ();
AXI4SR axis_host_1_sink_s ();
AXI4SR axis_host_2_sink_s ();
AXI4SR axis_card_0_sink_s ();
AXI4SR axis_card_1_sink_s ();
AXI4SR axis_card_2_sink_s ();

// register slices
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_0_sink),  .m_axis(axis_host_0_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_1_sink),  .m_axis(axis_host_1_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_host_2_sink),  .m_axis(axis_host_2_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_0_sink),  .m_axis(axis_card_0_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_1_sink),  .m_axis(axis_card_1_sink_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(axis_card_2_sink),  .m_axis(axis_card_2_sink_s));


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

    .m_axis_host_2_tdata(axis_host_2_src_s.tdata),
    .m_axis_host_2_tkeep(axis_host_2_src_s.tkeep),
    .m_axis_host_2_tlast(axis_host_2_src_s.tlast),
    .m_axis_host_2_tready(axis_host_2_src_s.tready),
    .m_axis_host_2_tvalid(axis_host_2_src_s.tvalid),
    .m_axis_host_2_tdest(),

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

    .m_axis_card_2_tdata(axis_card_2_src_s.tdata),
    .m_axis_card_2_tkeep(axis_card_2_src_s.tkeep),
    .m_axis_card_2_tlast(axis_card_2_src_s.tlast),
    .m_axis_card_2_tready(axis_card_2_src_s.tready),
    .m_axis_card_2_tvalid(axis_card_2_src_s.tvalid),
    .m_axis_card_2_tdest(),

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

    .s_axis_host_2_tdata(axis_host_2_sink_s.tdata),
    .s_axis_host_2_tkeep(axis_host_2_sink_s.tkeep),
    .s_axis_host_2_tlast(axis_host_2_sink_s.tlast),
    .s_axis_host_2_tready(axis_host_2_sink_s.tready),
    .s_axis_host_2_tvalid(axis_host_2_sink_s.tvalid),

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

    .s_axis_card_2_tdata(axis_card_2_sink_s.tdata),
    .s_axis_card_2_tkeep(axis_card_2_sink_s.tkeep),
    .s_axis_card_2_tlast(axis_card_2_sink_s.tlast),
    .s_axis_card_2_tready(axis_card_2_sink_s.tready),
    .s_axis_card_2_tvalid(axis_card_2_sink_s.tvalid),

    .s_axis_eth_rx_data_tdata(axis_rdma_0_sink.tdata),
    .s_axis_eth_rx_data_tdest(axis_rdma_0_sink.tid),
    .s_axis_eth_rx_data_tkeep(axis_rdma_0_sink.tkeep),
    .s_axis_eth_rx_data_tlast(axis_rdma_0_sink.tlast),
    .s_axis_eth_rx_data_tready(axis_rdma_0_sink.tready),
    .s_axis_eth_rx_data_tvalid(axis_rdma_0_sink.tvalid),

    .m_axis_eth_tx_data_tdata(axis_rdma_0_src.tdata),
    .m_axis_eth_tx_data_tdest(axis_rdma_0_src.tid), // not driven, default 0
    .m_axis_eth_tx_data_tkeep(axis_rdma_0_src.tkeep),
    .m_axis_eth_tx_data_tlast(axis_rdma_0_src.tlast),
    .m_axis_eth_tx_data_tready(axis_rdma_0_src.tready),
    .m_axis_eth_tx_data_tvalid(axis_rdma_0_src.tvalid),

    .s_axis_rdma_wr_req_tdata(rdma_0_wr_req.data),
    .s_axis_rdma_wr_req_tvalid(rdma_0_wr_req.valid),
    .s_axis_rdma_wr_req_tready(rdma_0_wr_req.ready),

    .s_axis_rdma_rd_req_tdata(rdma_0_rd_req.data),
    .s_axis_rdma_rd_req_tvalid(rdma_0_rd_req.valid),
    .s_axis_rdma_rd_req_tready(rdma_0_rd_req.ready),

    .m_axis_rdma_sq_tdata(rdma_0_sq.data),
    .m_axis_rdma_sq_tvalid(rdma_0_sq.valid),
    .m_axis_rdma_sq_tready(rdma_0_sq.ready),
    
    .s_axis_rdma_rq_tdata(rdma_0_rq.data),
    .s_axis_rdma_rq_tvalid(rdma_0_sq.valid),
    .s_axis_rdma_rq_tready(rdma_0_sq.ready)

);


assign axis_host_0_src_s.tid = 0;
assign axis_host_1_src_s.tid = 0;
assign axis_host_2_src_s.tid = 0;

assign axis_card_0_src_s.tid = 0;
assign axis_card_1_src_s.tid = 0;
assign axis_card_2_src_s.tid = 0;

endmodule