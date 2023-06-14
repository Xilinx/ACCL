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
    metaIntf.m			        tcp_0_listen_req,
    metaIntf.s			        tcp_0_listen_rsp,
    metaIntf.m			        tcp_0_open_req,
    metaIntf.s			        tcp_0_open_rsp,
    metaIntf.m			        tcp_0_close_req,
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
// always_comb axis_host_0_src.tie_off_m();
// always_comb axis_card_0_sink.tie_off_s();
// always_comb axis_card_0_src.tie_off_m();
// always_comb axis_host_1_sink.tie_off_s();
// always_comb axis_host_1_src.tie_off_m();
// always_comb axis_card_1_sink.tie_off_s();
// always_comb axis_card_1_src.tie_off_m();

/* -- USER LOGIC -------------------------------------------------------- */

// Constants
localparam integer COYOTE_AXIL_ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer COYOTE_AXIL_ADDR_MSB = 16;

// Master Data Stream
AXI4SR m_axis_dma0_s2mm ();
AXI4SR m_axis_dma1_s2mm ();
AXI4SR m_axis_dma0_s2mm_s ();
AXI4SR m_axis_dma1_s2mm_s ();

// register slices
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(m_axis_dma0_s2mm),  .m_axis(m_axis_dma0_s2mm_s));
axisr_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(m_axis_dma1_s2mm),  .m_axis(m_axis_dma1_s2mm_s));

// m_axis_dma0_s2mm_s multiplex to host_0_src and card_0_src according to the strm flag encoded in m_axis_dma0_s2mm_s.tid 
assign axis_host_0_src.tdata = m_axis_dma0_s2mm_s.tdata;
assign axis_host_0_src.tkeep = m_axis_dma0_s2mm_s.tkeep;
assign axis_host_0_src.tlast = m_axis_dma0_s2mm_s.tlast;
assign axis_host_0_src.tid = 0;
assign axis_host_0_src.tvalid = (m_axis_dma0_s2mm_s.tid == 1) ? m_axis_dma0_s2mm_s.tvalid : 1'b0;

assign axis_card_0_src.tdata = m_axis_dma0_s2mm_s.tdata;
assign axis_card_0_src.tkeep = m_axis_dma0_s2mm_s.tkeep;
assign axis_card_0_src.tlast = m_axis_dma0_s2mm_s.tlast;
assign axis_card_0_src.tid = 0;
assign axis_card_0_src.tvalid = (m_axis_dma0_s2mm_s.tid == 0) ? m_axis_dma0_s2mm_s.tvalid : 1'b0;

assign m_axis_dma0_s2mm_s.tready = (m_axis_dma0_s2mm_s.tid == 0) ? axis_card_0_src.tready: axis_host_0_src.tready;

// m_axis_dma1_s2mm_s multiplex to host_1_src and card_1_src according to the strm flag encoded in m_axis_dma1_s2mm_s.tid 
assign axis_host_1_src.tdata = m_axis_dma1_s2mm_s.tdata;
assign axis_host_1_src.tkeep = m_axis_dma1_s2mm_s.tkeep;
assign axis_host_1_src.tlast = m_axis_dma1_s2mm_s.tlast;
assign axis_host_1_src.tid = 0;
assign axis_host_1_src.tvalid = (m_axis_dma1_s2mm_s.tid == 1) ? m_axis_dma1_s2mm_s.tvalid : 1'b0;

assign axis_card_1_src.tdata = m_axis_dma1_s2mm_s.tdata;
assign axis_card_1_src.tkeep = m_axis_dma1_s2mm_s.tkeep;
assign axis_card_1_src.tlast = m_axis_dma1_s2mm_s.tlast;
assign axis_card_1_src.tid = 0;
assign axis_card_1_src.tvalid = (m_axis_dma1_s2mm_s.tid == 0) ? m_axis_dma1_s2mm_s.tvalid : 1'b0;

assign m_axis_dma1_s2mm_s.tready = (m_axis_dma1_s2mm_s.tid == 0) ? axis_card_1_src.tready: axis_host_1_src.tready;


// Slave Data Stream

AXI4S s_axis_dma0_mm2s ();
AXI4S s_axis_dma1_mm2s ();
AXI4S s_axis_dma0_mm2s_s ();
AXI4S s_axis_dma1_mm2s_s ();

// axis_host_0_sink and axis_card_0_sink multiplexed to single s_axis_dma0_mm2s_s stream, round-robin by tlast

axis_interconnect_512_2to1 s_axis_dma0_mm2s_s_merger (
  .ACLK(aclk), // input ACLK
  .ARESETN(aresetn), // input ARESETN
  .S00_AXIS_ACLK(aclk), // input S00_AXIS_ACLK
  .S01_AXIS_ACLK(aclk), // input S01_AXIS_ACLK
  .S00_AXIS_ARESETN(aresetn), // input S00_AXIS_ARESETN
  .S01_AXIS_ARESETN(aresetn), // input S01_AXIS_ARESETN

  .S00_AXIS_TVALID(axis_host_0_sink.tvalid), // input S00_AXIS_TVALID
  .S00_AXIS_TREADY(axis_host_0_sink.tready), // output S00_AXIS_TREADY
  .S00_AXIS_TDATA(axis_host_0_sink.tdata), // input [511 : 0] S00_AXIS_TDATA
  .S00_AXIS_TKEEP(axis_host_0_sink.tkeep), // input [63 : 0] S00_AXIS_TKEEP
  .S00_AXIS_TLAST(axis_host_0_sink.tlast), // input S00_AXIS_TLAST
  
  .S01_AXIS_TVALID(axis_card_0_sink.tvalid), // input S01_AXIS_TVALID
  .S01_AXIS_TREADY(axis_card_0_sink.tready), // output S01_AXIS_TREADY
  .S01_AXIS_TDATA(axis_card_0_sink.tdata), // input [511 : 0] S01_AXIS_TDATA
  .S01_AXIS_TKEEP(axis_card_0_sink.tkeep), // input [63 : 0] S01_AXIS_TKEEP
  .S01_AXIS_TLAST(axis_card_0_sink.tlast), // input S01_AXIS_TLAST
  
  .M00_AXIS_ACLK(aclk), // input M00_AXIS_ACLK
  .M00_AXIS_ARESETN(aresetn), // input M00_AXIS_ARESETN
  .M00_AXIS_TVALID(s_axis_dma0_mm2s_s.tvalid), // output M00_AXIS_TVALID
  .M00_AXIS_TREADY(s_axis_dma0_mm2s_s.tready), // input M00_AXIS_TREADY
  .M00_AXIS_TDATA(s_axis_dma0_mm2s_s.tdata), // output [511 : 0] M00_AXIS_TDATA
  .M00_AXIS_TKEEP(s_axis_dma0_mm2s_s.tkeep), // output [63 : 0] M00_AXIS_TKEEP
  .M00_AXIS_TLAST(s_axis_dma0_mm2s_s.tlast), // output M00_AXIS_TLAST
  .S00_ARB_REQ_SUPPRESS(1'b0), // input S00_ARB_REQ_SUPPRESS
  .S01_ARB_REQ_SUPPRESS(1'b0) // input S01_ARB_REQ_SUPPRESS
);

// axis_host_1_sink and axis_card_1_sink multiplexed to single s_axis_dma1_mm2s_s stream, round-robin by tlast

axis_interconnect_512_2to1 s_axis_dma1_mm2s_s_merger (
  .ACLK(aclk), // input ACLK
  .ARESETN(aresetn), // input ARESETN
  .S00_AXIS_ACLK(aclk), // input S00_AXIS_ACLK
  .S01_AXIS_ACLK(aclk), // input S01_AXIS_ACLK
  .S00_AXIS_ARESETN(aresetn), // input S00_AXIS_ARESETN
  .S01_AXIS_ARESETN(aresetn), // input S01_AXIS_ARESETN

  .S00_AXIS_TVALID(axis_host_1_sink.tvalid), // input S00_AXIS_TVALID
  .S00_AXIS_TREADY(axis_host_1_sink.tready), // output S00_AXIS_TREADY
  .S00_AXIS_TDATA(axis_host_1_sink.tdata), // input [511 : 0] S00_AXIS_TDATA
  .S00_AXIS_TKEEP(axis_host_1_sink.tkeep), // input [63 : 0] S00_AXIS_TKEEP
  .S00_AXIS_TLAST(axis_host_1_sink.tlast), // input S00_AXIS_TLAST
  
  .S01_AXIS_TVALID(axis_card_1_sink.tvalid), // input S01_AXIS_TVALID
  .S01_AXIS_TREADY(axis_card_1_sink.tready), // output S01_AXIS_TREADY
  .S01_AXIS_TDATA(axis_card_1_sink.tdata), // input [511 : 0] S01_AXIS_TDATA
  .S01_AXIS_TKEEP(axis_card_1_sink.tkeep), // input [63 : 0] S01_AXIS_TKEEP
  .S01_AXIS_TLAST(axis_card_1_sink.tlast), // input S01_AXIS_TLAST
  
  .M00_AXIS_ACLK(aclk), // input M00_AXIS_ACLK
  .M00_AXIS_ARESETN(aresetn), // input M00_AXIS_ARESETN
  .M00_AXIS_TVALID(s_axis_dma1_mm2s_s.tvalid), // output M00_AXIS_TVALID
  .M00_AXIS_TREADY(s_axis_dma1_mm2s_s.tready), // input M00_AXIS_TREADY
  .M00_AXIS_TDATA(s_axis_dma1_mm2s_s.tdata), // output [511 : 0] M00_AXIS_TDATA
  .M00_AXIS_TKEEP(s_axis_dma1_mm2s_s.tkeep), // output [63 : 0] M00_AXIS_TKEEP
  .M00_AXIS_TLAST(s_axis_dma1_mm2s_s.tlast), // output M00_AXIS_TLAST
  .S00_ARB_REQ_SUPPRESS(1'b0), // input S00_ARB_REQ_SUPPRESS
  .S01_ARB_REQ_SUPPRESS(1'b0) // input S01_ARB_REQ_SUPPRESS
);

// slices
axis_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(s_axis_dma0_mm2s_s),  .m_axis(s_axis_dma0_mm2s));
axis_reg_array #(.N_STAGES(4)) (.aclk(aclk), .aresetn(aresetn), .s_axis(s_axis_dma1_mm2s_s),  .m_axis(s_axis_dma1_mm2s));


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

    .m_axis_dma0_s2mm_0_tdata(m_axis_dma0_s2mm.tdata),
    .m_axis_dma0_s2mm_0_tkeep(m_axis_dma0_s2mm.tkeep),
    .m_axis_dma0_s2mm_0_tlast(m_axis_dma0_s2mm.tlast),
    .m_axis_dma0_s2mm_0_tready(m_axis_dma0_s2mm.tready),
    .m_axis_dma0_s2mm_0_tvalid(m_axis_dma0_s2mm.tvalid),
    .m_axis_dma0_s2mm_0_tdest(m_axis_dma0_s2mm.tid),

    .m_axis_dma1_s2mm_0_tdata(m_axis_dma1_s2mm.tdata),
    .m_axis_dma1_s2mm_0_tkeep(m_axis_dma1_s2mm.tkeep),
    .m_axis_dma1_s2mm_0_tlast(m_axis_dma1_s2mm.tlast),
    .m_axis_dma1_s2mm_0_tready(m_axis_dma1_s2mm.tready),
    .m_axis_dma1_s2mm_0_tvalid(m_axis_dma1_s2mm.tvalid),
    .m_axis_dma1_s2mm_0_tdest(m_axis_dma1_s2mm.tid),

    .m_axis_eth_close_connection_0_tdata(tcp_0_close_req.data),
    .m_axis_eth_close_connection_0_tkeep(),
    .m_axis_eth_close_connection_0_tlast(),
    .m_axis_eth_close_connection_0_tready(tcp_0_close_req.ready),
    .m_axis_eth_close_connection_0_tstrb(),
    .m_axis_eth_close_connection_0_tvalid(tcp_0_close_req.valid),

    .m_axis_eth_listen_port_0_tdata(tcp_0_listen_req.data),
    .m_axis_eth_listen_port_0_tkeep(),
    .m_axis_eth_listen_port_0_tlast(),
    .m_axis_eth_listen_port_0_tready(tcp_0_listen_req.ready),
    .m_axis_eth_listen_port_0_tstrb(),
    .m_axis_eth_listen_port_0_tvalid(tcp_0_listen_req.valid),

    .m_axis_eth_open_connection_0_tdata(tcp_0_open_req.data),
    .m_axis_eth_open_connection_0_tkeep(),
    .m_axis_eth_open_connection_0_tlast(),
    .m_axis_eth_open_connection_0_tready(tcp_0_open_req.ready),
    .m_axis_eth_open_connection_0_tstrb(),
    .m_axis_eth_open_connection_0_tvalid(tcp_0_open_req.valid),

    .m_axis_eth_read_pkg_0_tdata(tcp_0_rd_pkg.data),
    .m_axis_eth_read_pkg_0_tkeep(),
    .m_axis_eth_read_pkg_0_tlast(),
    .m_axis_eth_read_pkg_0_tready(tcp_0_rd_pkg.ready),
    .m_axis_eth_read_pkg_0_tstrb(),
    .m_axis_eth_read_pkg_0_tvalid(tcp_0_rd_pkg.valid),

    .m_axis_eth_tx_data_0_tdata(axis_tcp_0_src.tdata),
    .m_axis_eth_tx_data_0_tdest(axis_tcp_0_src.tid),
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

    .s_axis_dma0_mm2s_0_tdata(s_axis_dma0_mm2s.tdata),
    .s_axis_dma0_mm2s_0_tkeep(s_axis_dma0_mm2s.tkeep),
    .s_axis_dma0_mm2s_0_tlast(s_axis_dma0_mm2s.tlast),
    .s_axis_dma0_mm2s_0_tready(s_axis_dma0_mm2s.tready),
    .s_axis_dma0_mm2s_0_tvalid(s_axis_dma0_mm2s.tvalid),

    .s_axis_dma1_mm2s_0_tdata(s_axis_dma1_mm2s.tdata),
    .s_axis_dma1_mm2s_0_tkeep(s_axis_dma1_mm2s.tkeep),
    .s_axis_dma1_mm2s_0_tlast(s_axis_dma1_mm2s.tlast),
    .s_axis_dma1_mm2s_0_tready(s_axis_dma1_mm2s.tready),
    .s_axis_dma1_mm2s_0_tvalid(s_axis_dma1_mm2s.tvalid),

    .s_axis_eth_notification_0_tdata(tcp_0_notify.data),
    .s_axis_eth_notification_0_tkeep(),
    .s_axis_eth_notification_0_tlast(),
    .s_axis_eth_notification_0_tready(tcp_0_notify.ready),
    .s_axis_eth_notification_0_tstrb(),
    .s_axis_eth_notification_0_tvalid(tcp_0_notify.valid),

    .s_axis_eth_open_status_0_tdata(tcp_0_open_rsp.data),
    .s_axis_eth_open_status_0_tkeep(),
    .s_axis_eth_open_status_0_tlast(),
    .s_axis_eth_open_status_0_tready(tcp_0_open_rsp.ready),
    .s_axis_eth_open_status_0_tstrb(),
    .s_axis_eth_open_status_0_tvalid(tcp_0_open_rsp.valid),

    .s_axis_eth_port_status_0_tdata(tcp_0_listen_rsp.data),
    .s_axis_eth_port_status_0_tkeep(),
    .s_axis_eth_port_status_0_tlast(),
    .s_axis_eth_port_status_0_tready(tcp_0_listen_rsp.ready),
    .s_axis_eth_port_status_0_tstrb(),
    .s_axis_eth_port_status_0_tvalid(tcp_0_listen_rsp.valid),

    .s_axis_eth_rx_data_0_tdata(axis_tcp_0_sink.tdata),
    .s_axis_eth_rx_data_0_tdest(axis_tcp_0_sink.tid),
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


endmodule