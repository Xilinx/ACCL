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
    AXI4SR.s                    axis_host_sink,
    AXI4SR.m                    axis_host_src,

    // AXI4S CARD STREAMS
    AXI4SR.s                    axis_card_sink,
    AXI4SR.m                    axis_card_src,
    
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
//always_comb axis_host_sink.tie_off_s();
//always_comb axis_host_src.tie_off_m();
//always_comb axis_card_sink.tie_off_s();
//always_comb axis_card_src.tie_off_m();

/* -- USER LOGIC -------------------------------------------------------- */

accl_bd_wrapper accl_system(
    .ap_clk_0(aclk),
    .ap_rst_n_0(aresetn),

    .S00_AXI_0_araddr(axi_ctrl.araddr),
    .S00_AXI_0_arburst(axi_ctrl.arburst),
    .S00_AXI_0_arcache(axi_ctrl.arcache),
    .S00_AXI_0_arlen(axi_ctrl.arlen),
    .S00_AXI_0_arlock(axi_ctrl.arlock),
    .S00_AXI_0_arprot(axi_ctrl.arprot),
    .S00_AXI_0_arqos(axi_ctrl.arqos),
    .S00_AXI_0_arready(axi_ctrl.arready),
    .S00_AXI_0_arsize(axi_ctrl.arsize),
    .S00_AXI_0_arvalid(axi_ctrl.arvalid),
    .S00_AXI_0_awaddr(axi_ctrl.awaddr),
    .S00_AXI_0_awburst(axi_ctrl.awburst),
    .S00_AXI_0_awcache(axi_ctrl.awcache),
    .S00_AXI_0_awlen(axi_ctrl.awlen),
    .S00_AXI_0_awlock(axi_ctrl.awlock),
    .S00_AXI_0_awprot(axi_ctrl.awprot),
    .S00_AXI_0_awqos(axi_ctrl.awqos),
    .S00_AXI_0_awready(axi_ctrl.awready),
    .S00_AXI_0_awsize(axi_ctrl.awsize),
    .S00_AXI_0_awvalid(axi_ctrl.awvalid),
    .S00_AXI_0_bready(axi_ctrl.bready),
    .S00_AXI_0_bresp(axi_ctrl.bresp),
    .S00_AXI_0_bvalid(axi_ctrl.bvalid),
    .S00_AXI_0_rdata(axi_ctrl.rdata),
    .S00_AXI_0_rlast(axi_ctrl.rlast),
    .S00_AXI_0_rready(axi_ctrl.rready),
    .S00_AXI_0_rresp(axi_ctrl.rresp),
    .S00_AXI_0_rvalid(axi_ctrl.rvalid),
    .S00_AXI_0_wdata(axi_ctrl.wdata),
    .S00_AXI_0_wlast(axi_ctrl.wlast),
    .S00_AXI_0_wready(axi_ctrl.wready),
    .S00_AXI_0_wstrb(axi_ctrl.wstrb),
    .S00_AXI_0_wvalid(axi_ctrl.wvalid),

    .cyt_byp_rd_cmd_0_tdata(bpss_rd_req.tdata),
    .cyt_byp_rd_cmd_0_tready(bpss_rd_req.tready),
    .cyt_byp_rd_cmd_0_tvalid(bpss_rd_req.tvalid),

    .cyt_byp_rd_sts_0_tdata(bpss_rd_done.tdata),
    .cyt_byp_rd_sts_0_tready(bpss_rd_done.tready),
    .cyt_byp_rd_sts_0_tvalid(bpss_rd_done.tvalid),

    .cyt_byp_wr_cmd_0_tdata(bpss_wr_req.tdata),
    .cyt_byp_wr_cmd_0_tready(bpss_wr_req.tready),
    .cyt_byp_wr_cmd_0_tvalid(bpss_wr_req.tvalid),

    .cyt_byp_wr_sts_0_tdata(bpss_wr_done.tdata),
    .cyt_byp_wr_sts_0_tready(bpss_wr_done.tready),
    .cyt_byp_wr_sts_0_tvalid(bpss_wr_done.tvalid),

    .m_axis_dma0_s2mm_0_tdata(),
    .m_axis_dma0_s2mm_0_tkeep(),
    .m_axis_dma0_s2mm_0_tlast(),
    .m_axis_dma0_s2mm_0_tready(),
    .m_axis_dma0_s2mm_0_tvalid(),

    .m_axis_dma1_s2mm_0_tdata(),
    .m_axis_dma1_s2mm_0_tkeep(),
    .m_axis_dma1_s2mm_0_tlast(),
    .m_axis_dma1_s2mm_0_tready(),
    .m_axis_dma1_s2mm_0_tvalid(),

    .m_axis_eth_close_connection_0_tdata(tcp_0_close_req.tdata),
    .m_axis_eth_close_connection_0_tkeep(tcp_0_close_req.tkeep),
    .m_axis_eth_close_connection_0_tlast(tcp_0_close_req.tlast),
    .m_axis_eth_close_connection_0_tready(tcp_0_close_req.tready),
    .m_axis_eth_close_connection_0_tstrb(tcp_0_close_req.tstrb),
    .m_axis_eth_close_connection_0_tvalid(tcp_0_close_req.tvalid),

    .m_axis_eth_listen_port_0_tdata(tcp_0_listen_req.tdata),
    .m_axis_eth_listen_port_0_tkeep(tcp_0_listen_req.tkeep),
    .m_axis_eth_listen_port_0_tlast(tcp_0_listen_req.tlast),
    .m_axis_eth_listen_port_0_tready(tcp_0_listen_req.tready),
    .m_axis_eth_listen_port_0_tstrb(tcp_0_listen_req.tstrb),
    .m_axis_eth_listen_port_0_tvalid(tcp_0_listen_req.tvalid),

    .m_axis_eth_open_connection_0_tdata(tcp_0_open_req.tdata),
    .m_axis_eth_open_connection_0_tkeep(tcp_0_open_req.tkeep),
    .m_axis_eth_open_connection_0_tlast(tcp_0_open_req.tlast),
    .m_axis_eth_open_connection_0_tready(tcp_0_open_req.tready),
    .m_axis_eth_open_connection_0_tstrb(tcp_0_open_req.tstrb),
    .m_axis_eth_open_connection_0_tvalid(tcp_0_open_req.tvalid),

    .m_axis_eth_read_pkg_0_tdata(tcp_0_rd_pkg.tdata),
    .m_axis_eth_read_pkg_0_tkeep(tcp_0_rd_pkg.tkeep),
    .m_axis_eth_read_pkg_0_tlast(tcp_0_rd_pkg.tlast),
    .m_axis_eth_read_pkg_0_tready(tcp_0_rd_pkg.tready),
    .m_axis_eth_read_pkg_0_tstrb(tcp_0_rd_pkg.tstrb),
    .m_axis_eth_read_pkg_0_tvalid(tcp_0_rd_pkg.tvalid),

    .m_axis_eth_tx_data_0_tdata(axis_tcp_0_src.tdata),
    .m_axis_eth_tx_data_0_tdest(axis_tcp_0_src.tdest),
    .m_axis_eth_tx_data_0_tkeep(axis_tcp_0_src.tkeep),
    .m_axis_eth_tx_data_0_tlast(axis_tcp_0_src.tlast),
    .m_axis_eth_tx_data_0_tready(axis_tcp_0_src.tready),
    .m_axis_eth_tx_data_0_tvalid(axis_tcp_0_src.tvalid),

    .m_axis_eth_tx_meta_0_tdata(tcp_0_tx_meta.tdata),
    .m_axis_eth_tx_meta_0_tkeep(tcp_0_tx_meta.tkeep),
    .m_axis_eth_tx_meta_0_tlast(tcp_0_tx_meta.tlast),
    .m_axis_eth_tx_meta_0_tready(tcp_0_tx_meta.tready),
    .m_axis_eth_tx_meta_0_tstrb(tcp_0_tx_meta.tstrb),
    .m_axis_eth_tx_meta_0_tvalid(tcp_0_tx_meta.tvalid),

    .s_axis_dma0_mm2s_0_tdata(),
    .s_axis_dma0_mm2s_0_tkeep(),
    .s_axis_dma0_mm2s_0_tlast(),
    .s_axis_dma0_mm2s_0_tready(),
    .s_axis_dma0_mm2s_0_tvalid(),

    .s_axis_dma1_mm2s_0_tdata(),
    .s_axis_dma1_mm2s_0_tkeep(),
    .s_axis_dma1_mm2s_0_tlast(),
    .s_axis_dma1_mm2s_0_tready(),
    .s_axis_dma1_mm2s_0_tvalid(),

    .s_axis_eth_notification_0_tdata(tcp_0_notify.tdata),
    .s_axis_eth_notification_0_tkeep(tcp_0_notify.tkeep),
    .s_axis_eth_notification_0_tlast(tcp_0_notify.tlast),
    .s_axis_eth_notification_0_tready(tcp_0_notify.tready),
    .s_axis_eth_notification_0_tstrb(tcp_0_notify.tstrb),
    .s_axis_eth_notification_0_tvalid(tcp_0_notify.tvalid),

    .s_axis_eth_open_status_0_tdata(tcp_0_open_rsp.tdata),
    .s_axis_eth_open_status_0_tkeep(tcp_0_open_rsp.tkeep),
    .s_axis_eth_open_status_0_tlast(tcp_0_open_rsp.tlast),
    .s_axis_eth_open_status_0_tready(tcp_0_open_rsp.tready),
    .s_axis_eth_open_status_0_tstrb(tcp_0_open_rsp.tstrb),
    .s_axis_eth_open_status_0_tvalid(tcp_0_open_rsp.tvalid),

    .s_axis_eth_port_status_0_tdata(tcp_0_listen_rsp.tdata),
    .s_axis_eth_port_status_0_tkeep(tcp_0_listen_rsp.tkeep),
    .s_axis_eth_port_status_0_tlast(tcp_0_listen_rsp.tlast),
    .s_axis_eth_port_status_0_tready(tcp_0_listen_rsp.tready),
    .s_axis_eth_port_status_0_tstrb(tcp_0_listen_rsp.tstrb),
    .s_axis_eth_port_status_0_tvalid(tcp_0_listen_rsp.tvalid),

    .s_axis_eth_rx_data_0_tdata(axis_tcp_0_sink.tdata),
    .s_axis_eth_rx_data_0_tdest(axis_tcp_0_sink.tdest),
    .s_axis_eth_rx_data_0_tkeep(axis_tcp_0_sink.tkeep),
    .s_axis_eth_rx_data_0_tlast(axis_tcp_0_sink.tlast),
    .s_axis_eth_rx_data_0_tready(axis_tcp_0_sink.tready),
    .s_axis_eth_rx_data_0_tvalid(axis_tcp_0_sink.tvalid),

    .s_axis_eth_rx_meta_0_tdata(tcp_0_rx_meta.tdata),
    .s_axis_eth_rx_meta_0_tkeep(tcp_0_rx_meta.tkeep),
    .s_axis_eth_rx_meta_0_tlast(tcp_0_rx_meta.tlast),
    .s_axis_eth_rx_meta_0_tready(tcp_0_rx_meta.tready),
    .s_axis_eth_rx_meta_0_tstrb(tcp_0_rx_meta.tstrb),
    .s_axis_eth_rx_meta_0_tvalid(tcp_0_rx_meta.tvalid),

    .s_axis_eth_tx_status_0_tdata(tcp_0_tx_stat.tdata),
    .s_axis_eth_tx_status_0_tkeep(tcp_0_tx_stat.tkeep),
    .s_axis_eth_tx_status_0_tlast(tcp_0_tx_stat.tlast),
    .s_axis_eth_tx_status_0_tready(tcp_0_tx_stat.tready),
    .s_axis_eth_tx_status_0_tstrb(tcp_0_tx_stat.tstrb),
    .s_axis_eth_tx_status_0_tvali(tcp_0_tx_stat.tvali)
);


endmodule