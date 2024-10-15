/**
  * Copyright (c) 2021, Systems Group, ETH Zurich
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

`timescale 1ns / 1ps

import lynxTypes::*;

`include "axi_macros.svh"
`include "lynx_macros.svh"

/**
 * @brief   TCP clock crossing
 *
 */
module tcp_ccross (
    // Network
    // metaIntf.m              m_tcp_listen_req_nclk,
    // metaIntf.s              s_tcp_listen_rsp_nclk,
    // metaIntf.m              m_tcp_open_req_nclk,
    // metaIntf.s              s_tcp_open_rsp_nclk,
    // metaIntf.m              m_tcp_close_req_nclk,
    metaIntf.s              s_tcp_notify_nclk,
    metaIntf.m              m_tcp_rd_pkg_nclk,
    metaIntf.s              s_tcp_rx_meta_nclk,
    metaIntf.m              m_tcp_tx_meta_nclk,
    metaIntf.s              s_tcp_tx_stat_nclk,
    AXI4S.m                 m_axis_tcp_tx_nclk, 
    AXI4S.s                 s_axis_tcp_rx_nclk,
    
    // User        
    // metaIntf.s              s_tcp_listen_req_aclk,
    // metaIntf.m              m_tcp_listen_rsp_aclk,
    // metaIntf.s              s_tcp_open_req_aclk,
    // metaIntf.m              m_tcp_open_rsp_aclk,
    // metaIntf.s              s_tcp_close_req_aclk,
    metaIntf.m              m_tcp_notify_aclk,
    metaIntf.s              s_tcp_rd_pkg_aclk,
    metaIntf.m              m_tcp_rx_meta_aclk,
    metaIntf.s              s_tcp_tx_meta_aclk,
    metaIntf.m              m_tcp_tx_stat_aclk,
    AXI4S.s                 s_axis_tcp_tx_aclk,      
    AXI4S.m                 m_axis_tcp_rx_aclk,

    input  wire             nclk,
    input  wire             nresetn,
    input  wire             aclk,
    input  wire             aresetn
);

// ---------------------------------------------------------------------------------------------------
// Crossings
// ---------------------------------------------------------------------------------------------------

    // // Port request and responses
    // axis_clock_converter_tcp_16 inst_ccross_tcp_listen_req (
    //     .m_axis_aclk(nclk),
    //     .s_axis_aclk(aclk),
    //     .s_axis_aresetn(aresetn),
    //     .m_axis_aresetn(nresetn),
    //     .s_axis_tvalid(s_tcp_listen_req_aclk.valid),
    //     .s_axis_tready(s_tcp_listen_req_aclk.ready),
    //     .s_axis_tdata (s_tcp_listen_req_aclk.data),
    //     .m_axis_tvalid(m_tcp_listen_req_nclk.valid),
    //     .m_axis_tready(m_tcp_listen_req_nclk.ready),
    //     .m_axis_tdata (m_tcp_listen_req_nclk.data)
    // );

    // axis_clock_converter_tcp_8 inst_tcp_listen_rsp (
    //     .m_axis_aclk(aclk),
    //     .s_axis_aclk(nclk),
    //     .s_axis_aresetn(nresetn),
    //     .m_axis_aresetn(aresetn),
    //     .s_axis_tvalid(s_tcp_listen_rsp_nclk.valid),
    //     .s_axis_tready(s_tcp_listen_rsp_nclk.ready),
    //     .s_axis_tdata (s_tcp_listen_rsp_nclk.data),
    //     .m_axis_tvalid(m_tcp_listen_rsp_aclk.valid),
    //     .m_axis_tready(m_tcp_listen_rsp_aclk.ready),
    //     .m_axis_tdata (m_tcp_listen_rsp_aclk.data)
    // );

    // // Open, close requests and responses
    // axis_clock_converter_tcp_48 inst_tcp_open_req (
    //     .m_axis_aclk(nclk),
    //     .s_axis_aclk(aclk),
    //     .s_axis_aresetn(aresetn),
    //     .m_axis_aresetn(nresetn),
    //     .s_axis_tvalid(s_tcp_open_req_aclk.valid),
    //     .s_axis_tready(s_tcp_open_req_aclk.ready),
    //     .s_axis_tdata (s_tcp_open_req_aclk.data),
    //     .m_axis_tvalid(m_tcp_open_req_nclk.valid),
    //     .m_axis_tready(m_tcp_open_req_nclk.ready),
    //     .m_axis_tdata (m_tcp_open_req_nclk.data)
    // );

    // axis_clock_converter_tcp_72 inst_tcp_open_rsp (
    //     .m_axis_aclk(aclk),
    //     .s_axis_aclk(nclk),
    //     .s_axis_aresetn(nresetn),
    //     .m_axis_aresetn(aresetn),
    //     .s_axis_tvalid(s_tcp_open_rsp_nclk.valid),
    //     .s_axis_tready(s_tcp_open_rsp_nclk.ready),
    //     .s_axis_tdata (s_tcp_open_rsp_nclk.data),
    //     .m_axis_tvalid(m_tcp_open_rsp_aclk.valid),
    //     .m_axis_tready(m_tcp_open_rsp_aclk.ready),
    //     .m_axis_tdata (m_tcp_open_rsp_aclk.data)
    // );

    // axis_clock_converter_tcp_16 inst_tcp_close_req (
    //     .m_axis_aclk(nclk),
    //     .s_axis_aclk(aclk),
    //     .s_axis_aresetn(aresetn),
    //     .m_axis_aresetn(nresetn),
    //     .s_axis_tvalid(s_tcp_close_req_aclk.valid),
    //     .s_axis_tready(s_tcp_close_req_aclk.ready),
    //     .s_axis_tdata (s_tcp_close_req_aclk.data),
    //     .m_axis_tvalid(m_tcp_close_req_nclk.valid),
    //     .m_axis_tready(m_tcp_close_req_nclk.ready),
    //     .m_axis_tdata (m_tcp_close_req_nclk.data)
    // );

    // Notifications
    axis_clock_converter_tcp_88 inst_tcp_notify (
        .m_axis_aclk(aclk),
        .s_axis_aclk(nclk),
        .s_axis_aresetn(nresetn),
        .m_axis_aresetn(aresetn),
        .s_axis_tvalid(s_tcp_notify_nclk.valid),
        .s_axis_tready(s_tcp_notify_nclk.ready),
        .s_axis_tdata (s_tcp_notify_nclk.data),
        .m_axis_tvalid(m_tcp_notify_aclk.valid),
        .m_axis_tready(m_tcp_notify_aclk.ready),
        .m_axis_tdata (m_tcp_notify_aclk.data)
    );

    // Read pkg
    axis_clock_converter_tcp_40 inst_rd_pkg (
        .m_axis_aclk(nclk),
        .s_axis_aclk(aclk),
        .s_axis_aresetn(aresetn),
        .m_axis_aresetn(nresetn),
        .s_axis_tvalid(s_tcp_rd_pkg_aclk.valid),
        .s_axis_tready(s_tcp_rd_pkg_aclk.ready),
        .s_axis_tdata (s_tcp_rd_pkg_aclk.data),
        .m_axis_tvalid(m_tcp_rd_pkg_nclk.valid),
        .m_axis_tready(m_tcp_rd_pkg_nclk.ready),
        .m_axis_tdata (m_tcp_rd_pkg_nclk.data)
    );

    // Read meta + data
    axis_clock_converter_tcp_16 inst_rx_meta (
        .m_axis_aclk(aclk),
        .s_axis_aclk(nclk),
        .s_axis_aresetn(nresetn),
        .m_axis_aresetn(aresetn),
        .s_axis_tvalid(s_tcp_rx_meta_nclk.valid),
        .s_axis_tready(s_tcp_rx_meta_nclk.ready),
        .s_axis_tdata (s_tcp_rx_meta_nclk.data),
        .m_axis_tvalid(m_tcp_rx_meta_aclk.valid),
        .m_axis_tready(m_tcp_rx_meta_aclk.ready),
        .m_axis_tdata (m_tcp_rx_meta_aclk.data)
    );

    // Write meta + data + status
    axis_clock_converter_tcp_40 inst_tcp_tx_meta (
        .m_axis_aclk(nclk),
        .s_axis_aclk(aclk),
        .s_axis_aresetn(aresetn),
        .m_axis_aresetn(nresetn),
        .s_axis_tvalid(s_tcp_tx_meta_aclk.valid),
        .s_axis_tready(s_tcp_tx_meta_aclk.ready),
        .s_axis_tdata (s_tcp_tx_meta_aclk.data),
        .m_axis_tvalid(m_tcp_tx_meta_nclk.valid),
        .m_axis_tready(m_tcp_tx_meta_nclk.ready),
        .m_axis_tdata (m_tcp_tx_meta_nclk.data)
    );

    axis_clock_converter_tcp_64 inst_tcp_tx_stat (
        .m_axis_aclk(aclk),
        .s_axis_aclk(nclk),
        .s_axis_aresetn(nresetn),
        .m_axis_aresetn(aresetn),
        .s_axis_tvalid(s_tcp_tx_stat_nclk.valid),
        .s_axis_tready(s_tcp_tx_stat_nclk.ready),
        .s_axis_tdata (s_tcp_tx_stat_nclk.data),
        .m_axis_tvalid(m_tcp_tx_stat_aclk.valid),
        .m_axis_tready(m_tcp_tx_stat_aclk.ready),
        .m_axis_tdata (m_tcp_tx_stat_aclk.data)
    );

    axis_clock_converter_tcp_512 inst_tcp_tx_data (
        .m_axis_aclk(nclk),
        .s_axis_aclk(aclk),
        .s_axis_aresetn(aresetn),
        .m_axis_aresetn(nresetn),
        .s_axis_tvalid(s_axis_tcp_tx_aclk.tvalid),
        .s_axis_tready(s_axis_tcp_tx_aclk.tready),
        .s_axis_tdata (s_axis_tcp_tx_aclk.tdata),
        .s_axis_tkeep (s_axis_tcp_tx_aclk.tkeep),
        .s_axis_tlast (s_axis_tcp_tx_aclk.tlast),
        .m_axis_tvalid(m_axis_tcp_tx_nclk.tvalid),
        .m_axis_tready(m_axis_tcp_tx_nclk.tready),
        .m_axis_tdata (m_axis_tcp_tx_nclk.tdata),
        .m_axis_tkeep (m_axis_tcp_tx_nclk.tkeep),
        .m_axis_tlast (m_axis_tcp_tx_nclk.tlast)
    );

    axis_clock_converter_tcp_512 inst_rx_data (
        .m_axis_aclk(aclk),
        .s_axis_aclk(nclk),
        .s_axis_aresetn(nresetn),
        .m_axis_aresetn(aresetn),
        .s_axis_tvalid(s_axis_tcp_rx_nclk.tvalid),
        .s_axis_tready(s_axis_tcp_rx_nclk.tready),
        .s_axis_tdata (s_axis_tcp_rx_nclk.tdata),
        .s_axis_tkeep (s_axis_tcp_rx_nclk.tkeep),
        .s_axis_tlast (s_axis_tcp_rx_nclk.tlast),
        .m_axis_tvalid(m_axis_tcp_rx_aclk.tvalid),
        .m_axis_tready(m_axis_tcp_rx_aclk.tready),
        .m_axis_tdata (m_axis_tcp_rx_aclk.tdata),
        .m_axis_tkeep (m_axis_tcp_rx_aclk.tkeep),
        .m_axis_tlast (m_axis_tcp_rx_aclk.tlast)
    );

endmodule