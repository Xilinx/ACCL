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

    // NOTIFY
    metaIntf.m                  notify,

    // DESCRIPTORS
    metaIntf.m                  sq_rd, 
    metaIntf.m                  sq_wr,
    metaIntf.s                  cq_rd,
    metaIntf.s                  cq_wr,
    metaIntf.s                  rq_rd,
    metaIntf.s                  rq_wr,

    // HOST DATA STREAMS
    AXI4SR.s                    axis_host_recv [N_STRM_AXI],
    AXI4SR.m                    axis_host_send [N_STRM_AXI],

    // CARD DATA STREAMS
    AXI4SR.s                    axis_card_recv [N_CARD_AXI],
    AXI4SR.m                    axis_card_send [N_CARD_AXI],

    // RDMA DATA STREAMS REQUESTER
    AXI4SR.s                    axis_rreq_recv [N_RDMA_AXI],
    AXI4SR.m                    axis_rreq_send [N_RDMA_AXI],

    // RDMA DATA STREAMS RESPONDER
    AXI4SR.s                    axis_rrsp_recv [N_RDMA_AXI],
    AXI4SR.m                    axis_rrsp_send [N_RDMA_AXI],

    // Clock and reset
    input  wire                 aclk,
    input  wire[0:0]            aresetn
);

/* -- Tie-off unused interfaces and signals ----------------------------- */
always_comb notify.tie_off_m();


/* -- USER LOGIC -------------------------------------------------------- */

// Constants
localparam integer COYOTE_AXIL_ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer COYOTE_AXIL_ADDR_MSB = 16;

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

    .cyt_sq_rd_cmd_tdata(sq_rd.data),
    .cyt_sq_rd_cmd_tready(sq_rd.ready),
    .cyt_sq_rd_cmd_tvalid(sq_rd.valid),

    .cyt_cq_rd_sts_0_tdata(cq_rd.data),
    .cyt_cq_rd_sts_0_tready(cq_rd.ready),
    .cyt_cq_rd_sts_0_tvalid(cq_rd.valid),

    .cyt_sq_wr_cmd_tdata(sq_wr.data),
    .cyt_sq_wr_cmd_tready(sq_wr.ready),
    .cyt_sq_wr_cmd_tvalid(sq_wr.valid),

    .cyt_cq_wr_sts_0_tdata(cq_wr.data),
    .cyt_cq_wr_sts_0_tready(cq_wr.ready),
    .cyt_cq_wr_sts_0_tvalid(cq_wr.valid),

    .cyt_rq_rd_tdata(rq_rd.data),
    .cyt_rq_rd_tready(rq_rd.ready),
    .cyt_rq_rd_tvalid(rq_rd.valid),

    .cyt_rq_wr_tdata(rq_wr.data),
    .cyt_rq_wr_tready(rq_wr.ready),
    .cyt_rq_wr_tvalid(rq_wr.valid),

    .m_axis_host_0_tdata(axis_host_send[0].tdata),
    .m_axis_host_0_tkeep(axis_host_send[0].tkeep),
    .m_axis_host_0_tlast(axis_host_send[0].tlast),
    .m_axis_host_0_tready(axis_host_send[0].tready),
    .m_axis_host_0_tvalid(axis_host_send[0].tvalid),
    .m_axis_host_0_tdest(),

    .m_axis_host_1_tdata(axis_host_send[1].tdata),
    .m_axis_host_1_tkeep(axis_host_send[1].tkeep),
    .m_axis_host_1_tlast(axis_host_send[1].tlast),
    .m_axis_host_1_tready(axis_host_send[1].tready),
    .m_axis_host_1_tvalid(axis_host_send[1].tvalid),
    .m_axis_host_1_tdest(),

    .m_axis_host_2_tdata(axis_host_send[2].tdata),
    .m_axis_host_2_tkeep(axis_host_send[2].tkeep),
    .m_axis_host_2_tlast(axis_host_send[2].tlast),
    .m_axis_host_2_tready(axis_host_send[2].tready),
    .m_axis_host_2_tvalid(axis_host_send[2].tvalid),
    .m_axis_host_2_tdest(),

    .m_axis_card_0_tdata(axis_card_send[0].tdata),
    .m_axis_card_0_tkeep(axis_card_send[0].tkeep),
    .m_axis_card_0_tlast(axis_card_send[0].tlast),
    .m_axis_card_0_tready(axis_card_send[0].tready),
    .m_axis_card_0_tvalid(axis_card_send[0].tvalid),
    .m_axis_card_0_tdest(),

    .m_axis_card_1_tdata(axis_card_send[1].tdata),
    .m_axis_card_1_tkeep(axis_card_send[1].tkeep),
    .m_axis_card_1_tlast(axis_card_send[1].tlast),
    .m_axis_card_1_tready(axis_card_send[1].tready),
    .m_axis_card_1_tvalid(axis_card_send[1].tvalid),
    .m_axis_card_1_tdest(),

    .m_axis_card_2_tdata(axis_card_send[2].tdata),
    .m_axis_card_2_tkeep(axis_card_send[2].tkeep),
    .m_axis_card_2_tlast(axis_card_send[2].tlast),
    .m_axis_card_2_tready(axis_card_send[2].tready),
    .m_axis_card_2_tvalid(axis_card_send[2].tvalid),
    .m_axis_card_2_tdest(),

    .s_axis_host_0_tdata(axis_host_recv[0].tdata),
    .s_axis_host_0_tkeep(axis_host_recv[0].tkeep),
    .s_axis_host_0_tlast(axis_host_recv[0].tlast),
    .s_axis_host_0_tready(axis_host_recv[0].tready),
    .s_axis_host_0_tvalid(axis_host_recv[0].tvalid),

    .s_axis_host_1_tdata(axis_host_recv[1].tdata),
    .s_axis_host_1_tkeep(axis_host_recv[1].tkeep),
    .s_axis_host_1_tlast(axis_host_recv[1].tlast),
    .s_axis_host_1_tready(axis_host_recv[1].tready),
    .s_axis_host_1_tvalid(axis_host_recv[1].tvalid),

    .s_axis_host_2_tdata(axis_host_recv[2].tdata),
    .s_axis_host_2_tkeep(axis_host_recv[2].tkeep),
    .s_axis_host_2_tlast(axis_host_recv[2].tlast),
    .s_axis_host_2_tready(axis_host_recv[2].tready),
    .s_axis_host_2_tvalid(axis_host_recv[2].tvalid),

    .s_axis_card_0_tdata(axis_card_recv[0].tdata),
    .s_axis_card_0_tkeep(axis_card_recv[0].tkeep),
    .s_axis_card_0_tlast(axis_card_recv[0].tlast),
    .s_axis_card_0_tready(axis_card_recv[0].tready),
    .s_axis_card_0_tvalid(axis_card_recv[0].tvalid),

    .s_axis_card_1_tdata(axis_card_recv[1].tdata),
    .s_axis_card_1_tkeep(axis_card_recv[1].tkeep),
    .s_axis_card_1_tlast(axis_card_recv[1].tlast),
    .s_axis_card_1_tready(axis_card_recv[1].tready),
    .s_axis_card_1_tvalid(axis_card_recv[1].tvalid),

    .s_axis_card_2_tdata(axis_card_recv[2].tdata),
    .s_axis_card_2_tkeep(axis_card_recv[2].tkeep),
    .s_axis_card_2_tlast(axis_card_recv[2].tlast),
    .s_axis_card_2_tready(axis_card_recv[2].tready),
    .s_axis_card_2_tvalid(axis_card_recv[2].tvalid),

    .cyt_rreq_recv_0_tdata(axis_rreq_recv[0].tdata),
    .cyt_rreq_recv_0_tkeep(axis_rreq_recv[0].tkeep),
    .cyt_rreq_recv_0_tlast(axis_rreq_recv[0].tlast),
    .cyt_rreq_recv_0_tready(axis_rreq_recv[0].tready),
    .cyt_rreq_recv_0_tvalid(axis_rreq_recv[0].tvalid),

    .cyt_rreq_recv_1_tdata(axis_rreq_recv[1].tdata),
    .cyt_rreq_recv_1_tkeep(axis_rreq_recv[1].tkeep),
    .cyt_rreq_recv_1_tlast(axis_rreq_recv[1].tlast),
    .cyt_rreq_recv_1_tready(axis_rreq_recv[1].tready),
    .cyt_rreq_recv_1_tvalid(axis_rreq_recv[1].tvalid),

    .cyt_rreq_send_0_tdata(axis_rreq_send[0].tdata),
    .cyt_rreq_send_0_tdest(),
    .cyt_rreq_send_0_tkeep(axis_rreq_send[0].tkeep),
    .cyt_rreq_send_0_tlast(axis_rreq_send[0].tlast),
    .cyt_rreq_send_0_tready(axis_rreq_send[0].tready),
    .cyt_rreq_send_0_tstrb(),
    .cyt_rreq_send_0_tvalid(axis_rreq_send[0].tvalid),

    .cyt_rreq_send_1_tdata(axis_rreq_send[1].tdata),
    .cyt_rreq_send_1_tdest(),
    .cyt_rreq_send_1_tkeep(axis_rreq_send[1].tkeep),
    .cyt_rreq_send_1_tlast(axis_rreq_send[1].tlast),
    .cyt_rreq_send_1_tready(axis_rreq_send[1].tready),
    .cyt_rreq_send_1_tstrb(),
    .cyt_rreq_send_1_tvalid(axis_rreq_send[1].tvalid),

    .cyt_rrsp_recv_0_tdata(axis_rrsp_recv[0].tdata),
    .cyt_rrsp_recv_0_tkeep(axis_rrsp_recv[0].tkeep),
    .cyt_rrsp_recv_0_tlast(axis_rrsp_recv[0].tlast),
    .cyt_rrsp_recv_0_tready(axis_rrsp_recv[0].tready),
    .cyt_rrsp_recv_0_tvalid(axis_rrsp_recv[0].tvalid),

    .cyt_rrsp_recv_1_tdata(axis_rrsp_recv[1].tdata),
    .cyt_rrsp_recv_1_tkeep(axis_rrsp_recv[1].tkeep),
    .cyt_rrsp_recv_1_tlast(axis_rrsp_recv[1].tlast),
    .cyt_rrsp_recv_1_tready(axis_rrsp_recv[1].tready),
    .cyt_rrsp_recv_1_tvalid(axis_rrsp_recv[1].tvalid),

    .cyt_rrsp_send_0_tdata(axis_rrsp_send[0].tdata),
    .cyt_rrsp_send_0_tkeep(axis_rrsp_send[0].tkeep),
    .cyt_rrsp_send_0_tlast(axis_rrsp_send[0].tlast),
    .cyt_rrsp_send_0_tready(axis_rrsp_send[0].tready),
    .cyt_rrsp_send_0_tvalid(axis_rrsp_send[0].tvalid),

    .cyt_rrsp_send_1_tdata(axis_rrsp_send[1].tdata),
    .cyt_rrsp_send_1_tkeep(axis_rrsp_send[1].tkeep),
    .cyt_rrsp_send_1_tlast(axis_rrsp_send[1].tlast),
    .cyt_rrsp_send_1_tready(axis_rrsp_send[1].tready),
    .cyt_rrsp_send_1_tvalid(axis_rrsp_send[1].tvalid)

);


// ila_top ila_top(
//     .clk(aclk),
//     .probe0(sq_wr.valid), //1
//     .probe1(sq_wr.ready), //1
//     .probe2(sq_wr.data), //128
//     .probe3(sq_rd.valid), //1
//     .probe4(sq_rd.ready), //1
//     .probe5(sq_rd.data), //128
//     .probe6(rq_wr.valid), //1
//     .probe7(rq_wr.ready), //1
//     .probe8(rq_wr.data), //128
//     .probe9(rq_rd.valid), //1
//     .probe10(rq_rd.ready), //1
//     .probe11(rq_rd.data), //128
//     .probe12(axis_rreq_send[0].tvalid),
//     .probe13(axis_rreq_send[0].tready),
//     .probe14(axis_rreq_send[1].tvalid),
//     .probe15(axis_rreq_send[1].tready),
//     .probe16(axis_rrsp_recv[0].tvalid),
//     .probe17(axis_rrsp_recv[0].tready),
//     .probe18(axis_rrsp_recv[1].tvalid),
//     .probe19(axis_rrsp_recv[1].tready),
//     .probe20(axis_rreq_send[0].tlast),
//     .probe21(axis_rreq_send[1].tlast),
//     .probe22(axis_rrsp_recv[0].tlast),
//     .probe23(axis_rrsp_recv[1].tlast),
//     .probe24(cq_rd.data), //32
//     .probe25(cq_rd.valid),
//     .probe26(cq_rd.ready),
//     .probe27(cq_wr.data), //32
//     .probe28(cq_wr.valid),
//     .probe29(cq_wr.ready),
//     .probe30(axis_host_send[0].tvalid),
//     .probe31(axis_host_send[0].tready),
//     .probe32(axis_host_send[1].tvalid),
//     .probe33(axis_host_send[1].tready),
//     .probe34(axis_host_send[2].tvalid),
//     .probe35(axis_host_send[2].tready),
//     .probe36(axis_card_send[0].tvalid),
//     .probe37(axis_card_send[0].tready),
//     .probe38(axis_card_send[1].tvalid),
//     .probe39(axis_card_send[1].tready),
//     .probe40(axis_card_send[2].tvalid),
//     .probe41(axis_card_send[2].tready),
//     .probe42(axis_host_recv[0].tvalid),
//     .probe43(axis_host_recv[0].tready),
//     .probe44(axis_host_recv[1].tvalid),
//     .probe45(axis_host_recv[1].tready),
//     .probe46(axis_host_recv[2].tvalid),
//     .probe47(axis_host_recv[2].tready),
//     .probe48(axis_card_recv[0].tvalid),
//     .probe49(axis_card_recv[0].tready),
//     .probe50(axis_card_recv[1].tvalid),
//     .probe51(axis_card_recv[1].tready),
//     .probe52(axis_card_recv[2].tvalid),
//     .probe53(axis_card_recv[2].tready)
// );




assign axis_host_send[0].tid = 0;
assign axis_host_send[1].tid = 0;
assign axis_host_send[2].tid = 0;

assign axis_card_send[0].tid = 0;
assign axis_card_send[1].tid = 0;
assign axis_card_send[2].tid = 0;

assign axis_rreq_send[0].tid = 0;
assign axis_rreq_send[1].tid = 0;

assign axis_rrsp_send[0].tid = 0;
assign axis_rrsp_send[1].tid = 0;

endmodule