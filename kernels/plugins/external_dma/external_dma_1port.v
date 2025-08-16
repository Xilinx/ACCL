

`timescale 1 ns / 1 ps

module external_dma_1port
(
  input ap_clk,
  input ap_rst_n,

  input [15:0] s_axi_control_araddr,
  output s_axi_control_arready,
  input s_axi_control_arvalid,
  input [15:0] s_axi_control_awaddr,
  output s_axi_control_awready,
  input s_axi_control_awvalid,
  input s_axi_control_bready,
  output [1:0] s_axi_control_bresp,
  output s_axi_control_bvalid,
  output [31:0] s_axi_control_rdata,
  input s_axi_control_rready,
  output [1:0] s_axi_control_rresp,
  output s_axi_control_rvalid,
  input [31:0] s_axi_control_wdata,
  output s_axi_control_wready,
  input [3:0] s_axi_control_wstrb,
  input s_axi_control_wvalid,

  
  output [63:0] m_axi_0_araddr,
  output [1:0] m_axi_0_arburst,
  output [3:0] m_axi_0_arcache,
  output [7:0] m_axi_0_arlen,
  output [2:0] m_axi_0_arprot,
  input m_axi_0_arready,
  output [2:0] m_axi_0_arsize,
  output [3:0] m_axi_0_aruser,
  output m_axi_0_arvalid,
  output [63:0] m_axi_0_awaddr,
  output [1:0] m_axi_0_awburst,
  output [3:0] m_axi_0_awcache,
  output [7:0] m_axi_0_awlen,
  output [2:0] m_axi_0_awprot,
  input m_axi_0_awready,
  output [2:0] m_axi_0_awsize,
  output [3:0] m_axi_0_awuser,
  output m_axi_0_awvalid,
  output m_axi_0_bready,
  input [1:0] m_axi_0_bresp,
  input m_axi_0_bvalid,
  input [511:0] m_axi_0_rdata,
  input m_axi_0_rlast,
  output m_axi_0_rready,
  input [1:0] m_axi_0_rresp,
  input m_axi_0_rvalid,
  output [511:0] m_axi_0_wdata,
  output m_axi_0_wlast,
  input m_axi_0_wready,
  output [63:0] m_axi_0_wstrb,
  output m_axi_0_wvalid,



  input [511:0] s_axis_s2mm_tdata,
  input [63:0] s_axis_s2mm_tkeep,
  input [7:0] s_axis_s2mm_tdest,
  input s_axis_s2mm_tlast,
  output s_axis_s2mm_tready,
  input s_axis_s2mm_tvalid,

  output [511:0] m_axis_mm2s_tdata,
  output [63:0] m_axis_mm2s_tkeep,
  output m_axis_mm2s_tlast,
  input m_axis_mm2s_tready,
  output m_axis_mm2s_tvalid,

  input [103:0] s_axis_mm2s_cmd_tdata,
  output s_axis_mm2s_cmd_tready,
  input s_axis_mm2s_cmd_tvalid,
  input [7:0] s_axis_mm2s_cmd_tdest,

  output [7:0] m_axis_mm2s_sts_tdata,
  input m_axis_mm2s_sts_tready,
  output m_axis_mm2s_sts_tvalid,
  output [0:0] m_axis_mm2s_sts_tkeep,
  output m_axis_mm2s_sts_tlast,

  input [103:0] s_axis_s2mm_cmd_tdata,
  output s_axis_s2mm_cmd_tready,
  input s_axis_s2mm_cmd_tvalid,
  input [7:0] s_axis_s2mm_cmd_tdest,

  output [31:0] m_axis_s2mm_sts_tdata,
  input m_axis_s2mm_sts_tready,
  output m_axis_s2mm_sts_tvalid,
  output [3:0] m_axis_s2mm_sts_tkeep,
  output m_axis_s2mm_sts_tlast
);

  external_dma_bd ext_dma_bd(

        .s_axi_control_araddr(s_axi_control_araddr),
        .s_axi_control_arready(s_axi_control_arready),
        .s_axi_control_arvalid(s_axi_control_arvalid),
        .s_axi_control_awaddr(s_axi_control_awaddr),
        .s_axi_control_awready(s_axi_control_awready),
        .s_axi_control_awvalid(s_axi_control_awvalid),
        .s_axi_control_bready(s_axi_control_bready),
        .s_axi_control_bresp(s_axi_control_bresp),
        .s_axi_control_bvalid(s_axi_control_bvalid),
        .s_axi_control_rdata(s_axi_control_rdata),
        .s_axi_control_rready(s_axi_control_rready),
        .s_axi_control_rresp(s_axi_control_rresp),
        .s_axi_control_rvalid(s_axi_control_rvalid),
        .s_axi_control_wdata(s_axi_control_wdata),
        .s_axi_control_wready(s_axi_control_wready),
        .s_axi_control_wstrb(s_axi_control_wstrb),
        .s_axi_control_wvalid(s_axi_control_wvalid),

        
        .m_axi_0_araddr(m_axi_0_araddr),
        .m_axi_0_arburst(m_axi_0_arburst),
        .m_axi_0_arcache(m_axi_0_arcache),
        .m_axi_0_arlen(m_axi_0_arlen),
        .m_axi_0_arprot(m_axi_0_arprot),
        .m_axi_0_arready(m_axi_0_arready),
        .m_axi_0_arsize(m_axi_0_arsize),
        .m_axi_0_aruser(m_axi_0_aruser),
        .m_axi_0_arvalid(m_axi_0_arvalid),
        .m_axi_0_awaddr(m_axi_0_awaddr),
        .m_axi_0_awburst(m_axi_0_awburst),
        .m_axi_0_awcache(m_axi_0_awcache),
        .m_axi_0_awlen(m_axi_0_awlen),
        .m_axi_0_awprot(m_axi_0_awprot),
        .m_axi_0_awready(m_axi_0_awready),
        .m_axi_0_awsize(m_axi_0_awsize),
        .m_axi_0_awuser(m_axi_0_awuser),
        .m_axi_0_awvalid(m_axi_0_awvalid),
        .m_axi_0_bready(m_axi_0_bready),
        .m_axi_0_bresp(m_axi_0_bresp),
        .m_axi_0_bvalid(m_axi_0_bvalid),
        .m_axi_0_rdata(m_axi_0_rdata),
        .m_axi_0_rlast(m_axi_0_rlast),
        .m_axi_0_rready(m_axi_0_rready),
        .m_axi_0_rresp(m_axi_0_rresp),
        .m_axi_0_rvalid(m_axi_0_rvalid),
        .m_axi_0_wdata(m_axi_0_wdata),
        .m_axi_0_wlast(m_axi_0_wlast),
        .m_axi_0_wready(m_axi_0_wready),
        .m_axi_0_wstrb(m_axi_0_wstrb),
        .m_axi_0_wvalid(m_axi_0_wvalid),



        .s_axis_s2mm_tdata(s_axis_s2mm_tdata),
        .s_axis_s2mm_tkeep(s_axis_s2mm_tkeep),
        

        .s_axis_s2mm_tlast(s_axis_s2mm_tlast),
        .s_axis_s2mm_tready(s_axis_s2mm_tready),
        .s_axis_s2mm_tvalid(s_axis_s2mm_tvalid),

        .m_axis_mm2s_tdata(m_axis_mm2s_tdata),
        .m_axis_mm2s_tkeep(m_axis_mm2s_tkeep),
        .m_axis_mm2s_tlast(m_axis_mm2s_tlast),
        .m_axis_mm2s_tready(m_axis_mm2s_tready),
        .m_axis_mm2s_tvalid(m_axis_mm2s_tvalid),

        .s_axis_mm2s_cmd_tdata(s_axis_mm2s_cmd_tdata),
        .s_axis_mm2s_cmd_tready(s_axis_mm2s_cmd_tready),
        .s_axis_mm2s_cmd_tvalid(s_axis_mm2s_cmd_tvalid),
        


        .m_axis_mm2s_sts_tdata(m_axis_mm2s_sts_tdata),
        .m_axis_mm2s_sts_tready(m_axis_mm2s_sts_tready),
        .m_axis_mm2s_sts_tvalid(m_axis_mm2s_sts_tvalid),
        .m_axis_mm2s_sts_tkeep(m_axis_mm2s_sts_tkeep),
        .m_axis_mm2s_sts_tlast(m_axis_mm2s_sts_tlast),

        .s_axis_s2mm_cmd_tdata(s_axis_s2mm_cmd_tdata),
        .s_axis_s2mm_cmd_tready(s_axis_s2mm_cmd_tready),
        .s_axis_s2mm_cmd_tvalid(s_axis_s2mm_cmd_tvalid),
        


        .m_axis_s2mm_sts_tdata(m_axis_s2mm_sts_tdata),
        .m_axis_s2mm_sts_tready(m_axis_s2mm_sts_tready),
        .m_axis_s2mm_sts_tvalid(m_axis_s2mm_sts_tvalid),
        .m_axis_s2mm_sts_tkeep(m_axis_s2mm_sts_tkeep),
        .m_axis_s2mm_sts_tlast(m_axis_s2mm_sts_tlast),

        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n)
        );

endmodule
