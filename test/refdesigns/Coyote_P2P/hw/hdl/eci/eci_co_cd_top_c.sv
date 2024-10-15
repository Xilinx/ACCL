`ifndef ECI_CO_CD_TOP_C_SV
`define ECI_CO_CD_TOP_C_SV

import block_types::*;
import eci_cmd_defs::*;

import lynxTypes::*;

/* Request response pattern
 * CPU Initiated
 * * VC6,7 
 *   * GSYNC  - GSDN VC10/11 (if req is in VC6 resp is in VC10 and so on) 
 *   * GINV   - Ignored no response 
 *   * Others - Ignored no response  
 * 
 */
module eci_co_cd_top_c
  (
   input logic 					    clk,
   input logic 					    reset,

    // Coyote
    output logic [ECI_CL_WIDTH-1:0]      axis_dyn_out_tdata,
    output logic [ECI_CL_WIDTH/8-1:0]    axis_dyn_out_tkeep,
    output logic                         axis_dyn_out_tlast,  
    output logic                         axis_dyn_out_tvalid,
    input  logic                         axis_dyn_out_tready,

    input  logic [ECI_CL_WIDTH-1:0]      axis_dyn_in_tdata,
    input  logic [ECI_CL_WIDTH/8-1:0]    axis_dyn_in_tkeep,
    input  logic                         axis_dyn_in_tlast,
    input  logic                         axis_dyn_in_tvalid,
    output logic                         axis_dyn_in_tready,

    input  logic [39:0]                  rd_desc_addr,
    input  logic [19:0]                  rd_desc_len,
    input  logic                         rd_desc_valid,
    output logic                         rd_desc_ready,
    output logic                         rd_desc_done,

    input  logic [39:0]                  wr_desc_addr,
    input  logic [19:0]                  wr_desc_len,
    input  logic                         wr_desc_valid,
    output logic                         wr_desc_ready,
    output logic                         wr_desc_done,

   //------FPGA to CPU (MOB) VCs that are input to eci_module------//

   //------Request without data CO VCs------//
   output logic [ECI_WORD_WIDTH-1:0] f_vc7_co_o,
   output logic 				    f_vc7_co_valid_o,
   output logic [4:0] 	    f_vc7_co_size_o,
   input logic 					    f_vc7_co_ready_i,

   output logic [ECI_WORD_WIDTH-1:0] f_vc6_co_o,
   output logic 				    f_vc6_co_valid_o,
   output logic [4:0] 	    f_vc6_co_size_o,
   input logic 					    f_vc6_co_ready_i,
   //------end Request without data CO VCs------//

   //------Reqeust with data CD VCs 3,2------//
   output logic [17*ECI_WORD_WIDTH-1:0] f_vc3_cd_o,
   output logic 				    f_vc3_cd_valid_o,
   output logic [4:0] 	    f_vc3_cd_size_o,
   input logic 					    f_vc3_cd_ready_i,

   output logic [17*ECI_WORD_WIDTH-1:0] f_vc2_cd_o,
   output logic 				    f_vc2_cd_valid_o,
   output logic [4:0] 	    f_vc2_cd_size_o,
   input logic 					    f_vc2_cd_ready_i,
   //------end Reqeust with data CD VCs 3,2------//

   //------end FPGA to CPU (MOB) VCs that are input to eci_module------//


   //------CPU to FPGA (MIB) VCs that are output from eci module------//
   // c_ -> CPU initiated

   //------Response without data CO VCs------//
   input logic [ECI_WORD_WIDTH-1:0]  c_vc11_co_i,
   input logic 					    c_vc11_co_valid_i,
   input logic [4:0] 	    c_vc11_co_size_i,
   output logic 				    c_vc11_co_ready_o,

   input logic [ECI_WORD_WIDTH-1:0]  c_vc10_co_i,
   input logic 					    c_vc10_co_valid_i,
   input logic [4:0] 	    c_vc10_co_size_i,
   output logic 				    c_vc10_co_ready_o,
   //------end Response without data CO VCs------//

   //------Response with data CD VCs------//
   input logic [17*ECI_WORD_WIDTH-1:0]  c_vc5_cd_i,
   input logic 					    c_vc5_cd_valid_i,
   input logic [4:0] 	    c_vc5_cd_size_i,
   output logic 				    c_vc5_cd_ready_o,

   input logic [17*ECI_WORD_WIDTH-1:0]  c_vc4_cd_i,
   input logic 					    c_vc4_cd_valid_i,
   input logic [4:0] 	    c_vc4_cd_size_i,
   output logic 				    c_vc4_cd_ready_o
   //------end Response with data CD VCs------//
   //------end CPU to FPGA (MIB) VCs that are output from eci module------//
   );

   // ECI DMA
   // Write requests  to f_vc2, f_vc3    connected directly 
   // Read  requests  to f_vc6, f_vc7    connected directly 
   // Read  responses from c_vc4, c_vc5  connected directly 
   // Write responses from c_vc10.c_vc11 connected directly 
   eci_dma_2vc inst_eci_dma (
   	.aclk(clk),
   	.aresetn(~reset),

   	// Read descriptor Input
   	// Currently not connected, getting from VIO 
   	.s_axis_read_desc_addr  (rd_desc_addr),
   	.s_axis_read_desc_len   (rd_desc_len),
   	.s_axis_read_desc_valid (rd_desc_valid),
   	.s_axis_read_desc_ready (rd_desc_ready),

   	// Read Data output 
   	// Currently not connected sending to ILA   
   	.m_axis_read_data_tdata  (axis_dyn_out_tdata),
   	.m_axis_read_data_tkeep  (axis_dyn_out_tkeep),  	 
   	.m_axis_read_data_tlast  (axis_dyn_out_tlast),  	 
   	.m_axis_read_data_tvalid (axis_dyn_out_tvalid), 	 
   	.m_axis_read_data_tready (axis_dyn_out_tready),

   	// Read Descriptor Status Output
   	// Currently not connected
   	.m_axis_read_desc_status_valid (rd_desc_done),  

   	// Write Descriptor input
   	// Currently not connected, getting from VIO 
   	.s_axis_write_desc_addr  (wr_desc_addr), 
   	.s_axis_write_desc_len   (wr_desc_len),  
   	.s_axis_write_desc_valid (wr_desc_valid),
   	.s_axis_write_desc_ready (wr_desc_ready),

   	// Write Data Input
   	// Currently not connected, getting from VIO
   	.s_axis_write_data_tdata(axis_dyn_in_tdata),  
   	.s_axis_write_data_tkeep(axis_dyn_in_tkeep),  
   	.s_axis_write_data_tlast(axis_dyn_in_tlast),   
   	.s_axis_write_data_tvalid(axis_dyn_in_tvalid), 
   	.s_axis_write_data_tready(axis_dyn_in_tready), 
	
   	// Write status output
   	// Currently not connected,
   	.m_axis_write_desc_status_valid(wr_desc_done),

   	// Read request output 
   	// From FPGA to CPU through VC6
   	.rdreq0_vc_data_o  (f_vc6_co_o),
   	.rdreq0_vc_size_o  (f_vc6_co_size_o),
   	.rdreq0_vc_valid_o (f_vc6_co_valid_o),
   	.rdreq0_vc_ready_i (f_vc6_co_ready_i),

   	// Read request output
   	// From FPGA to CPU through VC7
   	.rdreq1_vc_data_o  (f_vc7_co_o),
   	.rdreq1_vc_size_o  (f_vc7_co_size_o),
   	.rdreq1_vc_valid_o (f_vc7_co_valid_o),
   	.rdreq1_vc_ready_i (f_vc7_co_ready_i),

   	// Read response input
   	// From CPU to FPGA through VC4
   	.rdresp0_vc_data_i  (c_vc4_cd_i),
   	.rdresp0_vc_size_i  (c_vc4_cd_size_i),
   	.rdresp0_vc_valid_i (c_vc4_cd_valid_i),
   	.rdresp0_vc_ready_o (c_vc4_cd_ready_o),

   	// Read response input
   	// From CPU to FPGA through VC5
   	.rdresp1_vc_data_i  (c_vc5_cd_i),
   	.rdresp1_vc_size_i  (c_vc5_cd_size_i),
   	.rdresp1_vc_valid_i (c_vc5_cd_valid_i),
   	.rdresp1_vc_ready_o (c_vc5_cd_ready_o),

   	// Write request output
   	// From FPGA to CPU through VC2
   	.wrreq0_vc_data_o  (f_vc2_cd_o),
   	.wrreq0_vc_size_o  (f_vc2_cd_size_o),
   	.wrreq0_vc_valid_o (f_vc2_cd_valid_o),
   	.wrreq0_vc_ready_i (f_vc2_cd_ready_i),

   	// Write request output
   	// From FPGA to CPU through VC3
   	.wrreq1_vc_data_o  (f_vc3_cd_o),
   	.wrreq1_vc_size_o  (f_vc3_cd_size_o),
   	.wrreq1_vc_valid_o (f_vc3_cd_valid_o),
   	.wrreq1_vc_ready_i (f_vc3_cd_ready_i),

   	// Write response input
   	// From CPU to FPGA through VC10
   	.wrresp0_vc_data_i  (c_vc10_co_i),
   	.wrresp0_vc_size_i  (c_vc10_co_size_i),
   	.wrresp0_vc_valid_i (c_vc10_co_valid_i),
   	.wrresp0_vc_ready_o (c_vc10_co_ready_o),

   	// Write response input
   	// From CPU to FPGA through VC11
   	.wrresp1_vc_data_i  (c_vc11_co_i),
   	.wrresp1_vc_size_i  (c_vc11_co_size_i),
   	.wrresp1_vc_valid_i (c_vc11_co_valid_i),
   	.wrresp1_vc_ready_o (c_vc11_co_ready_o)	
  );

endmodule // eci_co_cd_top
`endif
