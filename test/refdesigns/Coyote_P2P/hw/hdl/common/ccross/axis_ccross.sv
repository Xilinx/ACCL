`timescale 1ns / 1ps

import lynxTypes::*;

module axis_ccross (
	input logic 			s_aclk,
	input logic 			s_aresetn,
	input logic 			m_aclk,
	input logic 			m_aresetn,
	
	AXI4S.s 				s_axis,
	AXI4S.m 				m_axis
);


	axis_clock_converter_512 inst_reg_slice (
		.s_axis_aclk(s_aclk),
		.s_axis_aresetn(s_aresetn),
		.m_axis_aclk(m_aclk),
		.m_axis_aresetn(m_aresetn),
		.s_axis_tvalid(s_axis.tvalid),
		.s_axis_tready(s_axis.tready),
		.s_axis_tdata(s_axis.tdata),
		.s_axis_tkeep(s_axis.tkeep),
		.s_axis_tlast(s_axis.tlast),
		.m_axis_tvalid(m_axis.tvalid),
		.m_axis_tready(m_axis.tready),
		.m_axis_tdata(m_axis.tdata),
		.m_axis_tkeep(m_axis.tkeep),
		.m_axis_tlast(m_axis.tlast)
	);

endmodule