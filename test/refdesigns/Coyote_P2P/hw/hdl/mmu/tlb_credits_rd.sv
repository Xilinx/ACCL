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

/**
 * @brief   TLB credit based system for the read requests.
 *
 * Prevents region stalls from propagating to the whole system.
 *
 *  @param ID_REG           Number of associated vFPGA
 *  @param DATA_BITS        Size of the data bus
 */
module tlb_credits_rd #(
    parameter integer ID_REG = 0,
    parameter integer DATA_BITS = AXI_DATA_BITS
) (
    input  logic            aclk,
    input  logic            aresetn,
    
    // Requests
    dmaIntf.s               s_req,
    dmaIntf.m               m_req,

    // Data read
    input  logic            rxfer,
    output cred_t           rd_dest
);

// -- Constants
localparam integer BEAT_LOG_BITS = $clog2(DATA_BITS/8);
localparam integer BLEN_BITS = LEN_BITS - BEAT_LOG_BITS;

// -- FSM
typedef enum logic[0:0]  {ST_IDLE, ST_READ} state_t;
logic [0:0] state_C, state_N;

// -- Internal regs
logic [7:0] cred_reg_C, cred_reg_N;
logic [BLEN_BITS-1:0] cnt_C, cnt_N;
logic [DEST_BITS-1:0] dest_C, dest_N;
logic [PID_BITS-1:0] pid_C, pid_N;

// -- Internal signals
logic req_sent;
logic req_done;

logic [BLEN_BITS-1:0] rd_len;

metaIntf #(.STYPE(logic[PID_BITS+DEST_BITS+BLEN_BITS-1:0])) req_que_in ();
metaIntf #(.STYPE(logic[PID_BITS+DEST_BITS+BLEN_BITS-1:0])) req_que_out ();

// -- REG
always_ff @(posedge aclk) begin: PROC_REG
if (aresetn == 1'b0) begin
	  state_C <= ST_IDLE;
    cred_reg_C <= 0;
    cnt_C <= 'X;
    dest_C <= 'X;
    pid_C <= 'X;

    s_req.rsp <= 0;
end
else
    state_C <= state_N;
    cred_reg_C <= cred_reg_N;
    cnt_C <= cnt_N;
    dest_C <= dest_N;
    pid_C <= pid_N;

    s_req.rsp <= m_req.rsp;
end

// -- NSL
always_comb begin: NSL
	state_N = state_C;

	case(state_C)
		ST_IDLE: 
			state_N = req_que_out.valid ? ST_READ : ST_IDLE;

    ST_READ:
        state_N = req_done ? (req_que_out.valid ? ST_READ : ST_IDLE) : ST_READ;

	endcase // state_C
end

// -- DP
always_comb begin
  cred_reg_N = cred_reg_C;
  cnt_N =  cnt_C;
  dest_N = dest_C;
  pid_N = pid_C;
  // IO
  s_req.ready = 1'b0;
  
  m_req.valid = 1'b0;
  m_req.req = s_req.req;

  // Status
  req_sent = s_req.valid && m_req.ready && req_que_in.ready && ((cred_reg_C < N_OUTSTANDING) || req_done);
  req_done = (cnt_C == 0) && rxfer;

  // Outstanding queue
  req_que_in.valid = 1'b0;
  rd_len = (s_req.req.len - 1) >> BEAT_LOG_BITS;
  req_que_in.data = {s_req.req.pid, s_req.req.dest, rd_len};
  req_que_out.ready = 1'b0;

  if(req_sent && !req_done)
      cred_reg_N = cred_reg_C + 1;
  else if(req_done && !req_sent)
      cred_reg_N = cred_reg_C - 1;

  if(req_sent) begin
      s_req.ready = 1'b1;
      m_req.valid = 1'b1;
      req_que_in.valid = 1'b1;
  end

  case(state_C)
    ST_IDLE: begin
      if(req_que_out.valid) begin
        req_que_out.ready = 1'b1;
        cnt_N = req_que_out.data[BLEN_BITS-1:0];
        dest_N = req_que_out.data[BLEN_BITS+:DEST_BITS];  
        pid_N = req_que_out.data[BLEN_BITS+DEST_BITS+:PID_BITS]; 
      end   
    end

    ST_READ: begin
      if(req_done) begin
        if(req_que_out.valid) begin
            req_que_out.ready = 1'b1;
            cnt_N = req_que_out.data[BLEN_BITS-1:0];
            dest_N = req_que_out.data[BLEN_BITS+:DEST_BITS];
            pid_N = req_que_out.data[BLEN_BITS+DEST_BITS+:PID_BITS];
        end
      end 
      else begin
        cnt_N = rxfer ? cnt_C - 1 : cnt_C;
      end
    end

  endcase
end

// Output dest
assign rd_dest.dest = dest_C;
assign rd_dest.pid = pid_C;
assign rd_dest.user = 0;

// Outstanding
queue_stream #(
  .QTYPE(logic [PID_BITS+DEST_BITS+BLEN_BITS-1:0]),
  .QDEPTH(N_OUTSTANDING)
) inst_dque (
  .aclk(aclk),
  .aresetn(aresetn),
  .val_snk(req_que_in.valid),
  .rdy_snk(req_que_in.ready),
  .data_snk(req_que_in.data),
  .val_src(req_que_out.valid),
  .rdy_src(req_que_out.ready),
  .data_src(req_que_out.data)
);


/////////////////////////////////////////////////////////////////////////////
// DEBUG
/////////////////////////////////////////////////////////////////////////////
`ifdef DBG_TLB_CREDITS_RD

`endif

endmodule