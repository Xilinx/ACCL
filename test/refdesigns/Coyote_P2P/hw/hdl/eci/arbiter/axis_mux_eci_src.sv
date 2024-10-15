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

module axis_mux_eci_src #(
    parameter integer MUX_DATA_BITS = ECI_DATA_BITS
) (
    input  logic                            aclk,
    input  logic                            aresetn,

    muxIntf.m                               mux_user,

    AXI4S.s                                 axis_in,
    AXI4S.m                                 axis_out [N_CHAN]
);

// -- Constants
localparam integer BEAT_LOG_BITS = $clog2(MUX_DATA_BITS/8);
localparam integer BLEN_BITS = LEN_BITS - BEAT_LOG_BITS;

// -- FSM
typedef enum logic[0:0]  {ST_IDLE, ST_MUX} state_t;
logic [0:0] state_C, state_N;

// -- Internal regs
logic [N_CHAN_BITS-1:0] id_C, id_N;
logic [BLEN_BITS-1:0] cnt_C, cnt_N;
logic ctl_C, ctl_N;

// -- Internal signals
logic tr_done;

// ----------------------------------------------------------------------------------------------------------------------- 
// -- Mux 
// ----------------------------------------------------------------------------------------------------------------------- 
// -- interface loop issues => temp signals
logic                                   axis_in_tvalid;
logic                                   axis_in_tready;
logic [MUX_DATA_BITS-1:0]               axis_in_tdata;
logic [MUX_DATA_BITS/8-1:0]             axis_in_tkeep;
logic                                   axis_in_tlast;

logic [N_CHAN-1:0]                        axis_out_tvalid;
logic [N_CHAN-1:0]                        axis_out_tready;
logic [N_CHAN-1:0][MUX_DATA_BITS-1:0]     axis_out_tdata;
logic [N_CHAN-1:0][MUX_DATA_BITS/8-1:0]   axis_out_tkeep;
logic [N_CHAN-1:0]                        axis_out_tlast;

assign axis_in_tvalid = axis_in.tvalid;
assign axis_in_tdata = axis_in.tdata;
assign axis_in_tkeep = axis_in.tkeep;
assign axis_in_tlast = axis_in.tlast;
assign axis_in.tready = axis_in_tready;

for(genvar i = 0; i < N_CHAN; i++) begin
    assign axis_out[i].tvalid = axis_out_tvalid[i];
    assign axis_out[i].tdata = axis_out_tdata[i];
    assign axis_out[i].tkeep = axis_out_tkeep[i];
    assign axis_out[i].tlast = axis_out_tlast[i];
    assign axis_out_tready[i] = axis_out[i].tready;
end

// -- Mux
always_comb begin
    for(int i = 0; i < N_CHAN; i++) begin
        axis_out_tdata[i] = axis_in_tdata;
        axis_out_tkeep[i] = axis_in_tkeep;
        axis_out_tlast[i] = axis_in_tlast & ctl_C;
        if(state_C == ST_MUX) begin
            axis_out_tvalid[i] = (id_C == i) ? axis_in_tvalid : 1'b0;
        end
        else begin
            axis_out_tvalid[i] = 1'b0;
        end
    end

    if(id_C < N_CHAN && state_C == ST_MUX) 
        axis_in_tready = axis_out_tready[id_C];
    else 
        axis_in_tready = 1'b0;
end

// ----------------------------------------------------------------------------------------------------------------------- 
// -- Memory subsystem 
// ----------------------------------------------------------------------------------------------------------------------- 
// -- REG
always_ff @(posedge aclk) begin: PROC_REG
if (aresetn == 1'b0) begin
	state_C <= ST_IDLE;
  cnt_C <= 'X;
  id_C <= 'X;
  ctl_C <= 'X;
end
else
  state_C <= state_N;
  cnt_C <= cnt_N;
  id_C <= id_N;
  ctl_C <= ctl_N;
end

// -- NSL
always_comb begin: NSL
	state_N = state_C;

	case(state_C)
		ST_IDLE: 
			state_N = mux_user.ready ? ST_MUX : ST_IDLE;

    ST_MUX:
      state_N = tr_done ? (mux_user.ready ? ST_MUX : ST_IDLE) : ST_MUX;

	endcase // state_C
end

// -- DP
always_comb begin : DP
  cnt_N = cnt_C;
  id_N = id_C;
  ctl_N = ctl_C;

  // Transfer done
  tr_done = (cnt_C == 0) && (axis_in_tvalid & axis_in_tready);

  // Memory subsystem
  mux_user.valid = 1'b0;

  case(state_C)
    ST_IDLE: begin
      if(mux_user.ready) begin
        mux_user.valid = 1'b1;
        id_N = mux_user.vfid;
        cnt_N = mux_user.len;
        ctl_N = mux_user.ctl;
      end   
    end

    ST_MUX: begin
      if(tr_done) begin
        if(mux_user.ready) begin
          mux_user.valid = 1'b1;
          id_N = mux_user.vfid;
          cnt_N = mux_user.len;   
          ctl_N = mux_user.ctl;       
        end
      end 
      else begin
        cnt_N = (axis_in_tvalid & axis_in_tready) ? cnt_C - 1 : cnt_C;
      end
    end

  endcase
end

endmodule