/*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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

#include "client_arbiter.h"


void client_arbiter(
	STREAM<command_word> cmd_clients[NUM_CTRL_STREAMS],
	STREAM<command_word> ack_clients[NUM_CTRL_STREAMS],
	STREAM<command_word>  &cmd_cclo,
	STREAM<command_word>  &ack_cclo
){
#pragma HLS INTERFACE axis register both port=cmd_clients
#pragma HLS INTERFACE axis register both port=ack_clients
#pragma HLS INTERFACE axis register both port=cmd_cclo
#pragma HLS INTERFACE axis register both port=ack_cclo
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS pipeline II=1 style=frp

	// 1-Hot indication of current token position of highest priority
	static ap_uint<NUM_CTRL_STREAMS>  pos = 1;
#pragma HLS reset variable=pos
	static ap_uint<4>  cnt = 0;
#pragma HLS reset variable=cnt

	if(cnt != 0) {
		command_word  cmd;
		for(int  i = 0; i < NUM_CTRL_STREAMS; i++) {
#pragma HLS unroll
			if(pos[i])  cmd = STREAM_READ(cmd_clients[i]);
		}
		STREAM_WRITE(cmd_cclo, cmd);

		if(--cnt == 0) {
			// Return ACK
			command_word  const ack = STREAM_READ(ack_cclo);
			for(int  i = 0; i < NUM_CTRL_STREAMS; i++) {
#pragma HLS unroll
				if(pos[i])  STREAM_WRITE(ack_clients[i], ack);
			}

			// Rotate priority position to left neighbor
			pos = (pos(NUM_CTRL_STREAMS-2, 0), pos[NUM_CTRL_STREAMS-1]);
		}
	}
	else {
		// Request vector with asserted added MSB
		ap_uint<1+NUM_CTRL_STREAMS>  rqst;
		for(int  i = 0; i < NUM_CTRL_STREAMS; i++) {
#pragma HLS unroll
			rqst[i] = !STREAM_IS_EMPTY(cmd_clients[i]);
		}
		rqst[NUM_CTRL_STREAMS] = 1;

		// 1-Hot grant vectors starting at token position and at LSB, respectively
		ap_uint<1+NUM_CTRL_STREAMS> const  grnt1 = (~rqst + pos) & rqst;
		ap_uint<1+NUM_CTRL_STREAMS> const  grnt0 = (~rqst +   1) & rqst;

		// Commit to highest-prio request if any
		if(!grnt0[NUM_CTRL_STREAMS]) {
			pos = grnt1[NUM_CTRL_STREAMS]? grnt0 : grnt1;
			cnt = 15;
		}
	}
}
