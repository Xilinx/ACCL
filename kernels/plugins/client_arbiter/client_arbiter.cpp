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
	hls::stream<ap_uint<32>> (&cmd_clients)[NCLIENTS],
	hls::stream<ap_uint<32>> (&ack_clients)[NCLIENTS],
	hls::stream<ap_uint<32>>  &cmd_cclo,
	hls::stream<ap_uint<32>>  &ack_cclo
){
#pragma HLS INTERFACE axis register both port=cmd_clients
#pragma HLS INTERFACE axis register both port=ack_clients
#pragma HLS INTERFACE axis register both port=cmd_cclo
#pragma HLS INTERFACE axis register both port=ack_cclo
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS pipeline II=1 style=frp

	// 1-Hot indication of current token position of highest priority
	static ap_uint<NCLIENTS>  pos = 1;
#pragma HLS reset variable=pos

	// Request vector with asserted added MSB
	ap_uint<1+NCLIENTS>  rqst;
	for(int  i = 0; i < NCLIENTS; i++) {
#pragma HLS unroll
		rqst[i] = !cmd_clients[i].empty();
	}
	rqst[NCLIENTS] = 1;

	// 1-Hot grant vectors starting at token position and at LSB, respectively
	ap_uint<1+NCLIENTS> const  grnt1 = (~rqst + pos) & rqst;
	ap_uint<1+NCLIENTS> const  grnt0 = (~rqst +   1) & rqst;

	// Take this round only if there is at least one request
	if(!grnt0[NCLIENTS]) {
		ap_uint<NCLIENTS> const  grant = grnt1[NCLIENTS]? grnt0 : grnt1;

		// Collect the granted command
		ap_uint<32>  cmd = 0;
		for(int  i = 0; i < NCLIENTS; i++) {
#pragma HLS unroll
			if(grant[i]) {
				ap_uint<32>  ccmd;
				cmd_clients[i].read_nb(ccmd);
				cmd |= ccmd;
			}
		}

		// Dispatch to CCLO
		cmd_cclo.write(cmd);
		ap_uint<32> const  ack = ack_cclo.read();

		// Return ACK
		for(int  i = 0; i < NCLIENTS; i++) {
#pragma HLS unroll
			if(grant[i])  ack_clients[i].write(ack);
		}

		// Rotate priority position to left neighbor
		pos = (grant(NCLIENTS-2, 0), grant[NCLIENTS-1]);
	}
}