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
#include "eth_intf.h"

using namespace std;



void rdma_sq_handler(
	STREAM<rdma_req_t> & rdma_sq,
	STREAM<eth_header> & cmd_in,
	STREAM<eth_header> & cmd_out
)
{
#pragma HLS INTERFACE axis register both port=rdma_sq
#pragma HLS INTERFACE axis register both port=cmd_in
#pragma HLS INTERFACE axis register both port=cmd_out
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1

// aggregate compact bit required as goes external to Coyote
#pragma HLS aggregate variable=rdma_sq compact=bit

	unsigned const bytes_per_word = DATA_WIDTH/8;

	static rdma_req_t rdma_req;
	static rdma_req_msg_t rdma_req_msg;
	static eth_header cmd_in_word;
	
	enum fsmStateType {WAIT_CMD, RNDZVS_VERB, RNDZVS_DONE, EGR_VERB};
    static fsmStateType  fsmState = WAIT_CMD;


	switch (fsmState)
    {
		case WAIT_CMD:
			if (!STREAM_IS_EMPTY(cmd_in)){
				//read commands from the command stream
				cmd_in_word = STREAM_READ(cmd_in);
				if (cmd_in_word.protoc == EGR_PROTOC){
					fsmState = EGR_VERB;
				}
			}
			break;
		case RNDZVS_VERB:
			// issue an RDMA WRITE to target remote address
			// and issue the eth cmd to the rdma packetizer
			// TODO: double check semantic of each field
			rdma_req.opcode = RDMA_WRITE;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = 0;
			rdma_req.mode = 0;
			rdma_req.last = 1;
			rdma_req.cmplt = 0;
			rdma_req.ssn = 0;
			rdma_req.offs = 0;
			rdma_req.rsrvd = 0;

			rdma_req_msg.lvaddr = 0;
			rdma_req_msg.rvaddr = cmd_in_word.rvaddr;
			rdma_req_msg.len = cmd_in_word.count;
			rdma_req_msg.params = 0;

			rdma_req.msg = (ap_uint<RDMA_MSG_BITS>)rdma_req_msg;
			STREAM_WRITE(rdma_sq, rdma_req);
			STREAM_WRITE(cmd_out, cmd_in_word);
			fsmState = RNDZVS_DONE;
			break;
		case RNDZVS_DONE:
			// issue an RDMA SEND with immediate data to tell the remote end that the WRITE is done
			rdma_req.opcode = RDMA_IMMED;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = 0; 
			rdma_req.mode = 0;
			rdma_req.last = 1;
			rdma_req.cmplt = 0;
			rdma_req.ssn = 0;
			rdma_req.offs = 0;
			rdma_req.rsrvd = 0;

			rdma_req_msg.lvaddr = 0;
			rdma_req_msg.rvaddr = 0;
			rdma_req_msg.len = 0;
			rdma_req_msg.params(7,0) = PARAMS_RNDZVS_WR_DONE;
			rdma_req_msg.params(HEADER_LENGTH+8-1,8) = (ap_uint<HEADER_LENGTH>)cmd_in_word;

			rdma_req.msg = (ap_uint<RDMA_MSG_BITS>)rdma_req_msg;
			STREAM_WRITE(rdma_sq, rdma_req);
			fsmState = WAIT_CMD;	
			break;
		case EGR_VERB:
			// issue an RDMA SEND
			// and issue the eth cmd to the rdma packetizer
			rdma_req.opcode = RDMA_SEND;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = 0; 
			rdma_req.mode = 0;
			rdma_req.last = 1;
			rdma_req.cmplt = 0;
			rdma_req.ssn = 0;
			rdma_req.offs = 0;
			rdma_req.rsrvd = 0;

			rdma_req_msg.lvaddr = 0;
			rdma_req_msg.rvaddr = 0;
			rdma_req_msg.len = cmd_in_word.count + bytes_per_word;
			rdma_req_msg.params = 0;

			rdma_req.msg = (ap_uint<RDMA_MSG_BITS>)rdma_req_msg;
			STREAM_WRITE(rdma_sq, rdma_req);
			STREAM_WRITE(cmd_out, cmd_in_word);
			fsmState = WAIT_CMD;
			break;  
    }

}
