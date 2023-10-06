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

#define UB_SQ_WORD 7

void rdma_sq_handler(
	STREAM<rdma_req_t> & rdma_sq,
	STREAM<ap_axiu<32,0,0,0>> & ub_sq,
	STREAM<eth_header> & cmd_in,
	STREAM<eth_header> & cmd_out
)
{
#pragma HLS INTERFACE axis register both port=rdma_sq
#pragma HLS INTERFACE axis register both port=ub_sq
#pragma HLS INTERFACE axis register both port=cmd_in
#pragma HLS INTERFACE axis register both port=cmd_out
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1

#pragma HLS aggregate variable=rdma_sq compact=bit

	unsigned const bytes_per_word = DATA_WIDTH/8;

	static rdma_req_t rdma_req;
	static eth_header cmd_in_word;
	
	enum fsmStateType {WAIT_CMD, RNDZVS_INIT_CMD, RNDZVS_MSG_CMD, RNDZVS_DONE_CMD, EGR_MSG_CMD};
    static fsmStateType  fsmState = WAIT_CMD;

	static ap_uint<32> ub_sq_cnt = 0;

	static ap_uint<32> ub_sq_vec[UB_SQ_WORD];
	#pragma HLS ARRAY_PARTITION variable=ub_sq_vec complete 



	switch (fsmState)
    {
		case WAIT_CMD:
			if (!STREAM_IS_EMPTY(cmd_in)){
				//read commands from the command stream
				cmd_in_word = STREAM_READ(cmd_in);
				if (cmd_in_word.msg_type == EGR_MSG){
					fsmState = EGR_MSG_CMD;
				} else if (cmd_in_word.msg_type == RNDZVS_MSG) {
					fsmState = RNDZVS_MSG_CMD;
				}
			} else if (!STREAM_IS_EMPTY(ub_sq)){
				fsmState = RNDZVS_INIT_CMD;
			}
			break;
		case RNDZVS_INIT_CMD:
			if (!STREAM_IS_EMPTY(ub_sq)){
				ub_sq_vec[ub_sq_cnt] = STREAM_READ(ub_sq).data;
				ub_sq_cnt++;
				if (ub_sq_cnt == UB_SQ_WORD){
					//issue sq command
					rdma_req.opcode = RDMA_SEND;
					rdma_req.qpn = ub_sq_vec[0];
					rdma_req.host = 0;
					rdma_req.len = bytes_per_word;//will send just a header
					rdma_req.vaddr = 0;
					STREAM_WRITE(rdma_sq, rdma_req);
					//issue packetizer command
					cmd_in_word.msg_type = RNDZVS_INIT;
					cmd_in_word.src = ub_sq_vec[6];
					cmd_in_word.seqn = 0;
					cmd_in_word.strm = 0;
					cmd_in_word.dst = rdma_req.qpn;
					cmd_in_word.vaddr(31,0) = ub_sq_vec[1];
					cmd_in_word.vaddr(RDMA_VADDR_BITS-1,32) = ub_sq_vec[2];
					cmd_in_word.host = ub_sq_vec[3];
					cmd_in_word.count = ub_sq_vec[4];
					cmd_in_word.tag = ub_sq_vec[5];
					STREAM_WRITE(cmd_out, cmd_in_word);
					// clear cnt and move back to WAIT_CMD
					ub_sq_cnt = 0;
					fsmState = WAIT_CMD;
				}
			}
			break;
		case RNDZVS_MSG_CMD:
			// issue an RDMA WRITE to target remote address
			// and issue the eth cmd to the rdma packetizer
			rdma_req.opcode = RDMA_WRITE;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = cmd_in_word.host;
			rdma_req.len = cmd_in_word.count; // msg size as no header will be packed into WRITE Verb
			rdma_req.vaddr = cmd_in_word.vaddr;
			STREAM_WRITE(rdma_sq, rdma_req);
			STREAM_WRITE(cmd_out, cmd_in_word);

			fsmState = RNDZVS_DONE_CMD;
			break;
		case RNDZVS_DONE_CMD:
			// issue an RDMA SEND to tell the remote end that the WRITE is done
			rdma_req.opcode = RDMA_SEND;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = 0;
			rdma_req.len = bytes_per_word; // only the header len
			rdma_req.vaddr = 0;
			STREAM_WRITE(rdma_sq, rdma_req);

			// and issue the eth cmd to the rdma packetizer
			cmd_in_word.msg_type = RNDZVS_WR_DONE;
			STREAM_WRITE(cmd_out, cmd_in_word);

			fsmState = WAIT_CMD;	
			break;
		case EGR_MSG_CMD:
			// issue an RDMA SEND
			// and issue the eth cmd to the rdma packetizer
			rdma_req.opcode = RDMA_SEND;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = cmd_in_word.host;
			rdma_req.len = cmd_in_word.count + bytes_per_word; // msg size plus the header size
			rdma_req.vaddr = 0; // not used in SEND Verb
			STREAM_WRITE(rdma_sq, rdma_req);
			STREAM_WRITE(cmd_out, cmd_in_word);

			fsmState = WAIT_CMD;
			break;  
    }

}
