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
	STREAM<rdma_meta_req_t> & rdma_meta_req,
	STREAM<rdma_meta_rsp_t> & rdma_meta_rsp,
	STREAM<eth_header> & cmd_in,
	STREAM<eth_header> & cmd_out
)
{
#pragma HLS INTERFACE axis register both port=rdma_sq
#pragma HLS INTERFACE axis register both port=rdma_meta_req
#pragma HLS INTERFACE axis register both port=rdma_meta_rsp
#pragma HLS INTERFACE axis register both port=cmd_in
#pragma HLS INTERFACE axis register both port=cmd_out
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1

// aggregate compact bit required as goes external to Coyote
#pragma HLS aggregate variable=rdma_sq compact=bit

	unsigned const bytes_per_word = DATA_WIDTH/8;

	static rdma_meta_req_t req;
	static rdma_meta_rsp_t rsp;
	static rdma_req_t rdma_req;
	static rdma_req_msg_t rdma_req_msg;
	static eth_header cmd_in_word;
	
	enum fsmStateType {WAIT_CMD, RNDZVS_WAIT_RADDR, RNDZVS_VERB, RNDZVS_DONE, EGR_VERB};
    static fsmStateType  fsmState = WAIT_CMD;


	switch (fsmState)
    {
		case WAIT_CMD:
			if (!STREAM_IS_EMPTY(cmd_in)){
				//read commands from the command stream
				cmd_in_word = STREAM_READ(cmd_in);
				if (cmd_in_word.protoc == RNDZVS_PROTOC) {
					// issue lookup request
					req.dst = cmd_in_word.dst;
					req.seqn = cmd_in_word.seqn;
					STREAM_WRITE(rdma_meta_req, req);
					fsmState = RNDZVS_WAIT_RADDR;
				} else if (cmd_in_word.protoc == EGR_PROTOC){
					fsmState = EGR_VERB;
				}
			}
			break;
		case RNDZVS_WAIT_RADDR:
			// re-issue the request until remote address is ready
			if (!STREAM_IS_EMPTY(rdma_meta_rsp)){
				rsp = STREAM_READ(rdma_meta_rsp);
				if (rsp.hit == 0) {
					STREAM_WRITE(rdma_meta_req, req);
				} else {
					fsmState = RNDZVS_VERB;
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
			rdma_req_msg.rvaddr = rsp.rvaddr;
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
			fsmState = WAIT_CMD;
			break;  
    }

}

// void rdma_sq_handler(
// 	STREAM<rdma_req_t> & rdma_sq,
// 	STREAM<rdma_meta_req_t> & rdma_meta_req,
// 	STREAM<rdma_meta_rsp_t> & rdma_meta_rsp,
// 	STREAM<eth_header> & cmd_in,
// 	STREAM<stream_word > & in,
// 	STREAM<stream_word > & out,
// 	STREAM<ap_uint<32> > & sts,
// 	unsigned int max_pktsize
// )
// {
// #pragma HLS INTERFACE axis register both port=rdma_sq
// #pragma HLS INTERFACE axis register both port=rdma_meta_req
// #pragma HLS INTERFACE axis register both port=rdma_meta_rsp
// #pragma HLS INTERFACE axis register both port=cmd_in
// #pragma HLS INTERFACE axis register both port=in
// #pragma HLS INTERFACE axis register both port=out
// #pragma HLS INTERFACE axis register both port=sts
// #pragma HLS INTERFACE s_axilite port=max_pktsize
// #pragma HLS INTERFACE s_axilite port=return

// // aggregate compact bit required as goes external to Coyote
// #pragma HLS aggregate variable=rdma_sq compact=bit

// 	unsigned const bytes_per_word = DATA_WIDTH/8;

// 	rdma_meta_req_t req;
// 	rdma_meta_rsp_t rsp;
// 	rdma_req_t rdma_req;
// 	rdma_req_msg_t rdma_req_msg;

// 	ap_uint<8> protocol = EGR_PROTOC;

// 	//read commands from the command stream
// 	eth_header cmd_in_word = STREAM_READ(cmd_in);

// 	// adopt Rendezevous 
// 	if (cmd_in_word.count > max_pktsize) {
// 		protocol = RNDZVS_PROTOC;
// 	} 

// 	int bytes_to_process = cmd_in_word.count;

// 	// RDMA request flow before the data transfer
// 	if (protocol == RNDZVS_PROTOC){

// 		bool rvaddr_valid = false;
// 		// issue lookup request
// 		req.dst = cmd_in_word.dst;
// 		req.seqn = cmd_in_word.seqn;
// 		STREAM_WRITE(rdma_meta_req, req);

// 		// loop until remote address is ready
// 		while (rvaddr_valid == false)
// 		{
// 		#pragma HLS PIPELINE II=1
// 			rsp = STREAM_READ(rdma_meta_rsp);
// 			rvaddr_valid = rsp.hit;
// 			if (rsp.hit == 0)
// 			{
// 				STREAM_WRITE(rdma_meta_req, req);
// 			}
// 		}

// 		// TODO: double check semantic of each field
// 		rdma_req.opcode = RDMA_WRITE;
// 		rdma_req.qpn = cmd_in_word.dst;
// 		rdma_req.host = 0;
// 		rdma_req.mode = 0;
// 		rdma_req.last = 1;
// 		rdma_req.cmplt = 0;
// 		rdma_req.ssn = 0;
// 		rdma_req.offs = 0;
// 		rdma_req.rsrvd = 0;

// 		rdma_req_msg.lvaddr = 0;
// 		rdma_req_msg.rvaddr = rsp.rvaddr;
// 		rdma_req_msg.len = cmd_in_word.count;
// 		rdma_req_msg.params = 0;

// 		rdma_req.msg = (ap_uint<RDMA_MSG_BITS>)rdma_req_msg;
// 		STREAM_WRITE(rdma_sq, rdma_req);
// 	} else {
// 		rdma_req.opcode = RDMA_IMMED;
// 		rdma_req.qpn = cmd_in_word.dst;
// 		rdma_req.host = 0;
// 		rdma_req.mode = 0;
// 		rdma_req.last = 1;
// 		rdma_req.cmplt = 0;
// 		rdma_req.ssn = 0;
// 		rdma_req.offs = 0;
// 		rdma_req.rsrvd = 0;

// 		// todo: check the lvaddr, rvaddr, len semantic for RDMA_IMMED
// 		rdma_req_msg.lvaddr = 0;
// 		rdma_req_msg.rvaddr = 0;
// 		rdma_req_msg.len = 0;  
// 		rdma_req_msg.params(7,0) = PARAMS_RNDZVS_WR_DONE;
// 		rdma_req_msg.params(HEADER_LENGTH+8-1,8) = (ap_uint<HEADER_LENGTH>)cmd_in_word;

// 		rdma_req.msg = (ap_uint<RDMA_MSG_BITS>)rdma_req_msg;
// 		STREAM_WRITE(rdma_sq, rdma_req);
// 	}

// 	// RDMA data transfer
// 	int bytes_processed  = 0;
	
// 	while(bytes_processed < bytes_to_process){
// 	#pragma HLS PIPELINE II=1
// 		stream_word outword;
// 		outword.dest = cmd_in_word.dst;
// 		outword.data = STREAM_READ(in).data;
		
// 		//signal ragged tail
// 		int bytes_left = (bytes_to_process - bytes_processed);
// 		if(bytes_left < bytes_per_word){
// 			outword.keep = (ap_uint<64>(1) << bytes_left)-1;
// 			bytes_processed += bytes_left;
// 		}else{
// 			outword.keep = -1;
// 			bytes_processed += bytes_per_word;
// 		}
// 		//if we run out of bytes, assert TLAST
// 		if(bytes_left <= bytes_per_word){
// 			outword.last = 1;
// 		}else{
// 			outword.last = 0;
// 		}
// 		//write output stream
// 		STREAM_WRITE(out, outword);
// 	}
// 	//acknowledge that message_seq has been sent successfully
// 	STREAM_WRITE(sts, cmd_in_word.seqn);

// 	// Post RDMA data transfer notification
// 	if (protocol == RNDZVS_PROTOC){
		
// 		rdma_req.opcode = RDMA_IMMED;
// 		rdma_req.qpn = cmd_in_word.dst;
// 		rdma_req.host = 0;
// 		rdma_req.mode = 0;
// 		rdma_req.last = 1;
// 		rdma_req.cmplt = 0;
// 		rdma_req.ssn = 0;
// 		rdma_req.offs = 0;
// 		rdma_req.rsrvd = 0;

// 		rdma_req_msg.lvaddr = 0;
// 		rdma_req_msg.rvaddr = 0;
// 		rdma_req_msg.len = 0;
// 		rdma_req_msg.params(7,0) = PARAMS_RNDZVS_WR_DONE;
// 		rdma_req_msg.params(HEADER_LENGTH+8-1,8) = (ap_uint<HEADER_LENGTH>)cmd_in_word;

// 		rdma_req.msg = (ap_uint<RDMA_MSG_BITS>)rdma_req_msg;
// 		STREAM_WRITE(rdma_sq, rdma_req);
// 	}

// }