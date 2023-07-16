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
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "eth_intf.h"
#include "rxbuf_offload.h"

using namespace std;


void rdma_rq_handler(
	STREAM<rdma_req_t> & rdma_rq, 
	STREAM<rdma_req_t> & rdma_rq_fwd,
	STREAM<rxbuf_notification> & notify

) {
#pragma HLS INTERFACE axis register both port=rdma_rq
#pragma HLS INTERFACE axis register both port=rdma_rq_fwd
#pragma HLS INTERFACE axis register both port=notify
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

// aggregate compact bit required as goes external to Coyote
#pragma HLS aggregate variable=rdma_rq compact=bit
	
	static rdma_req_t rdma_req;
	static rdma_req_msg_t rdma_req_msg;
	static ap_uint<RDMA_PARAMS_BITS> params;
	static ap_uint<8> params_opcode = 0; 
	static eth_header hdr;
	static rxbuf_notification rxbuf_notif;
	static rxbuf_signature rxbuf_sig;
	static rdma_meta_upd_t meta_upd;

	if (!STREAM_IS_EMPTY(rdma_rq))
	{
		rdma_req = STREAM_READ(rdma_rq);

		// if it is not a completion signal
		if (rdma_req.cmplt != 1)
		{
			// if it stems from send with immediate data check the params within the message field of the request
			if (rdma_req.opcode == RDMA_IMMED)
			{
				rdma_req_msg = rdma_req.msg;
				params = rdma_req_msg.params;
				// get the opcode
				params_opcode = params(7,0);
				if (params_opcode == PARAMS_RNDZVS_INIT){
					// forward this command for decoding elsewhere
					STREAM_WRITE(rdma_rq_fwd, rdma_req);
				} else if (params_opcode == PARAMS_RNDZVS_WR_DONE){
					hdr = eth_header(params(HEADER_LENGTH+8-1,8));
					rxbuf_sig.tag = hdr.tag;
					rxbuf_sig.len = hdr.count;
					rxbuf_sig.src = hdr.src;
					rxbuf_sig.seqn = hdr.seqn;
					rxbuf_notif.index = 0; // TODO: figure out the mechanism of this index
					rxbuf_notif.signature = rxbuf_sig;
					STREAM_WRITE(notify, rxbuf_notif);
				} else if (params_opcode == PARAMS_EGR_HEADER){
					hdr = eth_header(params(HEADER_LENGTH+8-1,8));
				}
			} 
			// // if it stems from send without immediate data
			// else if (rdma_req.opcode == RDMA_SEND)
			// {
			// 	// decide what to do
			// }
		}

	}
}