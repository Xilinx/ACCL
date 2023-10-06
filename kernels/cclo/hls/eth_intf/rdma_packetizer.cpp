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

void rdma_packetizer(
	STREAM<stream_word > & in,
	STREAM<stream_word > & out,
	STREAM<eth_header> & cmd,
	STREAM<ap_uint<32> > & sts,
	unsigned int max_pktsize
)
{
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=cmd
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE s_axilite port=max_pktsize
#pragma HLS INTERFACE s_axilite port=return

	unsigned const bytes_per_word = DATA_WIDTH/8;

	//read commands from the command stream
	eth_header cmdword = STREAM_READ(cmd);

	int bytes_to_process;
	
	// RNDZVS_MSG doesn't have header
	if (cmdword.msg_type == RNDZVS_MSG){
		bytes_to_process = cmdword.count;
	} else if (cmdword.msg_type == RNDZVS_INIT || cmdword.msg_type == RNDZVS_WR_DONE){
		bytes_to_process = bytes_per_word;
	} else {
		bytes_to_process = cmdword.count + bytes_per_word;
	}

	int bytes_processed  = 0;
	
	while(bytes_processed < bytes_to_process){
	#pragma HLS PIPELINE II=1
		stream_word outword;
		outword.dest = cmdword.dst;
		//if this is the first word and it is not the RNDZVS_MSG, put the header
		if((bytes_processed == 0) && (cmdword.msg_type != RNDZVS_MSG)){
			outword.data(DATA_WIDTH-1, HEADER_LENGTH) = 0;
			outword.data(HEADER_LENGTH-1,0) = (ap_uint<HEADER_LENGTH>)cmdword;
		} else {
			outword.data = STREAM_READ(in).data;
		}
		//signal ragged tail
		int bytes_left = (bytes_to_process - bytes_processed);
		if(bytes_left < bytes_per_word){
			outword.keep = (ap_uint<64>(1) << bytes_left)-1;
			bytes_processed += bytes_left;
		}else{
			outword.keep = -1;
			bytes_processed += bytes_per_word;
		}
		//if we run out of bytes, assert TLAST
		if(bytes_left <= bytes_per_word){
			outword.last = 1;
		}else{
			outword.last = 0;
		}
		//write output stream
		STREAM_WRITE(out, outword);
	}
	// acknowledge to the DMP that message_seq has been sent successfully
	// only ack to RNDZVS_MSG and EGR_MSG that are managed by DMP
	ap_uint<32> outsts;
	if (cmdword.msg_type == RNDZVS_MSG || cmdword.msg_type == EGR_MSG){
		STREAM_WRITE(sts, cmdword.seqn);
	}
}