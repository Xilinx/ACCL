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

#include "call_probe.h"


void call_probe(
	bool capture,
	ap_uint<32> count,
	ap_uint<32> *mem,
	STREAM<command_word> &cmd_upstream,
	STREAM<command_word> &ack_upstream,
	STREAM<command_word> &cmd_downstream,
	STREAM<command_word> &ack_downstream
){
#pragma HLS INTERFACE axis register both port=cmd_upstream
#pragma HLS INTERFACE axis register both port=ack_upstream
#pragma HLS INTERFACE axis register both port=cmd_downstream
#pragma HLS INTERFACE axis register both port=ack_downstream
#pragma HLS INTERFACE m_axi port=mem depth=160 offset=slave num_write_outstanding=4 bundle=mem
#pragma HLS INTERFACE s_axilite port=capture
#pragma HLS INTERFACE s_axilite port=count
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS PIPELINE II=1

	STREAM<ap_uint<32>> tmpbuf;
	#pragma HLS STREAM variable=tmpbuf depth=16

	command_word w;
	ap_uint<32> duration;

	for(unsigned cnt=0; cnt<count; cnt++){
		//forward call and copy arguments to our local buffer
		for(int  i = 0; i < 15; i++) {
			w = STREAM_READ(cmd_upstream);
			STREAM_WRITE(cmd_downstream, w);
			if(capture){
				STREAM_WRITE(tmpbuf, w.data);
			}
		}
		if(capture){
			duration = 0;
			//increment duration while call is ongoing
			while(STREAM_IS_EMPTY(ack_downstream)){
				#pragma HLS PIPELINE II=1
				duration++;
			}
		}
		//forward acknowledgement
		STREAM_WRITE(ack_upstream, STREAM_READ(ack_downstream));
		if(capture){
			//save duration
			STREAM_WRITE(tmpbuf, duration);
			//dump local buffer to memory
			for(int i=0; i<16; i++) {
				mem[i+16*cnt] = STREAM_READ(tmpbuf);
			}
		}
	}
}
