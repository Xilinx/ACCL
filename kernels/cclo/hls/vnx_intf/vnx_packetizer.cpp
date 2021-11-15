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

#include "vnx.h"

using namespace hls;
using namespace std;

void vnx_packetizer(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
			stream<ap_axiu<DATA_WIDTH,0,0,16> > & out,
			stream<ap_uint<32> > & cmd,
			stream<ap_uint<32> > & sts,
			unsigned int max_pktsize) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=cmd
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE s_axilite port=max_pktsize
#pragma HLS INTERFACE s_axilite port=return

unsigned const bytes_per_word = DATA_WIDTH/8;

//read commands from the command stream
unsigned int destination = cmd.read()(15,0);
int message_bytes 		 = cmd.read();
int message_tag 		 = cmd.read();
int message_src 		 = cmd.read();
int message_seq 		 = cmd.read();
int bytes_to_process = message_bytes + bytes_per_word;

unsigned int pktsize = 0;
int bytes_processed  = 0;

while(bytes_processed < bytes_to_process){
#pragma HLS PIPELINE II=1
	ap_axiu<DATA_WIDTH,0,0,16> outword;
	outword.dest = destination;
	//if this is the first word, put the count in a header
	if(bytes_processed == 0){
		outword.data(HEADER_COUNT_END, HEADER_COUNT_START) 	= message_bytes;
		outword.data(HEADER_TAG_END	 , HEADER_TAG_START  )  = message_tag;
		outword.data(HEADER_SRC_END	 , HEADER_SRC_START  )  = message_src;
		outword.data(HEADER_SEQ_END	 , HEADER_SEQ_START  )  = message_seq;		
	} else {
		outword.data = in.read().data;
	}
	//signal ragged tail
	int bytes_left = (bytes_to_process - bytes_processed);
	if(bytes_left < bytes_per_word){
		ap_uint<bytes_per_word> keep = 1;
		keep = keep << bytes_left;
		keep -= 1;
		outword.keep = keep;
		bytes_processed += bytes_left;
	}else{
		outword.keep = -1;
		bytes_processed += bytes_per_word;
	}
	pktsize++;
	//after every max_pktsize words, or if we run out of bytes, assert TLAST
	if((pktsize == max_pktsize) || (bytes_left <= bytes_per_word)){
		outword.last = 1;
		pktsize = 0;
	}else{
		outword.last = 0;
	}
	//write output stream
	out.write(outword);
}
//acknowledge that message_seq has been sent successfully
ap_uint<32> outsts;
sts.write(message_seq);
}