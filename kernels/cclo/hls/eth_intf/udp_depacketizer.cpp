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
#include <iostream>

using namespace std;

void udp_depacketizer(
    STREAM<stream_word > & in,
    STREAM<stream_word > & out,
    STREAM<eth_header> & sts) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE s_axilite port=return

unsigned const bytes_per_word = DATA_WIDTH/8;

//copy count from header into sts stream
stream_word inword = STREAM_READ(in);
stream_word outword;
eth_header hdr = eth_header(inword.data(HEADER_LENGTH-1,0));
int count = hdr.count;

if(hdr.strm == 0){
	STREAM_WRITE(sts, hdr);
}

while(count > 0){
#pragma HLS PIPELINE II=1
	inword = STREAM_READ(in);
	outword.data = inword.data;
	outword.keep = inword.keep;
	outword.dest = hdr.strm;
	count -= bytes_per_word;
	if(count <= 0){
		outword.last = 1;
	}else{
		outword.last = 0;
	}
	STREAM_WRITE(out, outword);
}

}
