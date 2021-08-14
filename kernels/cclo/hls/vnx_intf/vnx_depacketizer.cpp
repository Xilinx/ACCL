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
#include "hls_stream.h"
#include "ap_int.h"

using namespace hls;
using namespace std;

#define DATA_WIDTH 512
#define HEADER_COUNT_START 0
#define HEADER_COUNT_END   31
#define HEADER_TAG_START   HEADER_COUNT_END+1
#define HEADER_TAG_END	   HEADER_TAG_START+31
#define HEADER_SRC_START   HEADER_TAG_END+1
#define HEADER_SRC_END	   HEADER_SRC_START+31
#define HEADER_SEQ_START   HEADER_SRC_END+1
#define HEADER_SEQ_END	   HEADER_SEQ_START+31

void vnx_depacketizer(	stream<ap_axiu<DATA_WIDTH,0,0,16> > & in,
			stream<ap_axiu<DATA_WIDTH,0,0,0> > & out,
			stream<ap_uint<32> > & sts) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE s_axilite port=return

unsigned const bytes_per_word = DATA_WIDTH/8;

//copy count from header into sts stream
ap_axiu<DATA_WIDTH,0,0,16> inword = in.read();
ap_axiu<DATA_WIDTH,0,0,0> outword;
//read header and put in sts stream
int count 	= (inword.data)(HEADER_COUNT_END, HEADER_COUNT_START);
int tag 	= (inword.data)(HEADER_TAG_END	, HEADER_TAG_START 	);
int src 	= (inword.data)(HEADER_SRC_END	, HEADER_SRC_START 	);
int seq 	= (inword.data)(HEADER_SEQ_END	, HEADER_SEQ_START	);

sts.write(count);
sts.write(tag);
sts.write(src);
sts.write(seq);

while(count > 0){
#pragma HLS PIPELINE II=1
	inword = in.read();
	outword.data = inword.data;
	outword.keep = inword.keep;
	count -= bytes_per_word;
	if(count <= 0){
		outword.last = 1;
	}else{
		outword.last = 0;
	}
	out.write(outword);
}

}
