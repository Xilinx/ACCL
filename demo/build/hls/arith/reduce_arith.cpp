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

template<unsigned int data_width, typename T>
void elementwise_sum(	stream<ap_axiu<data_width,0,0,0> > & in1,
				stream<ap_axiu<data_width,0,0,0> > & in2,
				stream<ap_axiu<data_width,0,0,0> > & out,
				unsigned int count) {
	
	unsigned const dwb = 8*sizeof(T);
	unsigned const simd = data_width / dwb;
	
	for (unsigned int i = 0; i < count; i++) {
#pragma HLS PIPELINE II=1
		ap_axiu<data_width,0,0,0> op1_word = in1.read();
		ap_axiu<data_width,0,0,0> op2_word = in2.read();
		ap_uint<data_width> op1 = op1_word.data;
		ap_uint<data_width> op2 = op2_word.data;
		ap_uint<data_width> res;
		for (unsigned int j = 0; j < simd; j++) {
#pragma HLS UNROLL
			ap_uint<dwb> op1_word = op1((j+1)*dwb-1,j*dwb);
			ap_uint<dwb> op2_word = op2((j+1)*dwb-1,j*dwb);
			T op1_word_t = *reinterpret_cast<T*>(&op1_word);
			T op2_word_t = *reinterpret_cast<T*>(&op2_word);
			T sum = op1_word_t + op2_word_t;
			ap_uint<dwb> res_word = *reinterpret_cast<ap_uint<dwb>*>(&sum);
			res((j+1)*dwb-1,j*dwb) = res_word;
		}
		ap_axiu<data_width,0,0,0> wword;
		wword.data = res;
		wword.last = (i == (count-1));
		wword.keep = op1_word.keep & op2_word.keep;
		out.write(wword);
	}
//TODO: copy control signals (last/keep) from inputs to output
}

void reduce_arith(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in1,
			stream<ap_axiu<DATA_WIDTH,0,0,0> > & in2,
			stream<ap_axiu<DATA_WIDTH,0,0,0> > & out,
			unsigned int count,
			unsigned int function) {
#pragma HLS INTERFACE axis register both port=in1
#pragma HLS INTERFACE axis register both port=in2
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE s_axilite port=count
#pragma HLS INTERFACE s_axilite port=function
#pragma HLS INTERFACE s_axilite port=return
if(function == 0)
	elementwise_sum<DATA_WIDTH, float>(in1, in2, out, count);
else if(function == 1)
	elementwise_sum<DATA_WIDTH, double>(in1, in2, out, count);
else if(function == 2)
	elementwise_sum<DATA_WIDTH, int>(in1, in2, out, count);
else
	elementwise_sum<DATA_WIDTH, long>(in1, in2, out, count);
//TODO: add MAX and other reduction functions
}
