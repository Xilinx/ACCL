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
#include <stdint.h>
#include "reduce_sum.h"

using namespace hls;
using namespace std;

template<unsigned int data_width, unsigned int dest_width, typename T>
void stream_add(STREAM<ap_axiu<2*data_width,0,0,dest_width> > & in,
                STREAM<ap_axiu<data_width,0,0,dest_width> > & out) {

	unsigned const dwb = 8*sizeof(T);
	unsigned const simd = data_width / dwb;
	
	int done = 0; 

	while(done == 0) {
#pragma HLS PIPELINE II=1
		ap_axiu<2*data_width,0,0,dest_width> op_block = STREAM_READ(in);
		ap_uint<data_width> op1 = op_block.data(data_width-1,0);
		ap_uint<data_width> op2 = op_block.data(2*data_width-1,data_width);
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
		ap_axiu<data_width,0,0,dest_width> wword;
		wword.data = res;
		wword.last = op_block.last;
		wword.keep = op_block.keep;
		STREAM_WRITE(out, wword);

		done = (op_block.last == 1);
	}
}

void reduce_sum_float(STREAM<ap_axiu<2*DATA_WIDTH,0,0,DEST_WIDTH> > & in, STREAM<stream_word> & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
stream_add<DATA_WIDTH, DEST_WIDTH, float>(in, out);
}

void reduce_sum_double(STREAM<ap_axiu<2*DATA_WIDTH,0,0,DEST_WIDTH> > & in, STREAM<stream_word> & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
stream_add<DATA_WIDTH, DEST_WIDTH, double>(in, out);
}

void reduce_sum_int32_t(STREAM<ap_axiu<2*DATA_WIDTH,0,0,DEST_WIDTH> > & in, STREAM<stream_word> & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
stream_add<DATA_WIDTH, DEST_WIDTH, int32_t>(in, out);
}

void reduce_sum_int64_t(STREAM<ap_axiu<2*DATA_WIDTH,0,0,DEST_WIDTH> > & in, STREAM<stream_word> & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
stream_add<DATA_WIDTH, DEST_WIDTH, int64_t>(in, out);
}

#ifdef REDUCE_HALF_PRECISION
void reduce_sum_half(STREAM<ap_axiu<2*DATA_WIDTH,0,0,DEST_WIDTH> > & in, STREAM<stream_word> & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
stream_add<DATA_WIDTH, DEST_WIDTH, half>(in, out);
}
#endif