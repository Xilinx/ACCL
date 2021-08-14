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

using namespace hls;
using namespace std;

#ifndef DATA_WIDTH
#define DATA_WIDTH 512
#endif

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#define specialization(dt, dw) reduce_sum_ ## dt ## _ ## dw
#define top(dt, dw) specialization(dt, dw)

template<unsigned int data_width, typename T>
void stream_add(stream<ap_axiu<2*data_width,0,0,0> > & in,
                stream<ap_axiu<data_width,0,0,0> > & out) {

	unsigned const dwb = 8*sizeof(T);
	unsigned const simd = data_width / dwb;
	
	int done = 0; 

	while(done == 0) {
#pragma HLS PIPELINE II=1
		ap_axiu<2*data_width,0,0,0> op_block = in.read();
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
		ap_axiu<data_width,0,0,0> wword;
		wword.data = res;
		wword.last = op_block.last;
		wword.keep = op_block.keep;
		out.write(wword);

		done = (op_block.last == 1);
	}


}

void top(DATA_TYPE, DATA_WIDTH)(stream<ap_axiu<2*DATA_WIDTH,0,0,0> > & in,
				stream<ap_axiu<DATA_WIDTH,0,0,0> > & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
stream_add<DATA_WIDTH, DATA_TYPE>(in, out);
}
