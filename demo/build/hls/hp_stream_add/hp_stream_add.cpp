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

void hp_stream_add(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in1,
                    stream<ap_axiu<DATA_WIDTH,0,0,0> > & in2,
                    stream<ap_axiu<DATA_WIDTH,0,0,0> > & out) {
#pragma HLS INTERFACE axis register both port=in1
#pragma HLS INTERFACE axis register both port=in2
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

	unsigned const dwb = 16;
	unsigned const simd = DATA_WIDTH / dwb;
	
	int done = 0; 

	while(done == 0) {
#pragma HLS PIPELINE II=1
		ap_axiu<DATA_WIDTH,0,0,0> op1_block = in1.read();
		ap_axiu<DATA_WIDTH,0,0,0> op2_block = in2.read();
		ap_uint<DATA_WIDTH> op1 = op1_block.data;
		ap_uint<DATA_WIDTH> op2 = op2_block.data;
		ap_uint<DATA_WIDTH> res;
		for (unsigned int j = 0; j < simd; j++) {
#pragma HLS UNROLL
			ap_uint<dwb> op1_word = op1((j+1)*dwb-1,j*dwb);
			ap_uint<dwb> op2_word = op2((j+1)*dwb-1,j*dwb);
			half op1_word_t = *reinterpret_cast<half*>(&op1_word);
			half op2_word_t = *reinterpret_cast<half*>(&op2_word);
			half sum = op1_word_t + op2_word_t;
			ap_uint<dwb> res_word = *reinterpret_cast<ap_uint<dwb>*>(&sum);
			res((j+1)*dwb-1,j*dwb) = res_word;
		}
		ap_axiu<DATA_WIDTH,0,0,0> wword;
		wword.data = res;
		wword.last = op1_block.last & op1_block.last;
		wword.keep = op1_block.keep & op2_block.keep;
		out.write(wword);

		done = (op1_block.last == 1);
	}


}
