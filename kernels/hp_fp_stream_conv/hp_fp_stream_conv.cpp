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
#include "hp_fp_stream_conv.h"

using namespace hls;
using namespace std;

void hp_fp_stream_conv(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
                    stream<ap_axiu<DATA_WIDTH,0,0,0> > & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

    unsigned const simd = DATA_WIDTH / 32;

    int done = 0; 
    ap_axiu<DATA_WIDTH,0,0,0> in_block;
    ap_uint<DATA_WIDTH/2> in_data1, in_data2;
    ap_uint<DATA_WIDTH/16> in_keep1, in_keep2;
    ap_uint<DATA_WIDTH> res;
    ap_axiu<DATA_WIDTH,0,0,0> wword;

    while(done == 0) {
#pragma HLS PIPELINE II=1
        in_block = in.read();
        in_data1 = in_block.data(DATA_WIDTH/2-1,0);
        in_data2 = in_block.data(DATA_WIDTH-1,DATA_WIDTH/2);
        in_keep1 = in_block.keep(DATA_WIDTH/16-1,0);
        in_keep2 = in_block.keep(DATA_WIDTH/8-1,DATA_WIDTH/16);
        for (unsigned int j = 0; j < simd; j++) {
#pragma HLS UNROLL
            ap_uint<16> in_word = in_data1((j+1)*16-1,j*16);
            half hp_word = *reinterpret_cast<half*>(&in_word);
            float fp_word = (float)hp_word;
            ap_uint<32> res_word = *reinterpret_cast<ap_uint<32>*>(&fp_word);
            res((j+1)*32-1,j*32) = res_word;
        }
        for (unsigned int j=0; j<simd; j++){
#pragma HLS UNROLL
            wword.keep((j+1)*4-1,j*4) = (in_keep1((j+1)*2-1,j*2) == 3) ? 15 : 0;
        }
        wword.data = res;
        wword.last = (in_block.last == 1) && (in_keep2 == 0);
        out.write(wword);
        if(wword.last == 1){
            break;
        }
        for (unsigned int j = 0; j < simd; j++) {
#pragma HLS UNROLL
            ap_uint<16> in_word = in_data2((j+1)*16-1,j*16);
            half hp_word = *reinterpret_cast<half*>(&in_word);
            float fp_word = (float)hp_word;
            ap_uint<32> res_word = *reinterpret_cast<ap_uint<32>*>(&fp_word);
            res((j+1)*32-1,j*32) = res_word;
        }
        for (unsigned int j=0; j<simd; j++){
#pragma HLS UNROLL
            wword.keep((j+1)*4-1,j*4) = (in_keep2((j+1)*2-1,j*2) == 3) ? 15 : 0;
        }
        wword.data = res;
        wword.last = in_block.last;
        out.write(wword);
        done = (in_block.last == 1);
    }


}
