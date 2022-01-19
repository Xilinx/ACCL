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

#include "ap_int.h"
#include "fp_hp_stream_conv.h"

using namespace hls;
using namespace std;

void fp_hp_stream_conv(STREAM<stream_word> & in, STREAM<stream_word> & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

    unsigned const simd_in = DATA_WIDTH / 32;
    unsigned const simd_out = DATA_WIDTH / 16;

    int done = 0; 
    stream_word in_block;
    ap_uint<DATA_WIDTH> in_data;
    ap_uint<DATA_WIDTH> res;
    stream_word wword;

    while(done == 0) {
#pragma HLS PIPELINE II=1
        in_block = STREAM_READ(in);
        in_data = in_block.data;
        for (unsigned int j = 0; j < simd_in; j++) {
#pragma HLS UNROLL
            ap_uint<32> in_word = in_data((j+1)*32-1,j*32);
            float fp_word = *reinterpret_cast<float*>(&in_word);
            half hp_word = (half)fp_word;
            ap_uint<16> res_word = *reinterpret_cast<ap_uint<16>*>(&hp_word);
            res((j+1)*16-1,j*16) = res_word;
        }
        for (unsigned int j=0; j<simd_in; j++){
#pragma HLS UNROLL
            wword.keep((j+1)*2-1,j*2) = (in_block.keep((j+1)*4-1,j*4) == 15) ? 3 : 0;
        }
        if(in_block.last == 1){
            wword.data = res;
            wword.last = 1;
            wword.keep((DATA_WIDTH/8)-1,(DATA_WIDTH/8)/2) = 0;
            STREAM_WRITE(out, wword);
            break;
        }
        in_block = STREAM_READ(in);
        in_data = in_block.data;
        for (unsigned int j = 0; j < simd_in; j++) {
#pragma HLS UNROLL
            ap_uint<32> in_word = in_data((j+1)*32-1,j*32);
            float fp_word = *reinterpret_cast<float*>(&in_word);
            half hp_word = (half)fp_word;
            ap_uint<16> res_word = *reinterpret_cast<ap_uint<16>*>(&hp_word);
            res((j+simd_in+1)*16-1,(j+simd_in)*16) = res_word;
        }
        for (unsigned int j=0; j<simd_in; j++){
#pragma HLS UNROLL
            wword.keep((j+simd_in+1)*2-1,(j+simd_in)*2) = (in_block.keep((j+1)*4-1,j*4) == 15) ? 3 : 0;
        }
        wword.data = res;
        wword.last = in_block.last;
        STREAM_WRITE(out, wword);
        done = (in_block.last == 1);
    }


}
