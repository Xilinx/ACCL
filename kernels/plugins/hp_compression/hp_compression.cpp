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
#include "hp_compression.h"

using namespace hls;
using namespace std;

template<unsigned int data_width>
ap_uint<data_width/2> fp2hp(ap_uint<data_width> in){
#pragma HLS inline

    unsigned const simd = data_width / 32;  
    ap_uint<32> in_word;
    ap_uint<16> res_word;
    ap_uint<data_width/2> res;

    for (unsigned int j = 0; j < simd; j++) {
#pragma HLS UNROLL
        in_word = in((j+1)*32-1,j*32);
        float fp_word = *reinterpret_cast<float*>(&in_word);
        half hp_word = (half)fp_word;
        res_word = *reinterpret_cast<ap_uint<16>*>(&hp_word);
        res((j+1)*16-1,j*16) = res_word;
    }

    return res;
}

template<unsigned int data_width>
ap_uint<2*data_width> hp2fp(ap_uint<data_width> in){
#pragma HLS inline

    unsigned const simd = data_width / 16;  
    ap_uint<16> in_word;
    ap_uint<32> res_word;
    ap_uint<2*data_width> res;

    for (unsigned int j = 0; j < simd; j++) {
#pragma HLS UNROLL
        in_word = in((j+1)*16-1,j*16);
        half hp_word = *reinterpret_cast<half*>(&in_word);
        float fp_word = (float)hp_word;
        res_word = *reinterpret_cast<ap_uint<32>*>(&fp_word);
        res((j+1)*32-1,j*32) = res_word;
    }

    return res;
}

void hp_compression(STREAM<stream_word> & in, STREAM<stream_word> & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

    unsigned const simd = DATA_WIDTH / 32;

    stream_word in_block;
    stream_word wword;
    bool first_word = true;

    in_block = STREAM_READ(in);
    if(in_block.dest == 0){
        //convert float32 to float16
        do {
    #pragma HLS PIPELINE II=1 style=flp
            if(first_word){
                first_word = false;
            } else{
                in_block = STREAM_READ(in);
            }
            wword.data(DATA_WIDTH/2-1,0) = fp2hp<DATA_WIDTH>(in_block.data);
            for (unsigned int j=0; j<simd; j++){
    #pragma HLS UNROLL
                wword.keep((j+1)*2-1,j*2) = (in_block.keep((j+1)*4-1,j*4) == 15) ? 3 : 0;
            }
            if(in_block.last == 1){
                wword.data(DATA_WIDTH-1,DATA_WIDTH/2) = 0;
                wword.last = 1;
                wword.keep((DATA_WIDTH/8)-1,(DATA_WIDTH/8)/2) = 0;
                STREAM_WRITE(out, wword);
                break;
            }
            in_block = STREAM_READ(in);
            wword.data(DATA_WIDTH-1,DATA_WIDTH/2) = fp2hp<DATA_WIDTH>(in_block.data);
            for (unsigned int j=0; j<simd; j++){
    #pragma HLS UNROLL
                wword.keep((j+simd+1)*2-1,(j+simd)*2) = (in_block.keep((j+1)*4-1,j*4) == 15) ? 3 : 0;
            }
            wword.last = in_block.last;
            STREAM_WRITE(out, wword);
        } while(in_block.last != 1);
    } else{
        //convert float16 to float32
        do {
    #pragma HLS PIPELINE II=1 style=flp
            if(first_word){
                first_word = false;
            } else{
                in_block = STREAM_READ(in);
            }
            wword.data = hp2fp<DATA_WIDTH/2>(in_block.data(DATA_WIDTH/2-1,0));
            for (unsigned int j=0; j<simd; j++){
    #pragma HLS UNROLL
                wword.keep((j+1)*4-1,j*4) = (in_block.keep((j+1)*2-1,j*2) == 3) ? 15 : 0;
            }
            wword.last = (in_block.last == 1) && (in_block.keep((DATA_WIDTH/8)-1,(DATA_WIDTH/8)/2) == 0);
            STREAM_WRITE(out, wword);
            if(wword.last == 1){
                break;
            }
            wword.data = hp2fp<DATA_WIDTH>(in_block.data(DATA_WIDTH-1,DATA_WIDTH/2));
            for (unsigned int j=0; j<simd; j++){
    #pragma HLS UNROLL
                wword.keep((j+1)*4-1,j*4) = (in_block.keep((j+simd+1)*2-1,(j+simd)*2) == 3) ? 15 : 0;
            }
            wword.last = in_block.last;
            STREAM_WRITE(out, wword);
        } while(in_block.last != 1);
    }


}
