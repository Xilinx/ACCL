/*******************************************************************************
#  Copyright (C) 2022 Advanced Micro Devices, Inc
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
#
*******************************************************************************/

#include <vadd_put.h>

//hls-synthesizable function performing
//an elementwise increment on fp32 data in src 
//followed by a put to another rank, then
//writing result in dst
void vadd_put(
    float *src,
    float *dst,
    int count,
    unsigned int destination,
    //parameters pertaining to CCLO config
    ap_uint<32> comm_adr, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
){
#pragma HLS INTERFACE s_axilite port=count
#pragma HLS INTERFACE s_axilite port=destination
#pragma HLS INTERFACE s_axilite port=comm_adr
#pragma HLS INTERFACE s_axilite port=dpcfg_adr
#pragma HLS INTERFACE m_axi port=src offset=slave
#pragma HLS INTERFACE m_axi port=dst offset=slave
#pragma HLS INTERFACE axis port=cmd_to_cclo
#pragma HLS INTERFACE axis port=sts_from_cclo
#pragma HLS INTERFACE axis port=data_to_cclo
#pragma HLS INTERFACE axis port=data_from_cclo
#pragma HLS INTERFACE s_axilite port=return
    //set up interfaces
    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    //read data from src, increment it, 
    //and push the result into the CCLO stream
    ap_uint<512> tmpword;
    int word_count = 0;
    int rd_count = count;
    while(rd_count > 0){
        //read 16 floats into a 512b vector
        for(int i=0; (i<16) && (rd_count>0); i++){
            float inc = src[i+16*word_count]+1;
            tmpword((i+1)*32-1, i*32) = *reinterpret_cast<ap_uint<32>*>(&inc);
            rd_count--;
        }
        //send the vector to cclo
        data.push(tmpword, 0);
        word_count++;
    }
    //send command to CCLO
    //we're passing src as source and targeting stream 9
    //because we're streaming data, the address will be ignored
    accl.stream_put(count, 9, destination, (ap_uint<64>)src);
    //pull data from CCLO and write it to dst
    int wr_count = count;
    word_count = 0;
    while(wr_count > 0){
        //read vector from CCLO
        tmpword = data.pull().data;
        //read from the 512b vector into 16 floats
        for(int i=0; (i<16) && (wr_count>0); i++){
            ap_uint<32> val = tmpword((i+1)*32-1, i*32);
            dst[i+16*word_count] = *reinterpret_cast<float*>(&val);
            wr_count--;
        }
        word_count++;
    }
}