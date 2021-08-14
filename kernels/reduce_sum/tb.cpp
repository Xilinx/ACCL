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
#include<iostream>
#include <cstdlib>
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

void top(DATA_TYPE, DATA_WIDTH)(stream<ap_axiu<2*DATA_WIDTH,0,0,0> > & in,
				stream<ap_axiu<DATA_WIDTH,0,0,0> > & out);

int main(){

    stream<ap_axiu<2*DATA_WIDTH,0,0,0> > in;
    stream<ap_axiu<DATA_WIDTH,0,0,0> > out;
    stream<ap_axiu<DATA_WIDTH,0,0,0> > golden;
    
    ap_uint<DATA_WIDTH> inword1;
    ap_uint<DATA_WIDTH> inword2;
    ap_axiu<2*DATA_WIDTH,0,0,0> inword;
    ap_axiu<DATA_WIDTH,0,0,0> outword;
    ap_axiu<DATA_WIDTH,0,0,0> goldenword;

    srand(42);

    //130B transfer
    int len = 130;
    DATA_TYPE h1, h2;
    int dwt = 8*sizeof(DATA_TYPE);
    int simd = DATA_WIDTH/dwt;

    for(int i=len; i>0; i-=(DATA_WIDTH/8)){
        for(int j=0; j<simd; j++){
            h1 = (static_cast <DATA_TYPE> (rand()) / static_cast <DATA_TYPE> (RAND_MAX));
            h2 = (static_cast <DATA_TYPE> (rand()) / static_cast <DATA_TYPE> (RAND_MAX)); 
            DATA_TYPE sum = h1+h2;
            inword1((j+1)*dwt-1,j*dwt) = *reinterpret_cast<ap_uint<DATA_WIDTH>*>(&h1);
            inword2((j+1)*dwt-1,j*dwt) = *reinterpret_cast<ap_uint<DATA_WIDTH>*>(&h2);
            goldenword.data((j+1)*dwt-1,j*dwt) = *reinterpret_cast<ap_uint<DATA_WIDTH>*>(&sum);
        }
        inword.data(DATA_WIDTH-1, 0) = inword1;
        inword.data(2*DATA_WIDTH-1, DATA_WIDTH) = inword2;
        inword.last = (i < (DATA_WIDTH/8));
        inword.keep = (((long long)1<<i)-1);
        goldenword.last = inword.last;
        goldenword.keep = inword.keep;
        in.write(inword);
        golden.write(goldenword);
    }
    
    top(DATA_TYPE, DATA_WIDTH)(in, out);
    
    //parse data
    for(int i=len; i>0; i-=(DATA_WIDTH/8)){
        outword = out.read();
        goldenword = golden.read();
        if(outword.data != goldenword.data){
            cout << hex << outword.data << ":" << goldenword.data << dec << endl;
            for(int j=0; j<simd; j++){
                cout << "Output Word " << j << (DATA_TYPE)outword.data((j+1)*dwt-1,j*dwt) << endl;
                cout << "Golden Word " << j << (DATA_TYPE)goldenword.data((j+1)*dwt-1,j*dwt) << endl;
            }
            return 1;
        }
        if(outword.last != goldenword.last) return 1;
        if(outword.keep != goldenword.keep) return 1;
    }
    
    return 0;
}
