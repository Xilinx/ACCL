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

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

void hp_stream_add(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in1,
            stream<ap_axiu<DATA_WIDTH,0,0,0> > & in2,
            stream<ap_axiu<DATA_WIDTH,0,0,0> > & out);

int main(){

    stream<ap_axiu<DATA_WIDTH,0,0,0> > in1;
    stream<ap_axiu<DATA_WIDTH,0,0,0> > in2;
    stream<ap_axiu<DATA_WIDTH,0,0,0> > out;
    stream<ap_axiu<DATA_WIDTH,0,0,0> > golden;
    
    ap_axiu<DATA_WIDTH,0,0,0> inword1;
    ap_axiu<DATA_WIDTH,0,0,0> inword2;
    ap_axiu<DATA_WIDTH,0,0,0> outword;
    ap_axiu<DATA_WIDTH,0,0,0> goldenword;

    srand(42);

    //130B transfer
    int len = 130;
    half h1, h2;

    for(int i=len; i>0; i-=(DATA_WIDTH/8)){
        for(int j=0; j<(DATA_WIDTH/16); j++){
            h1 = (static_cast <half> (rand()) / static_cast <half> (RAND_MAX));
            h2 = (static_cast <half> (rand()) / static_cast <half> (RAND_MAX));
            half sum = h1+h2;
            inword1.data((j+1)*16-1,j*16) = *reinterpret_cast<ap_uint<DATA_WIDTH>*>(&h1);
            inword2.data((j+1)*16-1,j*16) = *reinterpret_cast<ap_uint<DATA_WIDTH>*>(&h2);
            goldenword.data((j+1)*16-1,j*16) = *reinterpret_cast<ap_uint<DATA_WIDTH>*>(&sum);
        }
        inword1.last = (i < (DATA_WIDTH/8));
        inword1.keep = (((long long)1<<i)-1);
        inword2.last = inword1.last;
        inword2.keep = inword1.keep;
        goldenword.last = inword1.last;
        goldenword.keep = inword1.keep;
        in1.write(inword1);
        in2.write(inword2);
        golden.write(goldenword);
    }
    
    hp_stream_add(in1, in2, out);
    
    //parse data
    for(int i=len; i>0; i-=(DATA_WIDTH/8)){
        outword = out.read();
        goldenword = golden.read();
        if(outword.data != goldenword.data){
            cout << hex << outword.data << ":" << goldenword.data << dec << endl;
            return 1;
        }
        if(outword.last != goldenword.last) return 1;
        if(outword.keep != goldenword.keep) return 1;
    }
    
    return 0;
}
