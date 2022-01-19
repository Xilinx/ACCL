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

void fp_hp_stream_conv(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
            stream<ap_axiu<DATA_WIDTH,0,0,0> > & out);

int main(){

    stream<ap_axiu<DATA_WIDTH,0,0,0> > in;
    stream<ap_axiu<DATA_WIDTH,0,0,0> > out;
    stream<ap_axiu<DATA_WIDTH,0,0,0> > golden;
    
    ap_axiu<DATA_WIDTH,0,0,0> inword1;
    ap_axiu<DATA_WIDTH,0,0,0> inword2;
    ap_axiu<DATA_WIDTH,0,0,0> outword;
    ap_axiu<DATA_WIDTH,0,0,0> goldenword;

    srand(42);

    //864B of inputs, 432B of outputs
    int len = 864;
    float f1, f2;
    half h1, h2;

    for(int i=len; i>0; i-=2*(DATA_WIDTH/8)){
        inword1.last = (i <= (DATA_WIDTH/8));
        inword2.last = (i > (DATA_WIDTH/8)) & (i <= 2*(DATA_WIDTH/8));
        if(inword1.last){ 
            inword1.keep = ((long long)1<<i)-1;
            inword2.keep = 0;
        } else if(inword2.last){
            inword1.keep = -1;
            inword2.keep = ((long long)1<<(i-(DATA_WIDTH/8)))-1;
        } else {
            inword1.keep = -1;
            inword2.keep = -1;
        }
        for(int j=0; j<(DATA_WIDTH/32); j++){
            f1 = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
            f2 = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
            inword1.data((j+1)*32-1,j*32) = *reinterpret_cast<ap_uint<32>*>(&f1);
            inword2.data((j+1)*32-1,j*32) = *reinterpret_cast<ap_uint<32>*>(&f2);
            h1 = (half)f1;
            h2 = (half)f2;
            goldenword.data((j+1)*16-1,j*16) = *reinterpret_cast<ap_uint<16>*>(&h1);
            goldenword.data((j+(DATA_WIDTH/32)+1)*16-1,(j+(DATA_WIDTH/32))*16) = *reinterpret_cast<ap_uint<16>*>(&h2);
            goldenword.keep((j+1)*2-1,j*2) = inword1.keep((j+1)*4-1,j*4) == 15 ? 3 : 0;
            goldenword.keep((j+(DATA_WIDTH/32)+1)*2-1,(j+(DATA_WIDTH/32))*2) = inword2.keep((j+1)*4-1,j*4) == 15 ? 3 : 0;
        }
        goldenword.last = inword1.last | inword2.last;
        in.write(inword1);
        cout << "Writing input stream" << endl;
        cout << hex << inword1.keep << dec << endl;
        cout << inword1.last << endl;
        if(inword1.last != 1){
            in.write(inword2);
            cout << "Writing input stream" << endl;
            cout << hex << inword2.keep << dec << endl;
            cout << inword2.last << endl;
        }
        golden.write(goldenword);
        cout << "Writing golden stream" << endl;
        cout << hex << goldenword.keep << dec << endl;
        cout << goldenword.last << endl;
    }
    
    fp_hp_stream_conv(in, out);
    
    //parse data
    for(int i=len/2; i>0; i-=(DATA_WIDTH/8)){
        outword = out.read();
        cout << "Reading output stream" << endl;
        cout << hex << outword.keep << dec << endl;
        cout << outword.last << endl;
        goldenword = golden.read();
        cout << "Reading golden stream" << endl;
        cout << hex << goldenword.keep << dec << endl;
        cout << goldenword.last << endl;

        if(outword.last != goldenword.last){
            cout << "Mismatch on tlast" << endl;
            return 1;
        }
        if(outword.keep != goldenword.keep){
            cout << "Mismatch on tkeep" << endl;
            return 1;
        }

        for(int j=0; j<DATA_WIDTH/16; j++){
            if(outword.data((j+1)*16-1,j*16) != goldenword.data((j+1)*16-1,j*16) && outword.keep((j+1)*2-1,j*2) == 3){
                cout << hex << outword.data << ":" << goldenword.data << dec << endl;
                return 1;
            }
        }

    }
    
    return 0;
}
