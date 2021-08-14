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

void hp_fp_stream_conv(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
            stream<ap_axiu<DATA_WIDTH,0,0,0> > & out);

int main(){

    stream<ap_axiu<DATA_WIDTH,0,0,0> > in("in");
    stream<ap_axiu<DATA_WIDTH,0,0,0> > out("out");
    stream<ap_axiu<DATA_WIDTH,0,0,0> > golden("golden");
    
    ap_axiu<DATA_WIDTH,0,0,0> goldenword1;
    ap_axiu<DATA_WIDTH,0,0,0> goldenword2;
    ap_axiu<DATA_WIDTH,0,0,0> outword;
    ap_axiu<DATA_WIDTH,0,0,0> inword;

    srand(42);

    int len = 128;
    float f1, f2;
    half h1, h2;

    for(int i=len; i>0; i-=(DATA_WIDTH/8)){
        goldenword1.last = (i <= (DATA_WIDTH/16));
        goldenword2.last = (i > (DATA_WIDTH/16)) & (i <= (DATA_WIDTH/8));
        if(goldenword1.last){
            goldenword1.keep = ((long long)1<<2*i)-1;
            goldenword2.keep = 0;
        } else if(goldenword2.last && i != DATA_WIDTH/8){
            goldenword1.keep = -1;
            goldenword2.keep = ((long long)1<<2*(i-(DATA_WIDTH/16)))-1;
        } else {
            goldenword1.keep = -1;
            goldenword2.keep = -1;
        }
        for(int j=0; j<(DATA_WIDTH/32); j++){
            h1 = (static_cast <half> (rand()) / static_cast <half> (RAND_MAX));
            h2 = (static_cast <half> (rand()) / static_cast <half> (RAND_MAX));
            f1 = (float)h1;
            f2 = (float)h2;
            goldenword1.data((j+1)*32-1,j*32) = *reinterpret_cast<ap_uint<32>*>(&f1);
            goldenword2.data((j+1)*32-1,j*32) = *reinterpret_cast<ap_uint<32>*>(&f2);
            inword.data((j+1)*16-1,j*16) = *reinterpret_cast<ap_uint<16>*>(&h1);
            inword.data((j+(DATA_WIDTH/32)+1)*16-1,(j+(DATA_WIDTH/32))*16) = *reinterpret_cast<ap_uint<16>*>(&h2);
            inword.keep((j+1)*2-1,j*2) = goldenword1.keep((j+1)*4-1,j*4) == 15 ? 3 : 0;
            inword.keep((j+(DATA_WIDTH/32)+1)*2-1,(j+(DATA_WIDTH/32))*2) = goldenword2.keep((j+1)*4-1,j*4) == 15 ? 3 : 0;
        }
        inword.last = goldenword1.last | goldenword2.last;
        golden.write(goldenword1);
        cout << "Writing golden stream" << endl;
        cout << hex << goldenword1.keep << dec << endl;
        cout << goldenword1.last << endl;
        if(goldenword1.last != 1){
            golden.write(goldenword2);
            cout << "Writing golden stream" << endl;
            cout << hex << goldenword2.keep << dec << endl;
            cout << goldenword2.last << endl;
        }
        in.write(inword);
        cout << "Writing input stream" << endl;
        cout << hex << inword.keep << dec << endl;
        cout << inword.last << endl;
    }
    
    hp_fp_stream_conv(in, out);
    
    //parse data
    for(int i=2*len; i>0; i-=(DATA_WIDTH/8)){
        outword = out.read();
        cout << "Reading output stream" << endl;
        cout << hex << outword.keep << dec << endl;
        cout << outword.last << endl;
        goldenword1 = golden.read();
        cout << "Reading golden stream" << endl;
        cout << hex << goldenword1.keep << dec << endl;
        cout << goldenword1.last << endl;

        if(outword.last != goldenword1.last){
            cout << "Mismatch on tlast" << endl;
            return 1;
        }
        if(outword.keep != goldenword1.keep){
            cout << "Mismatch on tkeep" << endl;
            return 1;
        }

        for(int j=0; j<DATA_WIDTH/32; j++){
            if(outword.data((j+1)*32-1,j*32) != goldenword1.data((j+1)*32-1,j*32) && outword.keep((j+1)*4-1,j*4) == 15){
                cout << hex << outword.data << ":" << goldenword1.data << dec << endl;
                return 1;
            }
        }

    }
    
    return 0;
}
