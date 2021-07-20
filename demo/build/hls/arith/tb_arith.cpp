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

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

void reduce_arith(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in1,
			stream<ap_axiu<DATA_WIDTH,0,0,0> > & in2,
			stream<ap_axiu<DATA_WIDTH,0,0,0> > & out,
			unsigned int count,
			unsigned int function);

int main(){

	stream<ap_axiu<DATA_WIDTH,0,0,0> > in1;
	stream<ap_axiu<DATA_WIDTH,0,0,0> > in2;
	stream<ap_axiu<DATA_WIDTH,0,0,0> > out;
	stream<ap_axiu<DATA_WIDTH,0,0,0> > golden;
	
	ap_axiu<DATA_WIDTH,0,0,0> inword;
	ap_axiu<DATA_WIDTH,0,0,0> outword;
	ap_axiu<DATA_WIDTH,0,0,0> goldenword;

	//1024B transfer
	int len = 1024;

	for(int i=0; i<(len/64); i++){
		inword.data = i;
		inword.last = (i == len/64-1);
		if(inword.last == 1) inword.keep = 0xf;
		in1.write(inword);
		inword.data = 2*i;
		if(inword.last == 1) inword.keep = 0xff;
		in2.write(inword);
		inword.data = 3*i;
		if(inword.last == 1) inword.keep = 0xf; //this needs to be op1.keep & op2.keep
		golden.write(inword);
	}
	
	reduce_arith(in1, in2, out, len/64, 2);
	
	//parse data
	for(int i=0; i<len/64; i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data){
			cout << hex << outword.data << ":" << goldenword.data << dec << endl;
			return 1;
		}
		if(outword.last != goldenword.last) return 1;
		if(outword.keep != goldenword.keep) return 1;
		int last = outword.last;
		if(last == 0 && i==(len/64-1)) return 1;
	}
	
	return 0;
}
