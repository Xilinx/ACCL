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

#include "vnx.h"

using namespace hls;
using namespace std;

int main(){

	stream<ap_axiu<DATA_WIDTH,0,0,16> > in;
	stream<ap_axiu<DATA_WIDTH,0,0,16> > out;
	stream<ap_axiu<DATA_WIDTH,0,0,16> > golden;
	stream<ap_axiu<128,0,0,0> > sts;
	
	ap_axiu<DATA_WIDTH,0,0,16> inword;
	ap_axiu<DATA_WIDTH,0,0,16> outword;
	ap_axiu<128,0,0,0> stsword;
	ap_axiu<DATA_WIDTH,0,0,16> goldenword;
	
	int len;
	int tag = 5;
	int src = 6;
	int seq = 42;
	int strm = 9;

	//50B+64B transfer
	len = 50;
	int nwords = (len+63)/64;

	for(int i=0; i<nwords+1; i++){
		if(i==0){
			inword.data(31,0) 	= len;
			inword.data(63,32) 	= tag;
			inword.data(95,64) 	= src;
			inword.data(127,96) = seq;
			inword.data(159,128) = seq;
		} else {
			inword.data = i;
		}
		inword.last = (i%3 == 0) || (i==nwords);
		in.write(inword);
		if(i > 0)
			golden.write(inword);
	}
	
	vnx_depacketizer(in, out, sts);
	
	//parse header
	stsword = sts.read();
	if(stsword.data(31,0) != len) return 1;
	if(stsword.data(63,32) != tag) return 1;
	if(stsword.data(95,64) != src) return 1;
	if(stsword.data(127,96) != seq) return 1;
	//parse data
	for(int i=0; i<nwords; i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data) return 1;
		if(outword.dest != strm) return 1;
		int last = outword.last;
		if(i==nwords-1){
			if(last == 0) return 1;
		} else {
			if(last == 1) return 1;
		}
	}
	
	
	return 0;
}
