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

#include "stream_segmenter.h"

using namespace hls;
using namespace std;

int ntransfers(int nbytes){
	int bytes_per_transfer = DATA_WIDTH/8;
	return (nbytes+bytes_per_transfer-1)/bytes_per_transfer;
}

int main(){

	stream<ap_axiu<DATA_WIDTH,0,0,16> > in;
	stream<ap_axiu<DATA_WIDTH,0,0,16> > out;
	stream<ap_axiu<DATA_WIDTH,0,0,16> > golden;
	stream<segmenter_cmd > cmd;
	stream<ap_uint<32> > sts;
	
	segmenter_cmd cmdword;
	ap_axiu<DATA_WIDTH,0,0,16> inword;
	ap_axiu<DATA_WIDTH,0,0,16> outword;
	ap_axiu<DATA_WIDTH,0,0,16> goldenword;
	
	int len = 100;
	int dest = 3;
	cmdword.nwords = len;
	cmdword.dest = dest; 
	cmd.write(cmdword);

	for(int i=0; i<ntransfers(len); i++){
		inword.data = i;
		inword.last = (i==(ntransfers(len)-1));
		inword.dest = 1;
		in.write(inword);
		golden.write(inword);
	}
	
	krnl_packetizer(in, out, cmd, sts);
	
	//parse data
	for(int i=0; i<ntransfers(len); i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data) return 1;
		if(outword.last != goldenword.last) return 1;
		if(outword.dest != dest) return 1;
	}
	if(sts.read() != ntransfers(len))	return 1;
	
	return 0;
}