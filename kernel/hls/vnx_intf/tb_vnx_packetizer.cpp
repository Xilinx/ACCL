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

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

void vnx_packetizer(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
			stream<ap_axiu<DATA_WIDTH,0,0,16> > & out,
			stream<ap_uint<32> > & cmd,
			unsigned int max_pktsize);

int ntransfers(int nbytes){
	int bytes_per_transfer = DATA_WIDTH/8;
	return (nbytes+bytes_per_transfer-1)/bytes_per_transfer;
}

int main(){

	stream<ap_axiu<DATA_WIDTH,0,0,0> > in;
	stream<ap_axiu<DATA_WIDTH,0,0,16> > out;
	stream<ap_axiu<DATA_WIDTH,0,0,0> > golden;
	stream<ap_uint<32> > cmd;
	stream<ap_uint<32> > sts;
	
	ap_axiu<DATA_WIDTH,0,0,0> inword;
	ap_axiu<DATA_WIDTH,0,0,16> outword;
	ap_axiu<DATA_WIDTH,0,0,0> goldenword;
	
	int dest 	= 3;
	int len 	= 50;
	int tag 	= 5;
	int src 	= 6;
	int seq  	= 7;
	//1024B+64B transfer
	len = 50;
	cmd.write(dest);
	cmd.write(len);
	cmd.write(tag);
	cmd.write(src);
	cmd.write(seq);

	for(int i=0; i<ntransfers(len); i++){
		inword.data = i;
		inword.last = (i==(ntransfers(len)-1));
		in.write(inword);
		golden.write(inword);
	}
	
	vnx_packetizer(in, out, cmd, sts, 1536/64);
	
	//parse header
	outword = out.read();
	if(outword.dest != dest) return 1;
	if(outword.last != 0) return 1;
	if(outword.data(31,0) != len) return 1;
	if(outword.data(63,32) != tag) return 1;
	if(outword.data(95,64) != src) return 1;
	if(outword.data(127,96) != seq) return 1;
	//parse data
	for(int i=0; i<ntransfers(len); i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data) return 1;
		if(outword.last != goldenword.last) return 1;
	}
	if(sts.data(31,90) != message_seq)	return 1;
	
	//1536B transfer
	seq++;
	len = 1536-64;
	cmd.write(dest);
	cmd.write(len);
	cmd.write(tag);
	cmd.write(src);
	cmd.write(seq);
	
	for(int i=0; i<ntransfers(len); i++){
		inword.data = i;
		inword.last = (i==(ntransfers(len)-1));
		in.write(inword);
		golden.write(inword);
	}
	
	vnx_packetizer(in, out, cmd, 1536/64);
	
	//parse header
	outword = out.read();
	if(outword.dest != dest) return 1;
	if(outword.last != 0) return 1;
	if(outword.data(31,0) != len) return 1;
	if(outword.data(63,32) != tag) return 1;
	if(outword.data(95,64) != src) return 1;
	if(outword.data(127,96) != seq) return 1;
	//parse data
	for(int i=0; i<ntransfers(len); i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data) return 1;
		if(outword.last != goldenword.last) return 1;
	}
	if(sts.data(31,90) != message_seq)	return 1;
	
	//10KB transfer	
	seq++;
	len = 10*1024;
	cmd.write(dest);
	cmd.write(len);
	cmd.write(tag);
	cmd.write(src);
	cmd.write(seq);

	for(int i=0; i<ntransfers(len); i++){
		inword.data = i;
		inword.last = (i==(ntransfers(len)-1));
		in.write(inword);
		golden.write(inword);
	}
	
	vnx_packetizer(in, out, cmd, 1536/64);
	
	//parse header
	outword = out.read();
	if(outword.dest != dest) return 1;
	if(outword.last != 0) return 1;
	if(outword.data(31,0) != len) return 1;
	if(outword.data(63,32) != tag) return 1;
	if(outword.data(95,64) != src) return 1;
	if(outword.data(127,96) != seq) return 1;

	//parse data
	for(int i=0; i<ntransfers(len); i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data) return 1;
		if((i+2)*64 % 1536 == 0){
			int last = outword.last;
			if(last == 0) return 1;
		} else {
			if(outword.last != goldenword.last) return 1;
		}
	}
	if(sts.data(31,90) != message_seq)	return 1;

	return 0;
}