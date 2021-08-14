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

int ntransfers(int nbytes){
	int bytes_per_transfer = DATA_WIDTH/8;
	return (nbytes+bytes_per_transfer-1)/bytes_per_transfer;
}

void tcp_packetizer(stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
			stream<ap_axiu<DATA_WIDTH,0,0,0> > & out,
			stream<ap_uint<32> > & cmd,
			stream<ap_uint<96> > & cmd_txHandler,
			unsigned int max_pktsize
			);

int main()
{
	stream<ap_axiu<DATA_WIDTH,0,0,0> >  in;
	stream<ap_axiu<DATA_WIDTH,0,0,0> >  out;
	stream<ap_axiu<DATA_WIDTH,0,0,0> > golden;
	stream<ap_uint<32> >  cmd;
	stream<ap_uint<32> >  sts;
	stream<ap_uint<96> >  cmd_txHandler;

	ap_axiu<DATA_WIDTH,0,0,0> inword;
	ap_axiu<DATA_WIDTH,0,0,0> outword;
	ap_axiu<DATA_WIDTH,0,0,0> goldenword;

	ap_uint<96> tx_cmd;

	unsigned int max_pktsize = 1536/64;

	unsigned int session ;
	int message_bytes ;
	int message_tag ;
	int message_src ;
	int message_seq ;

	//send 1024 byte message
	session 		= 0;
	message_bytes 	= 1024;
	message_tag 	= 5;
	message_src 	= 1;
	message_seq		= 42;

	cmd.write(session);
	cmd.write(message_bytes);
	cmd.write(message_tag);
	cmd.write(message_src);
	cmd.write(message_seq);


	for(int i=0; i<ntransfers(message_bytes); i++){
		inword.data = i;
		inword.last = (i==(ntransfers(message_bytes)-1));
		in.write(inword);
		golden.write(inword);
	}

	tcp_packetizer(in,
			out,
			cmd,
			cmd_txHandler,
			sts,
			max_pktsize
			);

	//read cmd_txHandler
	tx_cmd = cmd_txHandler.read();
	if (tx_cmd(31,0) 	!= session) 			  return 1;
	if (tx_cmd(63,32) 	!= (message_bytes + 64) ) return 1;
	if (tx_cmd(95,64) 	!= max_pktsize) 		  return 1;

	//parse header
	outword = out.read();
	if(outword.last != 0)						return 1;
	if(outword.data(31,0) 	!= message_bytes) 	return 1;
	if(outword.data(63,32) 	!= message_tag) 	return 1;
	if(outword.data(95,64) 	!= message_src) 	return 1;
	if(outword.data(127,96) != message_seq) 	return 1;

	//parse data
	for(int i=0; i<ntransfers(message_bytes); i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data) return 1;
		if(outword.last != goldenword.last) return 1;
	}
	if(sts.data(31,90) != message_seq)	return 1;


	//send 1536-64 byte message
	session 		= 0;
	message_bytes 	= 1536-64;
	message_tag 	= 5;
	message_src 	= 1;
	message_seq		= 43;


	cmd.write(session);
	cmd.write(message_bytes);
	cmd.write(message_tag);
	cmd.write(message_src);
	cmd.write(message_seq);

	for(int i=0; i<ntransfers(message_bytes); i++){
		inword.data = i;
		inword.last = (i==(ntransfers(message_bytes)-1));
		in.write(inword);
		golden.write(inword);
	}

	tcp_packetizer(in,
			out,
			cmd,
			cmd_txHandler,
			max_pktsize
			);

	//read cmd_txHandler
	tx_cmd = cmd_txHandler.read();
	if (tx_cmd(31,0) 	!= session) 			  	return 1;
	if (tx_cmd(63,32) 	!= (message_bytes + 64) ) 	return 1;
	if (tx_cmd(95,64) 	!= max_pktsize) 			return 1;

	//parse header
	outword = out.read();
	if(outword.last != 0) return 1;
	if(outword.data(31,0	) != message_bytes) return 1;
	if(outword.data(63,32	) != message_tag) 	return 1;
	if(outword.data(95,64	) != message_src) 	return 1;
	if(outword.data(127,96	) != message_seq)	return 1;

	//parse data
	for(int i=0; i<ntransfers(message_bytes); i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data) return 1;
		if(outword.last != goldenword.last) return 1;
	}
	if(sts.data(31,90) != message_seq)	return 1;

	//send 10k byte message
	session 		= 0;
	message_bytes 	= 10*1024;
	message_tag 	= 5;
	message_src 	= 1;
	message_seq 	= 44;

	cmd.write(session);
	cmd.write(message_bytes);
	cmd.write(message_tag);
	cmd.write(message_src);
	cmd.write(message_seq);

	for(int i=0; i<ntransfers(message_bytes); i++){
		inword.data = i;
		inword.last = (i==(ntransfers(message_bytes)-1));
		in.write(inword);
		golden.write(inword);
	}

	tcp_packetizer(in,
			out,
			cmd,
			cmd_txHandler,
			max_pktsize
			);

	//read cmd_txHandler
	tx_cmd = cmd_txHandler.read();
	if (tx_cmd(31,0) != session) return 1;
	if (tx_cmd(63,32) != (message_bytes + 64) ) return 1;
	if (tx_cmd(95,64) != max_pktsize) return 1;

	//parse header
	outword = out.read();
	if(outword.last != 0) return 1;
	if(outword.data(31,0) 	!= message_bytes) return 1;
	if(outword.data(63,32) 	!= message_tag	) return 1;
	if(outword.data(95,64) 	!= message_src	) return 1;
	if(outword.data(127,96) != message_seq	) return 1;

	//parse data
	for(int i=0; i<ntransfers(message_bytes); i++){
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