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
#define HEADER_COUNT_START 0
#define HEADER_COUNT_END   31
#define HEADER_TAG_START   HEADER_COUNT_END+1
#define HEADER_TAG_END	   HEADER_TAG_START+31
#define HEADER_SRC_START   HEADER_TAG_END+1
#define HEADER_SRC_END	   HEADER_SRC_START+31
#define HEADER_SEQ_START   HEADER_SRC_END+1
#define HEADER_SEQ_END	   HEADER_SEQ_START+31



void tcp_packetizer(stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
			stream<ap_axiu<DATA_WIDTH,0,0,0> > & out,
			stream<ap_uint<32> > & cmd,
			stream<ap_uint<96> > & cmd_txHandler,
			stream<ap_uint<32> > & sts,
			unsigned int max_pktsize
			)
{
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=cmd
#pragma HLS INTERFACE axis register both port=cmd_txHandler
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE s_axilite port=max_pktsize
#pragma HLS INTERFACE s_axilite port=return

	unsigned const bytes_per_word = DATA_WIDTH/8;

	//read commands from the command stream
	unsigned int session 	 = cmd.read()(15,0);
	int message_bytes 		 = cmd.read();
	int message_tag 		 = cmd.read();
	int message_src 		 = cmd.read();
	int message_seq 		 = cmd.read();
	int bytes_to_process = message_bytes + bytes_per_word;

	//send command to txHandler
	ap_uint<96> tx_cmd;
	tx_cmd(31,0) 	= session;
	tx_cmd(63,32) 	= bytes_to_process;
	tx_cmd(95,64) 	= max_pktsize;
	cmd_txHandler.write(tx_cmd);

	unsigned int pktsize = 0;
	int bytes_processed  = 0;

	ap_axiu<DATA_WIDTH,0,0,0> outword;

	bool setHeader = false;

	//if this is the first word and the message payload fifo is not empty, put the count in a header
	while(setHeader == false)
	{
		if (!in.empty()) 
		{
		#pragma HLS PIPELINE II=1
			outword.data(HEADER_COUNT_END, HEADER_COUNT_START) 	= message_bytes;
			outword.data(HEADER_TAG_END	 , HEADER_TAG_START  )  = message_tag;
			outword.data(HEADER_SRC_END	 , HEADER_SRC_START  )  = message_src;
			outword.data(HEADER_SEQ_END	 , HEADER_SEQ_START  )  = message_seq;
			outword.keep = -1;
			outword.last = 0;
			out.write(outword);
			setHeader = true;
		}
	}
	
	// send the message
	while(bytes_processed < message_bytes){
	#pragma HLS PIPELINE II=1
		
		outword.data = in.read().data;
	
		//signal ragged tail
		int bytes_left = (message_bytes - bytes_processed);
		if(bytes_left < bytes_per_word){
			outword.keep = (1 << bytes_left)-1;
			bytes_processed += bytes_left;
		}else{
			outword.keep = -1;
			bytes_processed += bytes_per_word;
		}
		pktsize++;
		//after every max_pktsize words, or if we run out of bytes, assert TLAST
		if((pktsize == max_pktsize) || (bytes_left <= bytes_per_word)){
			outword.last = 1;
			pktsize = 0;
		}else{
			outword.last = 0;
		}
		//write output stream
		out.write(outword);
	}
	//acknowledge that message_seq has been sent successfully
	sts.write(message_seq)
}