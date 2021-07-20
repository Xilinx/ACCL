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
			unsigned int max_pktsize
			)
{
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=cmd
#pragma HLS INTERFACE axis register both port=cmd_txHandler
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
	
	while(bytes_processed < bytes_to_process){
	#pragma HLS PIPELINE II=1
		ap_axiu<DATA_WIDTH,0,0,0> outword;
		//if this is the first word, put the count in a header
		if(bytes_processed == 0){
			outword.data(HEADER_COUNT_END, HEADER_COUNT_START) 	= message_bytes;
			outword.data(HEADER_TAG_END	 , HEADER_TAG_START  )  = message_tag;
			outword.data(HEADER_SRC_END	 , HEADER_SRC_START  )  = message_src;
			outword.data(HEADER_SEQ_END	 , HEADER_SEQ_START  )  = message_seq;
		} else {
			outword.data = in.read().data;
		}
		//signal ragged tail
		int bytes_left = (bytes_to_process - bytes_processed);
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
}