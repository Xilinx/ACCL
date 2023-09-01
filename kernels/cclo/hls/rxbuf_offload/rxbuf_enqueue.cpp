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

#include "rxbuf_offload.h"
#include "Axi.h"

using namespace std;

void rxbuf_enqueue(
	STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &dma_cmd,
	STREAM<ap_uint<32> > &inflight_queue,
	unsigned int *rx_buffers
) {
#pragma HLS INTERFACE axis 		port=dma_cmd
#pragma HLS INTERFACE axis 		port=inflight_queue
#pragma HLS INTERFACE m_axi 	port=rx_buffers	depth=9*16 offset=slave num_read_outstanding=4	num_write_outstanding=4 bundle=mem
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS PIPELINE II=4

	unsigned int nbufs = 0;
	unsigned int max_len = 0;
	//poll nbuffers (base of rx buffers space) until it is non-zero
	//NOTE: software should write nbuffers *after* writing all the rest of the configuration
	if(nbufs == 0){
		nbufs = rx_buffers[0];
		if(nbufs == 0){
			return;
		}
		max_len = rx_buffers[1];
		rx_buffers += 2;
	}
	ap_uint<4> tag = 0;
	ap_axiu<104,0,0,DEST_WIDTH> cmd_word;
	hlslib::axi::Command<64, 23> cmd;
	#pragma HLS data_pack variable=dma_cmd struct_level
	//iterate until you run out of spare buffers
	for(int i=0; i < nbufs; i++){
		ap_uint<32> status;
		ap_uint<64> addr;
		status = rx_buffers[(i * SPARE_BUFFER_FIELDS) + STATUS_OFFSET];
		addr(31,  0) = rx_buffers[(i * SPARE_BUFFER_FIELDS) + ADDRL_OFFSET];
		addr(63, 32) = rx_buffers[(i * SPARE_BUFFER_FIELDS) + ADDRH_OFFSET];

		//look for IDLE spare buffers
		//can't be pipelined fully because of this test.
		if(status == STATUS_IDLE){
			//issue a cmd
			cmd.length = max_len;
			cmd.address = addr;
			cmd.tag = tag++;
			cmd_word.data = cmd;
			cmd_word.last = 1;//unused for now
			cmd_word.dest = 0;//unused for now
			STREAM_WRITE(dma_cmd, cmd_word);
			//update spare buffer status
			rx_buffers[(i * SPARE_BUFFER_FIELDS) + STATUS_OFFSET] = STATUS_ENQUEUED;
			//write to the in flight queue the spare buffer address in the exchange memory
			STREAM_WRITE(inflight_queue, i);
		}
	}
}

