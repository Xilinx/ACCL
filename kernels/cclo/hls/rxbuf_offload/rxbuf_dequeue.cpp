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

void rxbuf_dequeue(
	STREAM<ap_uint<32> > &dma_sts,
	STREAM<eth_header> &eth_hdr,
	STREAM<ap_uint<32> > &inflight_queue,
	STREAM<rxbuf_notification> &notification_queue,
	unsigned int *rx_buffers
) {
#pragma HLS INTERFACE axis 		port=dma_sts
#pragma HLS INTERFACE axis 		port=eth_hdr
#pragma HLS INTERFACE axis 		port=inflight_queue
#pragma HLS INTERFACE axis 		port=notification_queue
#pragma HLS INTERFACE m_axi 	port=rx_buffers depth=16*9 offset=slave num_read_outstanding=4 num_write_outstanding=4  bundle=mem
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS PIPELINE II=1 style=flp
	//get rx_buffer pointer from inflight queue
	ap_uint<32> spare_idx = STREAM_READ(inflight_queue), btt, new_status;
	eth_header header = STREAM_READ(eth_hdr);
	rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + spare_idx * SPARE_BUFFER_FIELDS + RX_TAG_OFFSET] = header.tag;
	rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + spare_idx * SPARE_BUFFER_FIELDS + RX_LEN_OFFSET] = header.count;
	rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + spare_idx * SPARE_BUFFER_FIELDS + RX_SRC_OFFSET] = header.src;
	rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + spare_idx * SPARE_BUFFER_FIELDS + SEQUENCE_NUMBER_OFFSET] = header.seqn;
	hlslib::axi::Status dma_status = hlslib::axi::Status(STREAM_READ(dma_sts));
	//interpret dma sts and write new spare_sts
	// 3-0 TAG 
	// 4 INTERNAL 	ERROR usually a btt=0 trigger this
	// 5 DECODE 	ERROR address decode error timeout
	// 6 SLAVE 		ERROR DMA encountered a slave reported error
	// 7 OKAY		the associated transfer command has been completed with the OKAY response on all intermediate transfers.
	if(!dma_status.okay | dma_status.slaveError | dma_status.decodeError | dma_status.internalError | (dma_status.bytesReceived != header.count)){
		new_status = STATUS_ERROR;
	}else{
		new_status = STATUS_RESERVED;
	}
	rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + spare_idx * SPARE_BUFFER_FIELDS + STATUS_OFFSET] = new_status;
	//send to the DMA mover data required to identify/resolve a receive: tag, src, count, address
	rxbuf_notification s;
	// s.addr(31, 0) = rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + spare_idx * SPARE_BUFFER_FIELDS + ADDRL_OFFSET];
	// s.addr(63,32) = rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + spare_idx * SPARE_BUFFER_FIELDS + ADDRH_OFFSET];
	s.index = spare_idx;
	s.signature.tag = header.tag;
	s.signature.src = header.src;
	s.signature.len = header.count;
	s.signature.seqn = header.seqn;
	STREAM_WRITE(notification_queue, s);
}
