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

using namespace std;

void dma2seg_cmd(STREAM<ap_axiu<104,0,0,DEST_WIDTH> > & dma_cmd_in,
			STREAM<ap_axiu<104,0,0,DEST_WIDTH> > & dma_cmd_out,
			STREAM<segmenter_cmd > & seg_cmd){
#pragma HLS INTERFACE axis register both port=dma_cmd_in
#pragma HLS INTERFACE axis register both port=dma_cmd_out
#pragma HLS INTERFACE axis register both port=seg_cmd
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1
	segmenter_cmd cmd;
	// get DMA command from upstream
	ap_axiu<104,0,0,DEST_WIDTH> dma_cmd = STREAM_READ(dma_cmd_in);
	// forward the DMA command downstream
	STREAM_WRITE(dma_cmd_out, dma_cmd);
	// get the count and convert to words
	ap_uint<23> count_bytes = dma_cmd.data(22,0);
	cmd.nwords = count_bytes(22,6) + (count_bytes(5,0) != 0);
	cmd.dest = dma_cmd.dest;//copy dest from command to data
	cmd.emit_ack = false;
	cmd.indeterminate_btt = true;
	STREAM_WRITE(seg_cmd, cmd);
}