# /*******************************************************************************
#  Copyright (C) 2024 Advanced Micro Devices, Inc
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

#include "dummy_cyt_dma.h"
#include "Axi.h"

using namespace std;

/*
This module just converts back from Coyote DMA command interface to a Xilinx DM interface for use in sim
*/
void cyt_dma(
	//Coyote Bypass interface command and status
	STREAM<cyt_req_t> &cyt_byp_wr_cmd,
	STREAM<ap_uint<16>> &cyt_byp_wr_sts,
	STREAM<cyt_req_t> &cyt_byp_rd_cmd,
	STREAM<ap_uint<16>> &cyt_byp_rd_sts,

	//DM command streams
	STREAM<ap_axiu<104,0,0,DEST_WIDTH>> &dma0_s2mm_cmd,
	STREAM<ap_axiu<104,0,0,DEST_WIDTH>> &dma1_s2mm_cmd,
	STREAM<ap_axiu<104,0,0,DEST_WIDTH>> &dma0_mm2s_cmd,
	STREAM<ap_axiu<104,0,0,DEST_WIDTH>> &dma1_mm2s_cmd,
	//DM status streams
	STREAM<ap_axiu<32,0,0,0>> &dma0_s2mm_sts,
	STREAM<ap_axiu<32,0,0,0>> &dma1_s2mm_sts,
	STREAM<ap_axiu<32,0,0,0>> &dma0_mm2s_sts,
	STREAM<ap_axiu<32,0,0,0>> &dma1_mm2s_sts

) {
#pragma HLS INTERFACE axis port=cyt_byp_rd_cmd
#pragma HLS INTERFACE axis port=cyt_byp_rd_sts
#pragma HLS INTERFACE axis port=cyt_byp_wr_cmd
#pragma HLS INTERFACE axis port=cyt_byp_wr_sts
#pragma HLS INTERFACE axis port=dma0_s2mm_cmd
#pragma HLS INTERFACE axis port=dma1_s2mm_cmd
#pragma HLS INTERFACE axis port=dma0_mm2s_cmd
#pragma HLS INTERFACE axis port=dma1_mm2s_cmd
#pragma HLS INTERFACE axis port=dma0_s2mm_sts
#pragma HLS INTERFACE axis port=dma1_s2mm_sts
#pragma HLS INTERFACE axis port=dma0_mm2s_sts
#pragma HLS INTERFACE axis port=dma1_mm2s_sts
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS aggregate variable=cyt_byp_wr_cmd compact=bit
#pragma HLS aggregate variable=cyt_byp_rd_cmd compact=bit

#pragma HLS pipeline II=1

ap_axiu<104,0,0,DEST_WIDTH> dm_cmd_with_dest;
hlslib::axi::Command<64, 23> dm_cmd;
cyt_req_t cyt_cmd;

dm_cmd.tag = 0;//zero out unused tag

if (!STREAM_IS_EMPTY(cyt_byp_wr_cmd)){
    cyt_cmd = STREAM_READ(cyt_byp_wr_cmd);
    dm_cmd.length = cyt_cmd.len(22,0);
    dm_cmd.address = cyt_cmd.vaddr;
    dm_cmd.eof = cyt_cmd.ctl;
    dm_cmd_with_dest.data = dm_cmd;
	dm_cmd_with_dest.dest = cyt_cmd.stream; // 1 if targeting host memory, 0 if targeting card memory
    dm_cmd_with_dest.last = 1;
    //forward to one or the other DMA based on dest 
    if(cyt_cmd.dest == 0){
        STREAM_WRITE(dma0_s2mm_cmd, dm_cmd_with_dest);
    } else if(cyt_cmd.dest == 1) {
        STREAM_WRITE(dma1_s2mm_cmd, dm_cmd_with_dest);
    }
}

if (!STREAM_IS_EMPTY(cyt_byp_rd_cmd)){
    cyt_cmd = STREAM_READ(cyt_byp_rd_cmd);
    dm_cmd.length = cyt_cmd.len(22,0);
    dm_cmd.address = cyt_cmd.vaddr;
    dm_cmd.eof = cyt_cmd.ctl;
    dm_cmd_with_dest.data = dm_cmd;
	dm_cmd_with_dest.dest = cyt_cmd.stream; // 1 if targeting host memory, 0 if targeting card memory
    dm_cmd_with_dest.last = 1;
    //forward to one or the other DMA based on dest 
    if(cyt_cmd.dest == 0){
        STREAM_WRITE(dma0_mm2s_cmd, dm_cmd_with_dest);
    } else if(cyt_cmd.dest == 1) {
        STREAM_WRITE(dma1_mm2s_cmd, dm_cmd_with_dest);
    }
}

ap_uint<16> byp_sts_word;
byp_sts_word(CYT_PID_BITS-1,0) = 0; //pid not used
byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS,CYT_DEST_BITS+CYT_PID_BITS) = 0;//strm not used
byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS+1,CYT_DEST_BITS+CYT_PID_BITS+1) = 0;//host not used

if (!STREAM_IS_EMPTY(dma0_s2mm_sts)){
    STREAM_READ(dma0_s2mm_sts);
    byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS-1,CYT_PID_BITS) = 0; //DMA0
    STREAM_WRITE(cyt_byp_wr_sts, byp_sts_word);
}

if (!STREAM_IS_EMPTY(dma1_s2mm_sts)){
    STREAM_READ(dma1_s2mm_sts);
    byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS-1,CYT_PID_BITS) = 1; //dest for DMA1
    STREAM_WRITE(cyt_byp_wr_sts, byp_sts_word);
}

if (!STREAM_IS_EMPTY(dma0_mm2s_sts)){
    STREAM_READ(dma0_mm2s_sts);
    byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS-1,CYT_PID_BITS) = 0; //DMA0
    STREAM_WRITE(cyt_byp_rd_sts, byp_sts_word);
}

if (!STREAM_IS_EMPTY(dma1_mm2s_sts)){
    STREAM_READ(dma1_mm2s_sts);
    byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS-1,CYT_PID_BITS) = 1; //DMA1
    STREAM_WRITE(cyt_byp_rd_sts, byp_sts_word);
}

}