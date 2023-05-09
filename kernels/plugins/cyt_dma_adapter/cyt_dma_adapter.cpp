/*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
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

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

using namespace std;

void cyt_dma_adapter(
	//DM command streams
	hls::stream<ap_uint<104>> &dma0_s2mm_cmd,
	hls::stream<ap_uint<104>> &dma1_s2mm_cmd,
	hls::stream<ap_uint<104>> &dma0_mm2s_cmd,
	hls::stream<ap_uint<104>> &dma1_mm2s_cmd,
	//DM status streams
	hls::stream<ap_axiu<32,0,0,0>> &dma0_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma0_mm2s_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_mm2s_sts,
	//DM data streams
	hls::stream<ap_axiu<512,0,0,0>> &dma0_s2mm,
	hls::stream<ap_axiu<512,0,0,0>> &dma1_s2mm,
	hls::stream<ap_axiu<512,0,0,0>> &dma0_mm2s,
	hls::stream<ap_axiu<512,0,0,0>> &dma1_mm2s,
	//Coyote Bypass interface command and status
	hls::stream<ap_uint<96>> &cyt_byp_wr_cmd,
	hls::stream<ap_uint<6>> &cyt_byp_wr_sts,
	hls::stream<ap_uint<96>> &cyt_byp_rd_cmd,
	hls::stream<ap_uint<6>> &cyt_byp_rd_sts,
	//Coyote data interfaces (2x)
	hls::stream<ap_axiu<512,0,0,0>> &cyt_dma0_s2mm,
	hls::stream<ap_axiu<512,0,0,0>> &cyt_dma1_s2mm,
	hls::stream<ap_axiu<512,0,0,0>> &cyt_dma0_mm2s,
	hls::stream<ap_axiu<512,0,0,0>> &cyt_dma1_mm2s
) {
#pragma HLS INTERFACE axis port=dma0_s2mm_cmd
#pragma HLS INTERFACE axis port=dma1_s2mm_cmd
#pragma HLS INTERFACE axis port=dma0_mm2s_cmd
#pragma HLS INTERFACE axis port=dma1_mm2s_cmd
#pragma HLS INTERFACE axis port=dma0_s2mm_sts
#pragma HLS INTERFACE axis port=dma1_s2mm_sts
#pragma HLS INTERFACE axis port=dma0_mm2s_sts
#pragma HLS INTERFACE axis port=dma1_mm2s_sts
#pragma HLS INTERFACE axis port=dma0_s2mm
#pragma HLS INTERFACE axis port=dma1_s2mm
#pragma HLS INTERFACE axis port=dma0_mm2s
#pragma HLS INTERFACE axis port=dma1_mm2s
#pragma HLS INTERFACE axis port=cyt_byp_rd_cmd
#pragma HLS INTERFACE axis port=cyt_byp_rd_sts
#pragma HLS INTERFACE axis port=cyt_byp_wr_cmd
#pragma HLS INTERFACE axis port=cyt_byp_wr_sts
#pragma HLS INTERFACE axis port=cyt_dma0_s2mm
#pragma HLS INTERFACE axis port=cyt_dma1_s2mm
#pragma HLS INTERFACE axis port=cyt_dma0_mm2s
#pragma HLS INTERFACE axis port=cyt_dma1_mm2s
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW disable_start_propagation
	//TODO:
	//
}
