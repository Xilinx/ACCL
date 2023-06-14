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

#pragma once

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "accl_hls.h"

using namespace std;

#define CYT_VADDR_BITS 48
#define CYT_LEN_BITS 28
#define CYT_DEST_BITS 4
#define CYT_PID_BITS 6
#define CYT_N_REGIONS_BITS 1

void cyt_dma_adapter(
	//DM command streams
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma0_s2mm_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma1_s2mm_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma0_mm2s_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma1_mm2s_cmd,
	//DM status streams
	hls::stream<ap_axiu<32,0,0,0>> &dma0_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma0_mm2s_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_mm2s_sts,
	//Coyote Bypass interface command and status
	hls::stream<ap_uint<96>> &cyt_byp_wr_cmd,
	hls::stream<ap_uint<16>> &cyt_byp_wr_sts,
	hls::stream<ap_uint<96>> &cyt_byp_rd_cmd,
	hls::stream<ap_uint<16>> &cyt_byp_rd_sts
);