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

#pragma once

#include "accl_hls.h"
#include "cyt.h"

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

);