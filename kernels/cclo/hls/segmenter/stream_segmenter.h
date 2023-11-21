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

#pragma once

#include "accl_hls.h"

typedef struct{
	ap_uint<32> nwords;
	ap_uint<16> dest;
	bool emit_ack;
	bool indeterminate_btt;
} segmenter_cmd;

void stream_segmenter(STREAM<stream_word > & in,
			STREAM<stream_word > & out,
			STREAM<segmenter_cmd > & cmd,
			STREAM<ap_uint<32> > & sts);

void dma2seg_cmd(STREAM<ap_axiu<104,0,0,DEST_WIDTH>  > & dma_cmd_in,
			STREAM<ap_axiu<104,0,0,DEST_WIDTH>  > & dma_cmd_out,
			STREAM<segmenter_cmd > & cmd);