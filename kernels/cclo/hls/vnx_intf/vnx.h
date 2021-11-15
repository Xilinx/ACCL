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

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

#ifndef DATA_WIDTH
#define DATA_WIDTH 512
#endif

#define HEADER_COUNT_START 0
#define HEADER_COUNT_END   31
#define HEADER_TAG_START   HEADER_COUNT_END+1
#define HEADER_TAG_END	   HEADER_TAG_START+31
#define HEADER_SRC_START   HEADER_TAG_END+1
#define HEADER_SRC_END	   HEADER_SRC_START+31
#define HEADER_SEQ_START   HEADER_SRC_END+1
#define HEADER_SEQ_END	   HEADER_SEQ_START+31

void vnx_packetizer(	hls::stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
			hls::stream<ap_axiu<DATA_WIDTH,0,0,16> > & out,
			hls::stream<ap_uint<32> > & cmd,
			hls::stream<ap_uint<32> > & sts,
			unsigned int max_pktsize);

void vnx_depacketizer(	hls::stream<ap_axiu<DATA_WIDTH,0,0,16> > & in,
			hls::stream<ap_axiu<DATA_WIDTH,0,0,0> > & out,
			hls::stream<ap_uint<32> > & sts);
