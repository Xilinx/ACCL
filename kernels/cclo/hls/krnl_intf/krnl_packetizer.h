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

void krnl_packetizer(	hls::stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
			hls::stream<ap_axiu<DATA_WIDTH,0,0,0> > & out,
			hls::stream<ap_uint<32> > & cmd,
			hls::stream<ap_uint<32> > & sts);
