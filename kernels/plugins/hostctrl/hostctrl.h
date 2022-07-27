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

#include "accl_hls.h"

void hostctrl(	ap_uint<32> scenario,
				ap_uint<32> len,
				ap_uint<32> comm,
				ap_uint<32> root_src_dst,
				ap_uint<32> function,
				ap_uint<32> msg_tag,
				ap_uint<32> datapath_cfg,
				ap_uint<32> compression_flags,
				ap_uint<32> stream_flags,
				ap_uint<64> addra,
				ap_uint<64> addrb,
				ap_uint<64> addrc,
				ap_int<32> *exchmem,
				STREAM<ap_axiu<32,0,0,0>> &cmd,
				STREAM<ap_axiu<32,0,0,0>> &sts
);