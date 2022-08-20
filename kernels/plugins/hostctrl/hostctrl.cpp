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

#include "hostctrl.h"

using namespace hls;
using namespace std;

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
				STREAM<command_word> &cmd,
				STREAM<command_word> &sts
) {
#pragma HLS INTERFACE s_axilite port=scenario
#pragma HLS INTERFACE s_axilite port=len
#pragma HLS INTERFACE s_axilite port=comm
#pragma HLS INTERFACE s_axilite port=root_src_dst
#pragma HLS INTERFACE s_axilite port=function
#pragma HLS INTERFACE s_axilite port=msg_tag
#pragma HLS INTERFACE s_axilite port=datapath_cfg
#pragma HLS INTERFACE s_axilite port=compression_flags
#pragma HLS INTERFACE s_axilite port=stream_flags
#pragma HLS INTERFACE s_axilite port=addra
#pragma HLS INTERFACE s_axilite port=addrb
#pragma HLS INTERFACE s_axilite port=addrc
#pragma HLS INTERFACE axis port=cmd
#pragma HLS INTERFACE axis port=sts
#pragma HLS INTERFACE s_axilite port=return

	accl_hls::ACCLCommand accl(cmd, sts);
	accl.start_call(
		scenario, len, comm, root_src_dst, function,
		msg_tag, datapath_cfg, compression_flags, stream_flags,
		addra, addrb, addrc
	);
	accl.finalize_call();

}
