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

void stream_segmenter(STREAM<stream_word > & in,
			STREAM<stream_word > & out,
			STREAM<segmenter_cmd > & cmd,
			STREAM<ap_uint<32> > & sts) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=cmd
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE ap_ctrl_none port=return

unsigned const bytes_per_word = DATA_WIDTH/8;

//read commands from the command stream
segmenter_cmd cmd_tmp = STREAM_READ(cmd);
int nwords = 0;
while(nwords<cmd_tmp.nwords){
#pragma HLS PIPELINE II=1
	stream_word tmp = STREAM_READ(in);
	tmp.dest = cmd_tmp.dest;
	if(cmd_tmp.indeterminate_btt){
		tmp.last = tmp.last | (nwords == (cmd_tmp.nwords-1));
	} else {
		tmp.last = (nwords == (cmd_tmp.nwords-1));
	}
	STREAM_WRITE(out, tmp);
	nwords++;
	if(tmp.last == 1) break;
}
//acknowledge by sending back the number of words
if(cmd_tmp.emit_ack){
	STREAM_WRITE(sts, nwords);
}
}