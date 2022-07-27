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
#include "ap_int.h"

using namespace hls;
using namespace std;

void loopback(STREAM<stream_word> & in, STREAM<stream_word> & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

unsigned const bytes_per_word = DATA_WIDTH/8;
stream_word tmp;

do{
#pragma HLS PIPELINE II=1
	tmp = STREAM_READ(in);
	STREAM_WRITE(out, tmp);
} while(tmp.last == 0);

}
