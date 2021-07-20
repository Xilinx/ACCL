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

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

void vnx_loopback(	stream<ap_axiu<DATA_WIDTH,0,0,16> > & in,
			stream<ap_axiu<DATA_WIDTH,0,0,16> > & out) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

unsigned const bytes_per_word = DATA_WIDTH/8;

//copy count from header into sts stream
ap_axiu<DATA_WIDTH,0,0,16> tmp;
tmp = in.read();
int count = (tmp.data)(31,0);
out.write(tmp);

while(count > 0){
#pragma HLS PIPELINE II=1
	out.write(in.read());
	count -= bytes_per_word;
}

}
