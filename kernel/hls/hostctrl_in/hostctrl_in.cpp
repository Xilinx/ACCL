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

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_utils.h"

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

void hostctrl_in_io(	
				ap_uint<32> scenario,
				ap_uint<32> len,
				ap_uint<32> comm,
				ap_uint<32> root_src_dst,
				ap_uint<32> function,
				ap_uint<32> msg_tag,
				ap_uint<32> buf0_type,
				ap_uint<32> buf1_type,
				ap_uint<32> buf2_type,
				ap_uint<64> addra,
				ap_uint<64> addrb,
				ap_uint<64> addrc,
				stream<ap_uint<32>> &cmd,
				stream<ap_uint<32>> &sts
){
#pragma HLS INTERFACE axis port=cmd
#pragma HLS INTERFACE axis port=sts

io_section:{
	#pragma HLS protocol fixed
	cmd.write(scenario);
	ap_wait();
	cmd.write(len);
	ap_wait();
	cmd.write(comm);
	ap_wait();
	cmd.write(root_src_dst);
	ap_wait();
	cmd.write(function);
	ap_wait();
	cmd.write(msg_tag);
	ap_wait();
	cmd.write(buf0_type);
	ap_wait();
	cmd.write(buf1_type);
	ap_wait();
	cmd.write(buf2_type);
	ap_wait();
	cmd.write(addra(31,0));
	ap_wait();
	cmd.write(addra(63,32));
	ap_wait();
	cmd.write(addrb(31,0));
	ap_wait();
	cmd.write(addrb(63,32));
	ap_wait();
	cmd.write(addrc(31,0));
	ap_wait();
	cmd.write(addrc(63,32));
	ap_wait();
	sts.read();
}

}

void hostctrl_in(	
                stream<ap_uint<DATA_WIDTH> > & in,
				stream<ap_uint<32> > & out
) {

#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out

    ap_uint<DATA_WIDTH> in_data = in.read();
    //Input stream needs to be optimized in the same way as hostctrl 
    ap_uint<32> scenario = in_data.range(31,0);
    ap_uint<32> len = in_data.range(63,32);
    ap_uint<32> comm = in_data.range(95,64);
    ap_uint<32> root_src_dst = in_data.range(127,96);
    ap_uint<32> function = in_data.range(159,128);
    ap_uint<32> msg_tag = in_data.range(191,160);
    ap_uint<32> buf0_type = in_data.range(223,192);
    ap_uint<32> buf1_type = in_data.range(255,224);
    ap_uint<32> buf2_type = in_data.range(287,256);
    ap_uint<64> addra = in_data.range(319,288);
    ap_uint<64> addrb = in_data.range(383,320);
    ap_uint<64> addrc = in_data.range(447,384);
    stream<ap_uint<32>> cmd;

	hostctrl_in_io(scenario, len, comm, root_src_dst, function, msg_tag, buf0_type, buf1_type, buf2_type, addra, addrb, addrc, cmd, out);
    
    //debug
	#ifndef __SYNTHESIS__
	in.write(in_data);
	#endif
}
