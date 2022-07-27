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
#include "accl_hls.h"
#include "ccl_offload_control.h"

using namespace hls;
using namespace std;

//iterates over rx buffers until match is found or a timeout expires
//matches count, src and tag if tag is not ANY
//returns the index of the spare_buffer or -1 if not found
int seek_rx_buffer(
    unsigned int src_rank,
    unsigned int count,
    unsigned int src_tag,
	ap_int<32> *exchmem
){
    unsigned int seq_num; //src_port TODO: use this variable to choose depending on session id or port
    int i;

    //parse rx buffers until match is found
    //matches count, src
    //matches tag or tag is ANY
    //return buffer index 
    unsigned int nbufs = *(exchmem+RX_BUFFER_COUNT_OFFSET);
    rx_buffer *rx_buf_list = (rx_buffer*)(exchmem+RX_BUFFER_COUNT_OFFSET/4+1);
    for(i=0; i<nbufs; i++){	
        if(rx_buf_list[i].status == STATUS_RESERVED)
        {
            if((rx_buf_list[i].rx_src == src_rank) && (rx_buf_list[i].rx_len == count))
            {
                if(((rx_buf_list[i].rx_tag == src_tag) || (src_tag == TAG_ANY)) && (rx_buf_list[i].sequence_number == seq_num) )
                {
                    return i;
                }
            }
        }
    }
    return -1;
}

int wait_on_rx(
    unsigned int src_rank,
    unsigned int count,
    unsigned int src_tag,
	unsigned int timeout,
	ap_int<32> *exchmem
){
    int idx, i;
    for(i = 0; timeout == 0 || i < timeout; i++){
        idx = seek_rx_buffer(src_rank, count, src_tag, exchmem);
        if(idx >= 0) return idx;
    }
    return -1;
}

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
#pragma HLS INTERFACE m_axi port=exchmem offset=slave num_read_outstanding=4 num_write_outstanding=4 bundle=mem
#pragma HLS INTERFACE axis port=cmd
#pragma HLS INTERFACE axis port=sts
#pragma HLS INTERFACE s_axilite port=return

	accl_hls::ACCLCommand accl(cmd, sts);
	// intercept Recv calls with function == 1
	// we'll only forward these receives once the target buffer
	// has been placed into a RX buffer
	// this way we avoid blocking the CCLO
	if(scenario == ACCL_RECV && function == 1){
		//wait until the buffer has been received, then forward the command
		//TODO: pass timeout down from host
		if(wait_on_rx(root_src_dst, len, msg_tag, 0, exchmem) > 0){
			accl.start_call(
				scenario, len, comm, root_src_dst, function,
				msg_tag, datapath_cfg, compression_flags, stream_flags,
				addra, addrb, addrc
			);
		}
	} else {
		//forward the command to the CCLO
		accl.start_call(
			scenario, len, comm, root_src_dst, function,
			msg_tag, datapath_cfg, compression_flags, stream_flags,
			addra, addrb, addrc
		);
	}
	accl.finalize_call();

}
