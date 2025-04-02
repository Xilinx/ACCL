/*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
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

#include "cyt.h"

using namespace std;


void cyt_cq_dm_sts_converter(hls::stream<cyt_ack_t> & cq_sts, 
						hls::stream<ap_axiu<32,0,0,0>> & dm0_sts,
						hls::stream<ap_axiu<32,0,0,0>> & dm1_sts,
						hls::stream<ap_uint<1+4+23>>& dm0_meta,
						hls::stream<ap_uint<1+4+23>>& dm1_meta)
{
#pragma HLS INTERFACE axis register  port=cq_sts
#pragma HLS INTERFACE axis register  port=dm0_sts
#pragma HLS INTERFACE axis register  port=dm1_sts
#pragma HLS INTERFACE axis register  port=dm0_meta
#pragma HLS INTERFACE axis register  port=dm1_meta
#pragma HLS aggregate variable=cq_sts compact=bit

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1

	enum fsmStateType {CYT_STS_0, CYT_STS_1, DM_STS_0, DM_STS_1};
    static fsmStateType  fsmState = CYT_STS_0;

	static cyt_ack_t cq_sts_word;

	static ap_axiu<32,0,0,0> dm_sts_word;
	static ap_uint<1+4+23> dm_meta_word;

	switch (fsmState)
    {
		// the first state reads a cq_sts as a workaround to handle the 2-cycle burst of cq_sts signal
		case CYT_STS_0:
            if (!STREAM_IS_EMPTY(cq_sts))
			{
				STREAM_READ(cq_sts);
				fsmState = CYT_STS_1;
			}
        break;
		case CYT_STS_1:
		 	if (!STREAM_IS_EMPTY(cq_sts))
			{
				cq_sts_word = STREAM_READ(cq_sts);
				
				// only process status if it is local memory completion status
				// only send back ack when the cq_sts stems from kernel issued bypass commands with host == 0
				// if dest == 2, this comes from wr_req/rd_req, no need to forward to data mover
				if((cq_sts_word.opcode == CYT_STRM_CARD || cq_sts_word.opcode == CYT_STRM_HOST) && cq_sts_word.host == 0 && (cq_sts_word.dest == 0 || cq_sts_word.dest == 1))
				{
					if (cq_sts_word.dest == 0) {
						fsmState = DM_STS_0;
					} else if (cq_sts_word.dest == 1) {
						fsmState = DM_STS_1;
					}
				}
				else{
					fsmState = CYT_STS_0;
				}
			}
		break;
		case DM_STS_0:
				if(!STREAM_IS_EMPTY(dm0_meta)){

					dm_meta_word = STREAM_READ(dm0_meta);

					dm_sts_word.data.range(3,0) = dm_meta_word(26,23); //tag
					dm_sts_word.data.range(4,4) = 0; // internal error
					dm_sts_word.data.range(5,5) = 0; // decode erro
					dm_sts_word.data.range(6,6) = 0; // slave error
					dm_sts_word.data.range(7,7) = 1; // OK
					dm_sts_word.data.range(30,8) = dm_meta_word(22,0); // bytes received
					dm_sts_word.data.range(31,31) = dm_meta_word(27,27); // EOP 
					dm_sts_word.last = 1;

					STREAM_WRITE(dm0_sts, dm_sts_word);

					fsmState = CYT_STS_0; // todo: add the check of eop flag
				}
		break;
		case DM_STS_1:
				if(!STREAM_IS_EMPTY(dm1_meta)){

					dm_meta_word = STREAM_READ(dm1_meta);

					dm_sts_word.data.range(3,0) = dm_meta_word(26,23); //tag
					dm_sts_word.data.range(4,4) = 0; // internal error
					dm_sts_word.data.range(5,5) = 0; // decode erro
					dm_sts_word.data.range(6,6) = 0; // slave error
					dm_sts_word.data.range(7,7) = 1; // OK
					dm_sts_word.data.range(30,8) = dm_meta_word(22,0); // bytes received
					dm_sts_word.data.range(31,31) = dm_meta_word(27,27); // EOP 
					dm_sts_word.last = 1;

					STREAM_WRITE(dm1_sts, dm_sts_word);

					fsmState = CYT_STS_0; // todo: add the check of eop flag
				}
		break;
		
	}
}

