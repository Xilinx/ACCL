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
#include "eth_intf.h"

using namespace std;


void cyt_rdma_mux_meta(
                hls::stream<rdma_req_t>& s_meta_0,
                hls::stream<req_t>& s_meta_1,
                hls::stream<rdma_req_t>& m_meta_0,
                hls::stream<req_t>& m_meta_1,
                hls::stream<ap_uint<8> >& meta_int
                )
{

#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    static rdma_req_t metaWord_0;
    static req_t metaWord_1;

    static ap_uint<8> dest = 0;

    // if there is a rdma_sq cmd
    if (!STREAM_IS_EMPTY(s_meta_0)){
		metaWord_0 = STREAM_READ(s_meta_0);
        STREAM_WRITE(m_meta_0, metaWord_0);

        // only if the opcode is RDMA WRITE/SEND and host flag equal 0, the data comes from the CCLO 
        if((metaWord_0.opcode == RDMA_WRITE || metaWord_0.opcode == RDMA_SEND) && metaWord_0.host == 0)
        {
            dest = 0;
            STREAM_WRITE(meta_int, dest);
        }
	} else if (!STREAM_IS_EMPTY(s_meta_1)){
        metaWord_1 = STREAM_READ(s_meta_1);
        STREAM_WRITE(m_meta_1, metaWord_1);
        dest = 1;
        STREAM_WRITE(meta_int, dest);
    }
}

void cyt_rdma_mux_data(	
                hls::stream<ap_uint<8> >& meta_int,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_0,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_1,
                hls::stream<ap_axiu<512, 0, 0, 8> >& m_axis
)
{
	#pragma HLS PIPELINE II=1
	#pragma HLS INLINE off

	enum fsmStateType {META, STREAM_0, STREAM_1};
    static fsmStateType  fsmState = META;

    static ap_axiu<512, 0, 0, 8> currWord;

    switch (fsmState)
    {
    case META:
        if (!STREAM_IS_EMPTY(meta_int))
        {
            ap_uint<8> dest = STREAM_READ(meta_int);
            if (dest == 0){
                fsmState = STREAM_0;
            } else {
                fsmState = STREAM_1;
            }
        }
        break;
    case STREAM_0:
        if (!STREAM_IS_EMPTY(s_axis_0))
        {
            currWord = STREAM_READ(s_axis_0);
            STREAM_WRITE(m_axis, currWord);
            if (currWord.last)
            {
                fsmState = META;
            }
        }
        break;
    case STREAM_1:
        if (!STREAM_IS_EMPTY(s_axis_1))
        {
            currWord = STREAM_READ(s_axis_1);
            STREAM_WRITE(m_axis, currWord);
            if (currWord.last)
            {
                fsmState = META;
            }
        }
        break;
    }
} 


// cyt rdma mux will arbitrate the data stream according to the accepted command signal
// the command can be either rdma_sq or the rd_req
// the data stream can be data stream coming from the cclo or from the host/card data stream
// these two streams are mux into single rdma m_axis data stream

void cyt_rdma_mux(
                hls::stream<rdma_req_t >& s_meta_0,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_0,
                hls::stream<req_t >& s_meta_1,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_1,
                hls::stream<rdma_req_t>& m_meta_0,
                hls::stream<req_t>& m_meta_1,
                hls::stream<ap_axiu<512, 0, 0, 8> >& m_axis
                )
{
#pragma HLS INTERFACE axis register  port=s_meta_0
#pragma HLS INTERFACE axis register  port=s_axis_0
#pragma HLS INTERFACE axis register  port=s_meta_1
#pragma HLS INTERFACE axis register  port=s_axis_1
#pragma HLS INTERFACE axis register  port=m_meta_0
#pragma HLS INTERFACE axis register  port=m_meta_1
#pragma HLS INTERFACE axis register  port=m_axis
#pragma HLS aggregate variable=s_meta_0 compact=bit
#pragma HLS aggregate variable=s_meta_1 compact=bit
#pragma HLS aggregate variable=m_meta_0 compact=bit
#pragma HLS aggregate variable=m_meta_1 compact=bit

#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS DATAFLOW disable_start_propagation

    static hls::stream<ap_uint<8> > meta_int;
	#pragma HLS STREAM depth=16 variable=meta_int

    cyt_rdma_mux_meta(
                s_meta_0,
                s_meta_1,
                m_meta_0,
                m_meta_1,
                meta_int
                );

    cyt_rdma_mux_data(	
                meta_int,
                s_axis_0,
                s_axis_1,
                m_axis
    );

}