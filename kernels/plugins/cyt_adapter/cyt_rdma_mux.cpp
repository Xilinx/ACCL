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
                hls::stream<cyt_req_t>& s_meta_1,
                hls::stream<cyt_rdma_req_t>& m_meta_0,
                hls::stream<cyt_req_t>& m_meta_1,
                hls::stream<ap_uint<8> >& meta_int
                )
{

#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    static rdma_req_t s_metaWord_0;
    static cyt_rdma_req_t m_metaWord_0;
    static cyt_req_t s_metaWord_1;
    static cyt_rdma_req_msg_t rdma_req_msg;
    static ap_uint<8> dest = 0;

    // if there is a rdma_sq cmd
    // sq command comes from CCLO only has WRITE and SEND Verb
    if (!STREAM_IS_EMPTY(s_meta_0)){
		s_metaWord_0 = STREAM_READ(s_meta_0);
        m_metaWord_0.opcode = s_metaWord_0.opcode;
        m_metaWord_0.qpn = s_metaWord_0.qpn;
        m_metaWord_0.host = 0; // data always managed by CCLO
        m_metaWord_0.mode = 0; // always PARSE
        m_metaWord_0.last = 1; // always assert last
        m_metaWord_0.cmplt = 0; // no need to ack
        m_metaWord_0.ssn = 0;
        m_metaWord_0.offs = 0;
        m_metaWord_0.rsrvd = 0;

        rdma_req_msg.lvaddr = 0; // we don't care about local vaddr
        rdma_req_msg.rvaddr(47,0) = s_metaWord_0.vaddr;
        rdma_req_msg.rvaddr(52,52) = s_metaWord_0.host;
        rdma_req_msg.len = s_metaWord_0.len;
        rdma_req_msg.params = 0;

        m_metaWord_0.msg = (ap_uint<CYT_RDMA_MSG_BITS>)rdma_req_msg;
        
        STREAM_WRITE(m_meta_0, m_metaWord_0);
        dest = 0;
        STREAM_WRITE(meta_int, dest);
	} 
    else if (!STREAM_IS_EMPTY(s_meta_1)){
        s_metaWord_1 = STREAM_READ(s_meta_1);
        STREAM_WRITE(m_meta_1, s_metaWord_1);
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
            if (currWord.last) // TODO: check by cnt instead of last
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
            if (currWord.last) // TODO: check by cnt instead of last
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
                hls::stream<cyt_req_t >& s_meta_1,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_1,
                hls::stream<cyt_rdma_req_t>& m_meta_0,
                hls::stream<cyt_req_t>& m_meta_1,
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
	#pragma HLS STREAM depth=4 variable=meta_int

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