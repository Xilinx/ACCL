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

void cyt_rdma_arbiter_meta(
                hls::stream<cyt_req_t >& s_meta,
                hls::stream<eth_notification>& m_meta_0,
                hls::stream<cyt_req_t >& m_meta_1,
                hls::stream<ap_uint<64> >& meta_int
                )
{

#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    static cyt_req_t reqWord;
    static eth_notification meta_notif;

    static ap_uint<64> meta_internal = 0;

    if (!STREAM_IS_EMPTY(s_meta)){
		reqWord = STREAM_READ(s_meta);
        // if (reqWord.strm == CYT_STRM_RDMA && reqWord.opcode == CYT_RDMA_SEND){
        if(reqWord.opcode == RC_SEND_FIRST || reqWord.opcode == RC_SEND_MIDDLE || reqWord.opcode == RC_SEND_LAST || reqWord.opcode == RC_SEND_ONLY ){
            meta_notif.type = 0; //don't care
            meta_notif.session_id(CYT_PID_BITS-1,0) = reqWord.pid;
            meta_notif.session_id(CYT_PID_BITS+CYT_DEST_BITS-1,CYT_PID_BITS) = reqWord.vfid; //TODO: check this
            meta_notif.length = reqWord.len;
            STREAM_WRITE(m_meta_0, meta_notif); 

            meta_internal(15,0) = reqWord.dest;
            meta_internal(31,16) = CYT_RDMA_SEND;
            meta_internal(63,32) = reqWord.len;
            STREAM_WRITE(meta_int, meta_internal);

        // } else if (reqWord.strm == CYT_STRM_RDMA && reqWord.opcode == CYT_RDMA_WRITE) {
        } else if (reqWord.opcode == RC_RDMA_WRITE_FIRST || reqWord.opcode == RC_RDMA_WRITE_MIDDLE || reqWord.opcode == RC_RDMA_WRITE_LAST || reqWord.opcode == RC_RDMA_WRITE_LAST_WITH_IMD || reqWord.opcode == RC_RDMA_WRITE_ONLY || reqWord.opcode == RC_RDMA_WRITE_ONLY_WIT_IMD) {
            // simply forward the rq_wr and the conversion from rq_wr to sq_wr is done downstream
            STREAM_WRITE(m_meta_1, reqWord);
            meta_internal(15,0) = reqWord.dest;
            meta_internal(31,16) = CYT_RDMA_WRITE;
            meta_internal(63,32) = reqWord.len;
            STREAM_WRITE(meta_int, meta_internal);
        }
		
	} 
}

// RDMA stream contains last signal only at the last message beats instead of every pkt
// Arbiter needs to arbite pkts according to the meta_internal_len instead of last
// We append the last signal for SEND data stream for each packet in case depacketizer needs the last
// We also append the last signal for WRITE data stream for each packet as the cyt adapter set the ctl bits always to 1
void cyt_rdma_arbiter_data(	
                hls::stream<ap_uint<64> >& meta_int,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_0,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_1,
                hls::stream<ap_axiu<512, 0, 0, 8> >& m_axis_0,
                hls::stream<ap_axiu<512, 0, 0, 8> >& m_axis_1
)
{
	#pragma HLS PIPELINE II=1
	#pragma HLS INLINE off

	enum fsmStateType {META, SEND_STREAM, WRITE_STREAM_CARD, WRITE_STREAM_HOST};
    static fsmStateType  fsmState = META;

    static ap_axiu<512, 0, 0, 8> currWord;

    static ap_uint<64> meta_internal;
    static ap_uint<16> meta_internal_host;
    static ap_uint<16> meta_internal_opcode;
    static ap_uint<32> meta_internal_len;
    static ap_uint<32> pkt_word;
    static ap_uint<32> word_cnt = 0;

    switch (fsmState)
    {
    case META:
        if (!STREAM_IS_EMPTY(meta_int))
        {
            meta_internal = STREAM_READ(meta_int);
            meta_internal_host = meta_internal(15,0);
            meta_internal_opcode = meta_internal(31,16);
            meta_internal_len = meta_internal(63,32);

            pkt_word = (meta_internal_len + 63) >> 6;

            if (meta_internal_opcode == CYT_RDMA_SEND){
                fsmState = SEND_STREAM;
            } else if (meta_internal_opcode == CYT_RDMA_WRITE && meta_internal_host == CYT_STRM_CARD){
                fsmState = WRITE_STREAM_CARD;
            } else if (meta_internal_opcode == CYT_RDMA_WRITE && meta_internal_host == CYT_STRM_HOST){
                fsmState = WRITE_STREAM_HOST;
            }
        }
        break;
    case SEND_STREAM:
        if (!s_axis_0.empty())
        {
            currWord = STREAM_READ(s_axis_0);
            word_cnt++;
            if (word_cnt == pkt_word)
            {
                word_cnt = 0;
                currWord.last = 1;
                fsmState = META;
            }
            STREAM_WRITE(m_axis_0, currWord);
        }
        break;
    case WRITE_STREAM_CARD:
        if (!s_axis_0.empty())
        {
            currWord = STREAM_READ(s_axis_0);
            ap_axiu<512, 0, 0, 8> outWord;

            outWord.data = currWord.data;
            outWord.keep = currWord.keep;
            outWord.last = currWord.last;
            outWord.dest = meta_internal_host; // use the host flag to indicate whether it is to host or device
            word_cnt++;
            
            if (word_cnt == pkt_word)
            {
                word_cnt = 0;
                outWord.last = 1;
                fsmState = META;
            }
            STREAM_WRITE(m_axis_1, outWord);
        }
        break;
    case WRITE_STREAM_HOST:
        if (!s_axis_1.empty())
        {
            currWord = STREAM_READ(s_axis_1);
            ap_axiu<512, 0, 0, 8> outWord;

            outWord.data = currWord.data;
            outWord.keep = currWord.keep;
            outWord.last = currWord.last;
            outWord.dest = meta_internal_host; // use the host flag to indicate whether it is to host or device
            word_cnt++;
            
            if (word_cnt == pkt_word)
            {
                word_cnt = 0;
                outWord.last = 1;
                fsmState = META;
            }
            STREAM_WRITE(m_axis_1, outWord);
        }
        break;
    }
} 

// check the command type of the s_meta, which corresponds to the rq_wr
// the input data stream can come from either s_axis_0/s_axis_1, corresponding to rrsp_recv_0/1. It should be selected based on the dest field of the s_meta
// if the dest is CYT_STRM_CARD, then consume from s_axis_0, and if dest is CYT_STRM_HOST, consume from s_axis_1
// if strm flag is CYT_STRM_RDMA, and the opcode is CYT_RDMA_SEND, this is a SEND Verb, route meta to eth notification and route data stream to channel 0
// if strm flag is CYT_STRM_RDMA, and the opcode is CYT_RDMA_WRITE, this is a WRITE Verb, route meta and data to channel 1
// if data routes to channel 1 (CYT_RDMA_WRITE), set the meta_internal field according to the dest flag in the cyt_req_t to indicate host/card
// compact bit pragma required for cyt_req_t as this interfaces with Coyote. 
void cyt_rdma_arbiter(
                hls::stream<cyt_req_t >& s_meta,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_0,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_1,
                hls::stream<eth_notification >& m_meta_0,
                hls::stream<ap_axiu<512, 0, 0, 8> >& m_axis_0,
                hls::stream<cyt_req_t>& m_meta_1,
                hls::stream<ap_axiu<512, 0, 0, 8> >& m_axis_1
                )
{
#pragma HLS INTERFACE axis register  port=s_meta
#pragma HLS INTERFACE axis register  port=s_axis_1
#pragma HLS INTERFACE axis register  port=s_axis_0
#pragma HLS INTERFACE axis register  port=m_meta_0
#pragma HLS INTERFACE axis register  port=m_axis_0
#pragma HLS INTERFACE axis register  port=m_meta_1
#pragma HLS INTERFACE axis register  port=m_axis_1

#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS aggregate variable=s_meta compact=bit
#pragma HLS aggregate variable=m_meta_1 compact=bit

#pragma HLS DATAFLOW disable_start_propagation

    static hls::stream<ap_uint<64> > meta_int;
	#pragma HLS STREAM depth=8 variable=meta_int

    cyt_rdma_arbiter_meta(
                s_meta,
                m_meta_0,
                m_meta_1,
                meta_int
                );

    cyt_rdma_arbiter_data(	
                meta_int,
                s_axis_0,
                s_axis_1,
                m_axis_0,
                m_axis_1
    );

}