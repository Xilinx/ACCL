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

// convert the cclo sq (rdma) to cyt sq (rdma)
// currently the cclo sq only contains WRITE/SEND rdma command
// we keep the conversion to cyt_sq_rd output just for consistency of interfaces and future extension
// the m_axis_cyt data stream corresponds to rreq_send and we use the dest to indicate whether host/device
// the s_axis_cyt data stream corresponds to rreq_recv and we simply consume it
void cclo_sq_adapter(
                hls::stream<rdma_req_t >& cclo_sq,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_cclo,
                hls::stream<cyt_req_t>& cyt_sq_wr,
                hls::stream<cyt_req_t>& cyt_sq_rd,
                hls::stream<ap_axiu<512, 0, 0, 8> >& m_axis_cyt,
                hls::stream<ap_axiu<512, 0, 0, 8> >& s_axis_cyt
                )
{
#pragma HLS INTERFACE axis register  port=cclo_sq
#pragma HLS INTERFACE axis register  port=s_axis_cclo
#pragma HLS INTERFACE axis register  port=cyt_sq_wr
#pragma HLS INTERFACE axis register  port=cyt_sq_rd
#pragma HLS INTERFACE axis register  port=m_axis_cyt
#pragma HLS INTERFACE axis register  port=s_axis_cyt
#pragma HLS aggregate variable=cclo_sq compact=bit
#pragma HLS aggregate variable=cyt_sq_wr compact=bit
#pragma HLS aggregate variable=cyt_sq_rd compact=bit

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1

    enum fsmStateType {META, WR_STREAM, RD_STREAM};
    static fsmStateType  fsmState = META;

    static rdma_req_t cclo_req;
    static cyt_req_t cyt_req;
    static ap_axiu<512, 0, 0, 8> currWord;
    static ap_uint<32> pkt_word;
    static ap_uint<32> word_cnt = 0;

    switch (fsmState)
    {
        case META:
            if(!STREAM_IS_EMPTY(cclo_sq)){
                cclo_req = STREAM_READ(cclo_sq);

                cyt_req.rsrvd = 0;
                cyt_req.offs = 0;
                cyt_req.host = 0;
                cyt_req.actv = 0;
                cyt_req.len = cclo_req.len;
                cyt_req.vaddr = cclo_req.vaddr;
                cyt_req.last = 1; // always assert last
                cyt_req.dest = cclo_req.host; // 0-device memory, 1-host memory;
                cyt_req.pid = cclo_req.qpn(CYT_PID_BITS-1,0); //qpn lowest bits are pid 
                cyt_req.vfid = cclo_req.qpn(CYT_DEST_BITS+CYT_PID_BITS-1,CYT_PID_BITS);
                cyt_req.remote = 0;
                cyt_req.rdma = 0;
                cyt_req.mode = 0; // always PARSE
                cyt_req.strm = CYT_STRM_RDMA;
                cyt_req.opcode = cclo_req.opcode; 

                pkt_word = (cyt_req.len + 63) >> 6;

                if(cyt_req.opcode == CYT_RDMA_WRITE || cyt_req.opcode == CYT_RDMA_SEND || cyt_req.opcode == CYT_RDMA_IMMED){
                    STREAM_WRITE(cyt_sq_wr, cyt_req);
                    fsmState = WR_STREAM;
                } else if (cyt_req.opcode == CYT_RDMA_READ) {
                    STREAM_WRITE(cyt_sq_rd, cyt_req);
                    fsmState = RD_STREAM;
                }
            }
        break;
        // move s_axis_cclo to m_axis_cyt and adjust the dest field
        case WR_STREAM:
            if (!STREAM_IS_EMPTY(s_axis_cclo))
            {
                currWord = STREAM_READ(s_axis_cclo);
                ap_axiu<512, 0, 0, 8> outWord;

                outWord.data = currWord.data;
                outWord.keep = currWord.keep;
                outWord.last = currWord.last;
                outWord.dest = cyt_req.dest; // use the dest flag to indicate whether it is to host or device
                word_cnt++;
                
                if (word_cnt == pkt_word)
                {
                    word_cnt = 0;
                    outWord.last = 1;
                    fsmState = META;
                }
                STREAM_WRITE(m_axis_cyt, outWord);
            }
        break;
        // just consume all the data
        case RD_STREAM:
            if(!STREAM_IS_EMPTY(s_axis_cyt)){
                currWord = STREAM_READ(s_axis_cyt);
                word_cnt++; 
                if (word_cnt == pkt_word)
                {
                    word_cnt = 0;
                    fsmState = META;
                }
            }
        break;
    }

}