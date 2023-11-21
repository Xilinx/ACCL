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
#include "eth_intf.h"
#ifndef ACCL_SYNTHESIS
#include "log.hpp"

extern Log logger;
#endif

using namespace std;

void tcp_txHandler(
    STREAM<stream_word>& s_data_in,
    STREAM<ap_uint<96> >& cmd_txHandler,
    STREAM<pkt32>& m_axis_tcp_tx_meta, 
    STREAM<stream_word>& m_axis_tcp_tx_data, 
    STREAM<pkt64>& s_axis_tcp_tx_status
){
#pragma HLS INTERFACE axis register both port=s_data_in
#pragma HLS INTERFACE axis register both port=cmd_txHandler
#pragma HLS INTERFACE axis register both port=m_axis_tcp_tx_meta
#pragma HLS INTERFACE axis register both port=m_axis_tcp_tx_data
#pragma HLS INTERFACE axis register both port=s_axis_tcp_tx_status
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    enum txHandlerStateType {WAIT_CMD, CHECK_REQ, WRITE_PKG};
    static txHandlerStateType txHandlerState = WAIT_CMD;

    static ap_uint<32> sessionID;
    static ap_uint<32> expectedTxByteCnt;
    static ap_uint<32> maxPkgWord;

    static ap_uint<16> length;
    static ap_uint<16> remaining_space;
    static ap_uint<8> error;
    static ap_uint<32> currentPkgWord = 0;
    static ap_uint<32> wordCnt = 0;

    static ap_uint<32> sentByteCnt = 0;
    
    pkt32 tx_meta_pkt;

    switch(txHandlerState)
    {
        case WAIT_CMD:
            if (!STREAM_IS_EMPTY(cmd_txHandler))
            {
                ap_uint<96> cmd = STREAM_READ(cmd_txHandler);
                sessionID = cmd(31,0);
                expectedTxByteCnt = cmd(63,32);
                maxPkgWord = cmd(95,64);

                tx_meta_pkt.data(15,0) = sessionID;

                if (maxPkgWord*(512/8) > expectedTxByteCnt)
                    tx_meta_pkt.data(31,16) = expectedTxByteCnt;
                else
                    tx_meta_pkt.data(31,16) = maxPkgWord*(512/8);

                STREAM_WRITE(m_axis_tcp_tx_meta, tx_meta_pkt);
                txHandlerState = CHECK_REQ;
            }
        break;
        case CHECK_REQ:
            if (!STREAM_IS_EMPTY(s_axis_tcp_tx_status))
               {
                    pkt64 txStatus_pkt = STREAM_READ(s_axis_tcp_tx_status);
                    sessionID = txStatus_pkt.data(15,0);
                    length = txStatus_pkt.data(31,16);
                    remaining_space = txStatus_pkt.data(61,32);
                    error = txStatus_pkt.data(63,62);
                    currentPkgWord = (length + (512/8) -1 ) >> 6; //current packet word length

                    //if no error, perpare the tx meta of the next packet
                    if (error == 0)
                    {
                        sentByteCnt = sentByteCnt + length;

                        if (sentByteCnt < expectedTxByteCnt)
                        {
                            tx_meta_pkt.data(15,0) = sessionID;

                            if (sentByteCnt + maxPkgWord*64 < expectedTxByteCnt )
                            {
                                tx_meta_pkt.data(31,16) = maxPkgWord*(512/8);
                                // currentPkgWord = maxPkgWord;
                            }
                            else
                            {
                                tx_meta_pkt.data(31,16) = expectedTxByteCnt - sentByteCnt;
                                // currentPkgWord = (expectedTxByteCnt - sentByteCnt)>>6;
                            }
                            
                            STREAM_WRITE(m_axis_tcp_tx_meta, tx_meta_pkt);
                        }
                          txHandlerState = WRITE_PKG;
                    }
                    //if error, resend the tx meta of current packet 
                    else
                    {
                        //Check if connection  was torn down
                        if (error == 1)
                        {
                            // logger << log_level::verbose << "Connection was torn down. " << sessionID << std::endl;
                        }
                        else
                        {
                            tx_meta_pkt.data(15,0) = sessionID;
                            tx_meta_pkt.data(31,16) = length;
                            STREAM_WRITE(m_axis_tcp_tx_meta, tx_meta_pkt);
                        }
                    }
               }
        break;
        case WRITE_PKG:
            wordCnt ++;
            stream_word currWord = STREAM_READ(s_data_in);
            stream_word currPkt;
            currPkt.data = currWord.data;
            currPkt.keep = currWord.keep;
            currPkt.dest = currWord.dest;
            currPkt.last = (wordCnt == currentPkgWord);
            STREAM_WRITE(m_axis_tcp_tx_data, currPkt);
            if (wordCnt == currentPkgWord)
            {
                wordCnt = 0;
                if (sentByteCnt >= expectedTxByteCnt)
                {
                    sentByteCnt = 0;
                    currentPkgWord = 0;
                    txHandlerState = WAIT_CMD;
                }
                else
                {
                    txHandlerState = CHECK_REQ;
                }
            }
                         
        break;
    }

}