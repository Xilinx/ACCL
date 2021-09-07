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

#define DWIDTH512 512
#define DWIDTH256 256
#define DWIDTH128 128
#define DWIDTH64 64
#define DWIDTH32 32
#define DWIDTH16 16
#define DWIDTH8 8

typedef ap_axiu<DWIDTH512, 0, 0, 0> pkt512;
typedef ap_axiu<DWIDTH256, 0, 0, 0> pkt256;
typedef ap_axiu<DWIDTH128, 0, 0, 0> pkt128;
typedef ap_axiu<DWIDTH64, 0, 0, 0> pkt64;
typedef ap_axiu<DWIDTH32, 0, 0, 0> pkt32;
typedef ap_axiu<DWIDTH16, 0, 0, 0> pkt16;
typedef ap_axiu<DWIDTH8, 0, 0, 0> pkt8;




void tcp_txHandler(
               hls::stream<pkt512 >& s_data_in,
               hls::stream<ap_uint<96> >& cmd_txHandler,
               hls::stream<pkt32>& m_axis_tcp_tx_meta, 
               hls::stream<pkt512>& m_axis_tcp_tx_data, 
               hls::stream<pkt64>& s_axis_tcp_tx_status
                )
{
#pragma HLS INTERFACE axis register both port=s_data_in
#pragma HLS INTERFACE axis register both port=cmd_txHandler
#pragma HLS INTERFACE axis register both port=m_axis_tcp_tx_meta
#pragma HLS INTERFACE axis register both port=m_axis_tcp_tx_data
#pragma HLS INTERFACE axis register both port=s_axis_tcp_tx_status
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

	enum txHandlerStateType {WAIT_CMD, WAIT_FIRST_DATA, CHECK_REQ, WRITE_PKG};
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
			if (!cmd_txHandler.empty())
			{
				ap_uint<96> cmd = cmd_txHandler.read();
				sessionID = cmd(31,0);
				expectedTxByteCnt = cmd(63,32);
				maxPkgWord = cmd(95,64);

				txHandlerState = WAIT_FIRST_DATA;
			}
		break;
          case WAIT_FIRST_DATA:
               if(!s_data_in.empty())
               {
                    tx_meta_pkt.data(15,0) = sessionID;

				if (maxPkgWord*(512/8) > expectedTxByteCnt)
					tx_meta_pkt.data(31,16) = expectedTxByteCnt;
				else
					tx_meta_pkt.data(31,16) = maxPkgWord*(512/8);

				m_axis_tcp_tx_meta.write(tx_meta_pkt);

                    txHandlerState = CHECK_REQ;
               }
          break;
		case CHECK_REQ:
			if (!s_axis_tcp_tx_status.empty())
               {
                    pkt64 txStatus_pkt = s_axis_tcp_tx_status.read();
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
                              
                              m_axis_tcp_tx_meta.write(tx_meta_pkt);
                        }
  						txHandlerState = WRITE_PKG;
                    }
                    //if error, resend the tx meta of current packet 
                    else
                    {
                         //Check if connection  was torn down
                         if (error == 1)
                         {
                              // std::cout << "Connection was torn down. " << sessionID << std::endl;
                         }
                         else
                         {
                              tx_meta_pkt.data(15,0) = sessionID;
                              tx_meta_pkt.data(31,16) = length;
                              m_axis_tcp_tx_meta.write(tx_meta_pkt);
                         }
                    }
               }
		break;
		case WRITE_PKG:
			wordCnt ++;
			ap_axiu<DWIDTH512, 0, 0, 0> currWord = s_data_in.read();
			ap_axiu<DWIDTH512, 0, 0, 0> currPkt;
               currPkt.data = currWord.data;
               currPkt.keep = currWord.keep;
               currPkt.last = (wordCnt == currentPkgWord);
               m_axis_tcp_tx_data.write(currPkt);
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