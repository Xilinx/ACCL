# /*******************************************************************************
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

#include "dummy_tcp_stack.h"
#include "Axi.h" //for axi::Stream

using namespace std;

#define DATA_WIDTH 512

#define DWIDTH512 512
#define DWIDTH128 128
#define DWIDTH64 64
#define DWIDTH32 32
#define DWIDTH16 16
#define DWIDTH8 8

typedef ap_axiu<DWIDTH512, 0, 0, 0> pkt512;
typedef ap_axiu<DWIDTH128, 0, 0, 0> pkt128;
typedef ap_axiu<DWIDTH64, 0, 0, 0> pkt64;
typedef ap_axiu<DWIDTH32, 0, 0, 0> pkt32;
typedef ap_axiu<DWIDTH16, 0, 0, 0> pkt16;
typedef ap_axiu<DWIDTH8, 0, 0, 0> pkt8;

void tx_handler (
    STREAM<pkt32>& s_axis_tcp_tx_meta, 
    STREAM<stream_word>& s_axis_tcp_tx_data, 
    STREAM<pkt64>& m_axis_tcp_tx_status,
    STREAM<stream_word>& out
){
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS INLINE off

    enum txFsmStateType {WAIT_META, WRITE_STATUS, TX_DATA};
    static txFsmStateType txFsmState = WAIT_META;

    static ap_uint<16> session = 0;
    static ap_uint<16> length = 0;
    static ap_uint<16> recvByte = 0;

    switch(txFsmState)
    {
    case WAIT_META:
        if (!STREAM_IS_EMPTY(s_axis_tcp_tx_meta))
        {
            pkt32 tx_meta_pkt = STREAM_READ(s_axis_tcp_tx_meta);
            session = tx_meta_pkt.data(15,0);
            length = tx_meta_pkt.data(31,16);
            txFsmState = WRITE_STATUS;
        }
    break;
    case WRITE_STATUS:
        {
            pkt64 tx_status_pkt;
            tx_status_pkt.data(15,0) = session;
            tx_status_pkt.data(31,16) = length;
            tx_status_pkt.data(61,32) = 8000;
            tx_status_pkt.data(63,62) = 0;
            STREAM_WRITE(m_axis_tcp_tx_status, tx_status_pkt);
            txFsmState = TX_DATA;
        }
    break;
    case TX_DATA:
        if (!STREAM_IS_EMPTY(s_axis_tcp_tx_data))
        {
            STREAM_WRITE(out, STREAM_READ(s_axis_tcp_tx_data));
            recvByte = recvByte + 64;
            if (recvByte >= length)
            {
                recvByte = 0;
                txFsmState = WAIT_META;
            }
        }
    break;

    }//switch

}

void rx_handler(
    STREAM<pkt128>& m_axis_tcp_notification, 
    STREAM<pkt32>& s_axis_tcp_read_pkg,
    STREAM<pkt16>& m_axis_tcp_rx_meta, 
    STREAM<stream_word>& m_axis_tcp_rx_data,
    STREAM<stream_word>& in
){
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS INLINE off

#ifdef ACCL_SYNTHESIS
static hls::stream<hlslib::axi::Stream<ap_uint<DATA_WIDTH>, DEST_WIDTH> > rxDataBuffer;
#pragma HLS STREAM variable=rxDataBuffer depth=512
#else
static hlslib::Stream<hlslib::axi::Stream<ap_uint<DATA_WIDTH>, DEST_WIDTH>, 512> rxDataBuffer;
#endif

    enum rxFsmStateType {BUFFER_DATA, SEND_NOTI, WAIT_REQ, SEND_MSG};
    static rxFsmStateType rxFsmState = BUFFER_DATA;

    static int msg_length = 0;
    static int msg_session = 0;
    stream_word currWord;

    switch (rxFsmState)
    {
    case BUFFER_DATA:
        if (!STREAM_IS_EMPTY(in)){
            do {
                #pragma HLS PIPELINE II=1
                currWord = STREAM_READ(in);
                STREAM_WRITE(rxDataBuffer, (hlslib::axi::Stream<ap_uint<DATA_WIDTH>, DEST_WIDTH>(currWord)));
                for(int i=0; i<DATA_WIDTH/8; i++){
                    msg_length += currWord.keep(i,i);
                }
            } while(currWord.last == 0);
            msg_session = currWord.dest;
            rxFsmState = SEND_NOTI;
        }
    break;
    case SEND_NOTI:
        if (!STREAM_IS_FULL(m_axis_tcp_notification))
        {
            //we acutally don't care about session, ip, port since the 
          //rank is encoded in the header
          pkt128 tcp_notification_pkt;
          tcp_notification_pkt.data(15,0) = msg_session; //session
          tcp_notification_pkt.data(31,16) = msg_length; //length of the data plus header
          tcp_notification_pkt.data(63,32) = 0; //ip
          tcp_notification_pkt.data(79,64) = 0; //port
          tcp_notification_pkt.data(80,80) = 0; //close
          STREAM_WRITE(m_axis_tcp_notification, tcp_notification_pkt);
          rxFsmState = WAIT_REQ;
        }
    break;
    case WAIT_REQ:
        if (!STREAM_IS_EMPTY(s_axis_tcp_read_pkg))
        {
            STREAM_READ(s_axis_tcp_read_pkg);
            pkt16 rx_meta_pkt;
            rx_meta_pkt.data = 0; //we don't care about the session id in this dummy
            STREAM_WRITE(m_axis_tcp_rx_meta, rx_meta_pkt);
            rxFsmState = SEND_MSG;
        }
    break;
    case SEND_MSG:
        if (!STREAM_IS_EMPTY(rxDataBuffer))
        {
            do{
                #pragma HLS PIPELINE II=1
                currWord = STREAM_READ(rxDataBuffer);
                STREAM_WRITE(m_axis_tcp_rx_data, currWord);
            }while(currWord.last == 0);
            msg_length = 0;
            msg_session = 0;
            rxFsmState = BUFFER_DATA;
        }
        
    break;

    }//switch
}

void network_krnl(
    STREAM<pkt128>& m_axis_tcp_notification, 
    STREAM<pkt32>& s_axis_tcp_read_pkg,
    STREAM<pkt16>& m_axis_tcp_rx_meta, 
    STREAM<stream_word>& m_axis_tcp_rx_data,
    STREAM<pkt32>& s_axis_tcp_tx_meta, 
    STREAM<stream_word>& s_axis_tcp_tx_data, 
    STREAM<pkt64>& m_axis_tcp_tx_status,
    STREAM<stream_word>& net_rx,
    STREAM<stream_word>& net_tx
){
#pragma HLS INTERFACE axis register both port=m_axis_tcp_notification
#pragma HLS INTERFACE axis register both port=s_axis_tcp_read_pkg
#pragma HLS INTERFACE axis register both port=m_axis_tcp_rx_meta
#pragma HLS INTERFACE axis register both port=m_axis_tcp_rx_data
#pragma HLS INTERFACE axis register both port=s_axis_tcp_tx_meta
#pragma HLS INTERFACE axis register both port=s_axis_tcp_tx_data
#pragma HLS INTERFACE axis register both port=m_axis_tcp_tx_status
#pragma HLS INTERFACE axis register both port=net_rx
#pragma HLS INTERFACE axis register both port=net_tx
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS DATAFLOW disable_start_propagation

rx_handler(
    m_axis_tcp_notification, 
    s_axis_tcp_read_pkg,
    m_axis_tcp_rx_meta, 
    m_axis_tcp_rx_data,
    net_rx
);
tx_handler(
    s_axis_tcp_tx_meta, 
    s_axis_tcp_tx_data, 
    m_axis_tcp_tx_status,
    net_tx
);

}
