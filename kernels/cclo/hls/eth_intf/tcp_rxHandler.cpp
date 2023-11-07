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

#ifdef RATE_CONTROL
void inflightReadHandler(
    STREAM<bool>& inflightReadCntReq,
    STREAM<ap_uint<32> >& inflightReadCntRsp,
    STREAM<bool>& incrCntReq,
    STREAM<bool>& decrCntReq
){
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1

    static ap_uint<32> inflightCnt = 0;
    #pragma HLS DEPENDENCE variable=inflightCnt inter false

    if (!STREAM_IS_EMPTY(decrCntReq) & (inflightCnt > 0))
    {
        STREAM_READ(decrCntReq);
        inflightCnt = inflightCnt - 1;
    }
    else if (!STREAM_IS_EMPTY(incrCntReq))
    {
        STREAM_READ(incrCntReq);
        inflightCnt = inflightCnt + 1;
    }
    else if (!STREAM_IS_EMPTY(inflightReadCntReq))
    {
        STREAM_READ(inflightReadCntReq);
        STREAM_WRITE(inflightReadCntRsp, inflightCnt);
    }
}

void requestFSM(
    STREAM<pkt128>& s_axis_tcp_notification, 
    STREAM<pkt32>& m_axis_tcp_read_pkg,
    STREAM<bool>& inflightReadCntReq,
    STREAM<ap_uint<32> >& inflightReadCntRsp,
    STREAM<bool>& incrCntReq,
    STREAM<eth_notification>& m_notif_out
){
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    enum requestFsmStateType {WAIT_NOTI, CHECK_IN_FLIGHT, BACK_OFF, REQUEST};
    static requestFsmStateType  requestFsmState = WAIT_NOTI;

    static ap_uint<8> counter = 0;

    switch(requestFsmState)
    {
    case WAIT_NOTI:
        if (!STREAM_IS_EMPTY(s_axis_tcp_notification))
        {
            bool readReq = true;
            STREAM_WRITE(inflightReadCntReq, readReq);
            requestFsmState = CHECK_IN_FLIGHT;
        }
    break;
    case CHECK_IN_FLIGHT:
        if (!STREAM_IS_EMPTY(inflightReadCntRsp))
        {
            ap_uint<32> inflightCnt = STREAM_READ(inflightReadCntRsp);
            if (inflightCnt < 4)
            {
                requestFsmState = REQUEST;
            }
            else
            {
                requestFsmState = BACK_OFF;
                // cout<<"Enter BACK_OFF"<<endl;
            }
        }
    break;
    case BACK_OFF:
        if (counter == 5)
        {
            counter = 0;
            requestFsmState = WAIT_NOTI;
        }
        else 
        {
            counter = counter + 1;
        }
    break;
    case REQUEST:
        pkt128 tcp_notification_pkt = STREAM_READ(s_axis_tcp_notification);
        ap_uint<16> sessionID = tcp_notification_pkt.data(15,0);
        ap_uint<16> length = tcp_notification_pkt.data(31,16);
        ap_uint<32> ipAddress = tcp_notification_pkt.data(63,32);
        ap_uint<16> dstPort = tcp_notification_pkt.data(79,64);
        ap_uint<1> closed = tcp_notification_pkt.data(80,80);
        STREAM_WRITE(m_notif_out, ((eth_notification){.session_id=sessionID, .length=length}));
#ifndef ACCL_SYNTHESIS
        std::stringstream ss;
        ss << "TCP RX Handler: Requesting data length=" << length << "\n";
        logger << log_level::verbose << ss.str();
#endif

        if (length!=0)
        {
            pkt32 readRequest_pkt;
            readRequest_pkt.data(15,0) = sessionID;
            readRequest_pkt.data(31,16) = length;
            STREAM_WRITE(m_axis_tcp_read_pkg, readRequest_pkt);
            bool incrReq = true;
            STREAM_WRITE(incrCntReq, incrReq);
        }
        requestFsmState = WAIT_NOTI;

    break;
    }
}

void consumeFSM(STREAM<pkt16>& s_axis_tcp_rx_meta, 
               STREAM<stream_word>& s_axis_tcp_rx_data,
               STREAM<stream_word>& m_data_out,
               STREAM<bool>& decrCntReq
               )
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    stream_word currWord;
    enum consumeFsmStateType {WAIT_PKG, CONSUME};
    static consumeFsmStateType  serverFsmState = WAIT_PKG;

    switch (serverFsmState)
    {
    case WAIT_PKG:
        if (!STREAM_IS_EMPTY(s_axis_tcp_rx_meta) && !STREAM_IS_EMPTY(s_axis_tcp_rx_data))
        {
            STREAM_READ(s_axis_tcp_rx_meta);
            stream_word receiveWord = STREAM_READ(s_axis_tcp_rx_data);
            currWord.data = receiveWord.data;
            currWord.keep = receiveWord.keep;
            currWord.last = receiveWord.last;
            STREAM_WRITE(m_data_out, currWord);
            if (!receiveWord.last)
            {
                serverFsmState = CONSUME;
            }
            else if (receiveWord.last)
            {
                bool decrReq = true;
                STREAM_WRITE(decrCntReq, decrReq);
            }
        }
        break;
    case CONSUME:
        if (!STREAM_IS_EMPTY(s_axis_tcp_rx_data))
        {
            stream_word receiveWord = STREAM_READ(s_axis_tcp_rx_data);
            currWord.data = receiveWord.data;
            currWord.keep = receiveWord.keep;
            currWord.last = receiveWord.last;
            STREAM_WRITE(m_data_out, currWord);
            if (receiveWord.last)
            {
                bool decrReq = true;
                STREAM_WRITE(decrCntReq, decrReq);
                serverFsmState = WAIT_PKG;
            }
        }
        break;
    }
}

void tcp_rxHandler(   
    STREAM<pkt128>& s_axis_tcp_notification, 
    STREAM<pkt32>& m_axis_tcp_read_pkg,
    STREAM<pkt16>& s_axis_tcp_rx_meta, 
    STREAM<stream_word>& s_axis_tcp_rx_data,
    STREAM<stream_word>& m_data_out,
    STREAM<eth_notification>& m_notif_out
){

#pragma HLS INTERFACE axis register both port=s_axis_tcp_notification
#pragma HLS INTERFACE axis register both port=m_axis_tcp_read_pkg
#pragma HLS INTERFACE axis register both port=s_axis_tcp_rx_meta
#pragma HLS INTERFACE axis register both port=s_axis_tcp_rx_data
#pragma HLS INTERFACE axis register both port=m_data_out
#pragma HLS INTERFACE axis register both port=m_notif_out
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW disable_start_propagation

    static STREAM<bool> inflightReadCntReq;
    static STREAM<ap_uint<32> > inflightReadCntRsp;
    static STREAM<bool> incrCntReq;
    static STREAM<bool> decrCntReq;
    #pragma HLS stream variable=inflightReadCntReq depth=2
    #pragma HLS stream variable=inflightReadCntRsp depth=2
    #pragma HLS stream variable=incrCntReq depth=2
    #pragma HLS stream variable=decrCntReq depth=2
    
    requestFSM(
        s_axis_tcp_notification, 
        m_axis_tcp_read_pkg,
        inflightReadCntReq,
        inflightReadCntRsp,
        incrCntReq,
        m_notif_out
    );

    inflightReadHandler(
        inflightReadCntReq,
        inflightReadCntRsp,
        incrCntReq,
        decrCntReq
    );

    consumeFSM(
        s_axis_tcp_rx_meta, 
        s_axis_tcp_rx_data,
        m_data_out,
        decrCntReq
    );

}

#else
void tcp_rxHandler(
    STREAM<pkt128>& s_axis_tcp_notification, 
    STREAM<pkt32>& m_axis_tcp_read_pkg,
    STREAM<pkt16>& s_axis_tcp_rx_meta, 
    STREAM<stream_word>& s_axis_tcp_rx_data,
    STREAM<stream_word>& m_data_out,
    STREAM<eth_notification>& m_notif_out
){

#pragma HLS INTERFACE axis register both port=s_axis_tcp_notification
#pragma HLS INTERFACE axis register both port=m_axis_tcp_read_pkg
#pragma HLS INTERFACE axis register both port=s_axis_tcp_rx_meta
#pragma HLS INTERFACE axis register both port=s_axis_tcp_rx_data
#pragma HLS INTERFACE axis register both port=m_data_out
#pragma HLS INTERFACE axis register both port=m_notif_out
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    stream_word currWord;
    enum consumeFsmStateType {WAIT_PKG, CONSUME};
    static consumeFsmStateType  serverFsmState = WAIT_PKG;

    if (!STREAM_IS_EMPTY(s_axis_tcp_notification))
    {
        pkt128 tcp_notification_pkt = STREAM_READ(s_axis_tcp_notification);
        ap_uint<16> sessionID = tcp_notification_pkt.data(15,0);
        ap_uint<16> length = tcp_notification_pkt.data(31,16);
        ap_uint<32> ipAddress = tcp_notification_pkt.data(63,32);
        ap_uint<16> dstPort = tcp_notification_pkt.data(79,64);
        ap_uint<1> closed = tcp_notification_pkt.data(80,80);
        STREAM_WRITE(m_notif_out, (eth_notification){.session_id=sessionID, .length=length});

        // cout<<"notification session "<<sessionID<<" length "<<length<<endl;

        if (length!=0)
        {
            pkt32 readRequest_pkt;
            readRequest_pkt.data(15,0) = sessionID;
            readRequest_pkt.data(31,16) = length;
            STREAM_WRITE(m_axis_tcp_read_pkg, readRequest_pkt);
        }
    }

    switch (serverFsmState)
    {
    case WAIT_PKG:
        if (!STREAM_IS_EMPTY(s_axis_tcp_rx_meta) && !STREAM_IS_EMPTY(s_axis_tcp_rx_data))
        {
            STREAM_READ(s_axis_tcp_rx_meta);
            stream_word receiveWord = STREAM_READ(s_axis_tcp_rx_data);
            currWord.data = receiveWord.data;
            currWord.keep = receiveWord.keep;
            currWord.last = receiveWord.last;
            STREAM_WRITE(m_data_out, currWord);
            if (!receiveWord.last)
            {
                serverFsmState = CONSUME;
            }
        }
        break;
    case CONSUME:
        if (!STREAM_IS_EMPTY(s_axis_tcp_rx_data))
        {
            stream_word receiveWord = STREAM_READ(s_axis_tcp_rx_data);
            currWord.data = receiveWord.data;
            currWord.keep = receiveWord.keep;
            currWord.last = receiveWord.last;
            STREAM_WRITE(m_data_out, currWord);
            if (receiveWord.last)
            {
                serverFsmState = WAIT_PKG;
            }
        }
        break;
    }
}
#endif