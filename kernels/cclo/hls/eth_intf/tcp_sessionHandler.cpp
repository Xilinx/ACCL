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

using namespace std;

void tcp_openConReq(	
	STREAM<ap_axiu<32,0,0,0> > & cmd,
	STREAM<pkt64>& m_axis_tcp_open_connection,
    STREAM<pkt64>& m_axis_tcp_close_connection
){
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS INLINE off
    if (!STREAM_IS_EMPTY(cmd)){
        unsigned int ip = (STREAM_READ(cmd)).data;
        int port = (STREAM_READ(cmd)).data;
        //if IP is zero, then this is a close request, and port is the session ID
        if(ip != 0){	
            pkt64 openConnection_pkt;
            openConnection_pkt.data(31,0) = ip;
            openConnection_pkt.data(47,32) = port;
            STREAM_WRITE(m_axis_tcp_open_connection, openConnection_pkt);
        } else {
            STREAM_WRITE(m_axis_tcp_close_connection, port);
        }
    }
}

void tcp_openConResp(   
    STREAM<ap_axiu<32,0,0,0> > & sts,
    STREAM<pkt128>& s_axis_tcp_open_status
){
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS INLINE off

    enum openConRespStateType {WAIT_STATUS, WR_SESSION, WR_IP, WR_PORT, WR_SUCCESS};
    static openConRespStateType openConRespState = WAIT_STATUS;

    pkt128 open_status_pkt;
    static unsigned int session; 
    static unsigned int success; 
    static unsigned int ip; 
    static unsigned int port; 

    switch(openConRespState)
    {   
    case WAIT_STATUS:
        if (!STREAM_IS_EMPTY(s_axis_tcp_open_status))
        {
            open_status_pkt = STREAM_READ(s_axis_tcp_open_status);
            session = open_status_pkt.data(15,0);
            success = open_status_pkt.data(23,16);
            ip = open_status_pkt.data(55,24);
            port = open_status_pkt.data(71,56);
            openConRespState = WR_SESSION;
        }
    break; 
    case WR_SESSION:
        if (!STREAM_IS_FULL(sts))
        {
            STREAM_WRITE(sts, ((ap_axiu<32,0,0,0>){.data=session, .last=0}));
            openConRespState = WR_IP;
        }
    break;
    case WR_IP:
        if (!STREAM_IS_FULL(sts))
        {
            STREAM_WRITE(sts, ((ap_axiu<32,0,0,0>){.data=ip, .last=0}));
            openConRespState = WR_PORT;
        }
    break;
    case WR_PORT:
        if (!STREAM_IS_FULL(sts))
        {
            STREAM_WRITE(sts, ((ap_axiu<32,0,0,0>){.data=port, .last=0}));
            openConRespState = WR_SUCCESS;
        }
    break;
    case WR_SUCCESS:
        if (!STREAM_IS_FULL(sts))
        {
            STREAM_WRITE(sts, ((ap_axiu<32,0,0,0>){.data=success, .last=1}));
            openConRespState = WAIT_STATUS;
        }
    break;

    }

}

void tcp_openPort(
    STREAM<ap_axiu<32,0,0,0> > & cmd,
    STREAM<ap_axiu<32,0,0,0> > & sts,
    STREAM<pkt16>& m_axis_tcp_listen_port, 
    STREAM<pkt8>& s_axis_tcp_port_status
){
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS INLINE off

    enum listenFsmStateType {OPEN_PORT, WAIT_PORT_STATUS, WR_STS};
    static listenFsmStateType listenState = OPEN_PORT;
    #pragma HLS RESET variable=listenState

    pkt16 listen_port_pkt;
    pkt8 port_status;
    static ap_uint<32> success;

    switch (listenState)
    {
    case OPEN_PORT:
        if (!STREAM_IS_EMPTY(cmd))
        {
            ap_uint<16> port = (STREAM_READ(cmd)).data;
            listen_port_pkt.data(15,0) = port;
            STREAM_WRITE(m_axis_tcp_listen_port, listen_port_pkt);
            listenState = WAIT_PORT_STATUS;
        }
        break;
    case WAIT_PORT_STATUS:
        if (!STREAM_IS_EMPTY(s_axis_tcp_port_status))
        {
            port_status = STREAM_READ(s_axis_tcp_port_status);
            success = port_status.data;
            listenState = WR_STS;
        }   
        break;
    case WR_STS:
        if (!STREAM_IS_FULL(sts))
        {
            STREAM_WRITE(sts, ((ap_axiu<32,0,0,0>){.data=success, .last=1}));
            listenState = OPEN_PORT;
        }
        break;
    }
}

void tcp_sessionHandler(
    STREAM<ap_axiu<32,0,0,0> > & port_cmd,
    STREAM<ap_axiu<32,0,0,0> > & port_sts,
	STREAM<ap_axiu<32,0,0,0> > & con_cmd,
    STREAM<ap_axiu<32,0,0,0> > & con_sts,
    STREAM<pkt16>& m_axis_tcp_listen_port, 
    STREAM<pkt8>& s_axis_tcp_port_status,
	STREAM<pkt64>& m_axis_tcp_open_connection,
    STREAM<pkt16>& m_axis_tcp_close_connection,
    STREAM<pkt128>& s_axis_tcp_open_status
){
#pragma HLS INTERFACE axis register both port=port_sts
#pragma HLS INTERFACE axis register both port=port_cmd
#pragma HLS INTERFACE axis register both port=m_axis_tcp_listen_port
#pragma HLS INTERFACE axis register both port=s_axis_tcp_port_status
#pragma HLS INTERFACE axis register both port=con_cmd
#pragma HLS INTERFACE axis register both port=m_axis_tcp_open_connection
#pragma HLS INTERFACE axis register both port=m_axis_tcp_close_connection
#pragma HLS INTERFACE axis register both port=con_sts
#pragma HLS INTERFACE axis register both port=s_axis_tcp_open_status
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW disable_start_propagation

tcp_openConReq(con_cmd, m_axis_tcp_open_connection, m_axis_tcp_close_connection);
tcp_openConResp(con_sts, s_axis_tcp_open_status);
tcp_openPort(port_cmd, port_sts, m_axis_tcp_listen_port, s_axis_tcp_port_status);

}
