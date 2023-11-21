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
#include "tcp_session_handler.h"

using namespace std;

void tcp_session_handler(
    unsigned int ip,
    unsigned int port_nr,
    bool close,	
    unsigned int *session_id,
    bool *success,
    STREAM<ap_axiu<16, 0, 0, 0>>& listen_port, 
    STREAM<ap_axiu<8, 0, 0, 0>>& port_status,
	STREAM<ap_axiu<64, 0, 0, 0>>& open_connection,
    STREAM<ap_axiu<16, 0, 0, 0>>& close_connection,
    STREAM<ap_axiu<128, 0, 0, 0>>& open_status
){
#pragma HLS INTERFACE s_axilite port=ip
#pragma HLS INTERFACE s_axilite port=port_nr
#pragma HLS INTERFACE s_axilite port=close
#pragma HLS INTERFACE s_axilite port=success
#pragma HLS INTERFACE s_axilite port=session_id
#pragma HLS INTERFACE axis register both port=listen_port
#pragma HLS INTERFACE axis register both port=port_status
#pragma HLS INTERFACE axis register both port=open_connection
#pragma HLS INTERFACE axis register both port=close_connection
#pragma HLS INTERFACE axis register both port=open_status

    //first open port, unless the instruction is to close
    if(!close){
        ap_axiu<16, 0, 0, 0> listen_port_pkt;
        listen_port_pkt.data(15,0) = port_nr;
        STREAM_WRITE(listen_port, listen_port_pkt);
        ap_axiu<8, 0, 0, 0> port_status_pkt = STREAM_READ(port_status);
        *success = port_status_pkt.data;
        //if we weren't successful setting up port, stop here
        if(!port_status_pkt.data) return;
    }
    //then open or close connection
    if(!close){
        ap_axiu<64, 0, 0, 0> openConnection_pkt;
        openConnection_pkt.data(31,0) = ip;
        openConnection_pkt.data(47,32) = port_nr;
        STREAM_WRITE(open_connection, openConnection_pkt);
        ap_axiu<128, 0, 0, 0> open_status_pkt;
        open_status_pkt = STREAM_READ(open_status);
        *session_id = open_status_pkt.data(15,0);
        *success = open_status_pkt.data(23,16);
    } else {
        ap_axiu<16, 0, 0, 0> closeConnection_pkt;
        closeConnection_pkt.data = *session_id;
        STREAM_WRITE(close_connection, closeConnection_pkt);
    }

}
