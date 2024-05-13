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

void tcp_session_handler(
    uint32_t ip,
    uint16_t port_nr,
    volatile uint16_t *session_id,
    volatile uint8_t *return_code,
    ACCL::tcpSessionHandlerOperation operation,
    STREAM<ap_axiu<16, 0, 0, 0>>& listen_port, 
    STREAM<ap_axiu<8, 0, 0, 0>>& port_status,
	STREAM<ap_axiu<64, 0, 0, 0>>& open_connection,
    STREAM<ap_axiu<16, 0, 0, 0>>& close_connection,
    STREAM<ap_axiu<128, 0, 0, 0>>& open_status
){
    #pragma HLS INTERFACE s_axilite port=operation
    #pragma HLS INTERFACE s_axilite port=ip
    #pragma HLS INTERFACE s_axilite port=port_nr

    #pragma HLS INTERFACE s_axilite port=session_id
    #pragma HLS INTERFACE ap_none port=session_id
    #pragma HLS INTERFACE s_axilite port=return_code
    #pragma HLS INTERFACE ap_none port=return_code
    #pragma HLS INTERFACE s_axilite port=return

    #pragma HLS INTERFACE axis register both port=listen_port
    #pragma HLS INTERFACE axis register both port=port_status
    #pragma HLS INTERFACE axis register both port=open_connection
    #pragma HLS INTERFACE axis register both port=close_connection
    #pragma HLS INTERFACE axis register both port=open_status

    if (operation == ACCL::tcpSessionHandlerOperation::OPEN_PORT) {
        ap_axiu<16, 0, 0, 0> listen_port_pkt;
        listen_port_pkt.data(15,0) = port_nr;
        STREAM_WRITE(listen_port, listen_port_pkt);
        while(STREAM_IS_EMPTY(port_status)) {}
        ap_axiu<8, 0, 0, 0> port_status_pkt = STREAM_READ(port_status);
        *return_code = port_status_pkt.data;
    } else if (operation == ACCL::tcpSessionHandlerOperation::OPEN_CONNECTION) {
        ap_axiu<64, 0, 0, 0> openConnection_pkt;
        openConnection_pkt.data(31,0) = ip;
        openConnection_pkt.data(47,32) = port_nr;
        STREAM_WRITE(open_connection, openConnection_pkt);
        while(STREAM_IS_EMPTY(open_status)) {}
        ap_axiu<128, 0, 0, 0> open_status_pkt = STREAM_READ(open_status);
        *session_id = open_status_pkt.data(15,0);
        *return_code = open_status_pkt.data(23, 16); 
    } else if (operation == ACCL::tcpSessionHandlerOperation::CLOSE_CONNECTION) {
        ap_axiu<16, 0, 0, 0> closeConnection_pkt;
        closeConnection_pkt.data = *session_id;
        STREAM_WRITE(close_connection, closeConnection_pkt);
        *return_code = 1;
    } else {
        *return_code = 0;
    }
}
