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

#include "dummy_cyt_rdma_stack.h"
#include "Axi.h" //for axi::Stream

using namespace std;

//Currently implements SEND and SEND_IMMEDIATE
//TODO: support for WRITE

//TX process reads a descriptor from SQ:
//if SEND_IMMEDIATE, send embedded message to TX
//if SEND, read message from rd_data and send to TX
void cyt_rdma_tx(
    STREAM<rdma_req_t> & rdma_sq,
    STREAM<stream_word> & rd_data,
    STREAM<stream_word> & tx
){

    enum txFsmStateType {WAIT_SQ, TX_DATA};
    static txFsmStateType txFsmState = WAIT_SQ;
    static ap_uint<32> txCount = 0;
    static rdma_req_msg_t msg;
    static rdma_req_t command;
    stream_word tx_word;
    

    switch(txFsmState){
        case WAIT_SQ:
            if(!STREAM_IS_EMPTY(rdma_sq)){
                command = STREAM_READ(rdma_sq);
                ap_uint<RDMA_REQ_BITS> command_uint = command;
                //send command over to remote
                tx_word.dest = command.qpn;
                if(RDMA_REQ_BITS > 512){
                    tx_word.last = 0;
                    tx_word.data = ((ap_uint<RDMA_REQ_BITS>)command)(511,0);
                    STREAM_WRITE(tx, tx_word);
                    tx_word.last = 1;
                    tx_word.data = ((ap_uint<RDMA_REQ_BITS>)command)(RDMA_REQ_BITS-1,512);
                    STREAM_WRITE(tx, tx_word);
                } else {
                    tx_word.last = 1;
                    tx_word.data = (ap_uint<RDMA_REQ_BITS>)command;
                    STREAM_WRITE(tx, tx_word);
                }
                //if SENDing, prepare for moving data
                if(command.opcode == RDMA_SEND){
                    msg = command.msg;
                    txCount = msg.len;
                    txFsmState = TX_DATA;
                }
            }
        case TX_DATA:
            if(!STREAM_IS_EMPTY(rd_data) && !STREAM_IS_FULL(tx)){
                //transmit in chunks of 4KB
                for(int i=0; i<4096; i+=64){
                    txCount += 64;
                    tx_word.last = ((i == 4032) | (txCount >= msg.len));
                    tx_word.dest = command.qpn;
                    tx_word.data = (STREAM_READ(rd_data)).data;
                    STREAM_WRITE(tx, tx_word);
                }
                if(txCount >= msg.len){
                    txFsmState = WAIT_SQ;
                }
            }
        break;
    }
}

//TX process reads a packet from RX
//first 64B word in packet is descriptor:
//if SEND_IMMEDIATE, send descriptor with embedded message to wr_data
//if SEND
//      send descriptor to RQ
//      read message from RX and send to wr_data
void cyt_rdma_rx(
    STREAM<rdma_req_t> & rdma_rq,
    STREAM<stream_word> & wr_data,
    STREAM<stream_word> & rx
){
    static bool cmd_active[(1<<RDMA_QPN_BITS)] = {0};
    static rdma_req_t cmd[(1<<RDMA_QPN_BITS)];
    static ap_uint<RDMA_LEN_BITS> cmd_rem[(1<<RDMA_QPN_BITS)];

    stream_word rx_word;

    if(!STREAM_IS_EMPTY(rx)){
        rx_word = STREAM_READ(rx);
        if(cmd_active[rx_word.dest]){
            ap_int<RDMA_LEN_BITS+1> rem = cmd_rem[rx_word.dest];
            do{
                STREAM_WRITE(wr_data, rx_word);
                rem -= 64;
            } while(rx_word.last == 0);
            if(rem <= 0){
                cmd_active[rx_word.dest] = false;
            } else {
                cmd_rem[rx_word.dest] = rem;
            }
        } else{
            ap_uint<RDMA_REQ_BITS> cmd_uint;
            if(RDMA_REQ_BITS > 512){
                cmd_uint(511,0) = rx_word.data;
                rx_word = STREAM_READ(rx);
                cmd_uint(RDMA_REQ_BITS-1,512) = rx_word.data;
            } else {
                cmd_uint = rx_word.data;
            }
            cmd[rx_word.dest] = cmd_uint;
            if(cmd[rx_word.dest].opcode != RDMA_WRITE){
                STREAM_WRITE(rdma_rq, cmd[rx_word.dest]);
            }
            if(cmd[rx_word.dest].opcode != RDMA_IMMED){
                cmd_active[rx_word.dest] = true;
                cmd_rem[rx_word.dest] = ((rdma_req_msg_t)cmd[rx_word.dest].msg).len;
            }
        }
    }
}

void cyt_rdma(
	STREAM<rdma_req_t> & rdma_sq,
    STREAM<rdma_req_t> & rdma_rq,
    STREAM<stream_word> & rd_data,
    STREAM<stream_word> & wr_data,
    STREAM<stream_word> & rx,
    STREAM<stream_word> & tx
){
#pragma HLS INTERFACE axis register both port=rdma_sq
#pragma HLS INTERFACE axis register both port=rdma_rq
#pragma HLS INTERFACE axis register both port=wr_data
#pragma HLS INTERFACE axis register both port=rd_data
#pragma HLS INTERFACE axis register both port=rx
#pragma HLS INTERFACE axis register both port=tx
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS DATAFLOW disable_start_propagation

cyt_rdma_tx(rdma_sq, rd_data, tx);
cyt_rdma_rx(rdma_rq, wr_data, rx);

}
