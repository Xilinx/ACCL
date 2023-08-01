# /*******************************************************************************
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

#include "dummy_cyt_rdma_stack.h"
#include "Axi.h" //for axi::Stream
#include <algorithm>

using namespace std;

//Currently implements only SEND and WRITEs with streaming data
//TODO: memory interfacing ops - READ/WRITE

//TX process reads a descriptor from SQ,
//then read message from rd_data and send to TX
//append appropriate header / tdest
void cyt_rdma_tx(
    STREAM<rdma_req_t> & rdma_sq,
    STREAM<stream_word> & rd_data,
    STREAM<stream_word> & tx
){

    enum txFsmStateType {WAIT_SQ, TX_DATA};
    static txFsmStateType txFsmState = WAIT_SQ;
    static rdma_req_t command;
    stream_word tx_word;
    

    switch(txFsmState){
        case WAIT_SQ:
            if(!STREAM_IS_EMPTY(rdma_sq)){
                command = STREAM_READ(rdma_sq);
                //advance to moving data
                txFsmState = TX_DATA;
            }
        case TX_DATA:
            if(!STREAM_IS_EMPTY(rd_data) && !STREAM_IS_FULL(tx)){
                //transmit in chunks of 4KB of data plus header
                //first send command over to remote in packet header
                tx_word.dest = command.qpn;
                tx_word.last = 0;
                tx_word.data(RDMA_REQ_BITS-1,0) = (ap_uint<RDMA_REQ_BITS>)command;
                STREAM_WRITE(tx, tx_word);
                unsigned int i;
                for(i=0; i<4096 && tx_word.last==0; i+=64){
                    tx_word = STREAM_READ(rd_data);
                    tx_word.last |= (i == 4032);
                    tx_word.dest = command.qpn;
                    STREAM_WRITE(tx, tx_word);
                }
                //update command length
                command.len = (command.len < 64*i) ? (unsigned)0 : (unsigned)(command.len - 64*i);
                //we're done with this command, listen on SQ
                if(command.len == 0){
                    txFsmState = WAIT_SQ;
                }
            }
        break;
    }
}

//TX process reads a packet from RX
//first 64B word in packet is command descriptor
//next is data
void cyt_rdma_rx(
    STREAM<eth_notification> & notif,
    STREAM<stream_word> & wr_data,
    STREAM<stream_word> & rx
){
    stream_word rx_word;

    if(!STREAM_IS_EMPTY(rx)){
        //get header
        rx_word = STREAM_READ(rx);
        rdma_req_t command = (ap_uint<RDMA_REQ_BITS>)rx_word.data;
        eth_notification current_notif;
        current_notif.session_id = command.qpn;
        current_notif.type = 0; //type not used here
        current_notif.length = std::min((unsigned)command.len,(unsigned)4096);
        STREAM_WRITE(notif, current_notif);
        //forward data
        do{
            STREAM_WRITE(wr_data, rx_word);
        } while(rx_word.last == 0);
    }
}

void cyt_rdma(
	STREAM<rdma_req_t> & rdma_sq,
    STREAM<eth_notification> & notif,
    STREAM<stream_word> & rd_data,
    STREAM<stream_word> & wr_data,
    STREAM<stream_word> & rx,
    STREAM<stream_word> & tx
){
#pragma HLS INTERFACE axis register both port=rdma_sq
#pragma HLS INTERFACE axis register both port=notif
#pragma HLS INTERFACE axis register both port=rd_data
#pragma HLS INTERFACE axis register both port=wr_data
#pragma HLS INTERFACE axis register both port=rx
#pragma HLS INTERFACE axis register both port=tx
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS DATAFLOW disable_start_propagation

cyt_rdma_tx(rdma_sq, rd_data, tx);
cyt_rdma_rx(notif, wr_data, rx);

}
