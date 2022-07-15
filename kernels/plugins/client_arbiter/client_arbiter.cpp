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

#include "client_arbiter.h"

void client_arbiter(
	hls::stream<ap_uint<32> > cmd_clients[NCLIENTS],
	hls::stream<ap_uint<32> > ack_clients[NCLIENTS],
	hls::stream<ap_uint<32> > & cmd_cclo,
	hls::stream<ap_uint<32> > & ack_cclo
){
#pragma HLS INTERFACE axis register both port=cmd_clients
#pragma HLS INTERFACE axis register both port=ack_clients
#pragma HLS INTERFACE axis register both port=cmd_cclo
#pragma HLS INTERFACE axis register both port=ack_cclo
#pragma HLS INTERFACE ap_ctrl_none port=return

for(int i=0; i<NCLIENTS; i++){
	if(!cmd_clients[i].empty()){
		//copy a whole command packet from client stream i to the CCLO side
		for(int j=0; j<15; j++){
        	cmd_cclo.write(cmd_clients[i].read());
		}
		//copy a completion flag from the CCLO side to the calling client
		ack_clients[i].write(ack_cclo.read());
    }
}

}