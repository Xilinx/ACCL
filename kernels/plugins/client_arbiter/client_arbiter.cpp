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
	STREAM<command_word> cmd_clients[NCLIENTS],
	STREAM<command_word> ack_clients[NCLIENTS],
	STREAM<command_word> & cmd_cclo,
	STREAM<command_word> & ack_cclo
){
#pragma HLS INTERFACE axis register both port=cmd_clients
#pragma HLS INTERFACE axis register both port=ack_clients
#pragma HLS INTERFACE axis register both port=cmd_cclo
#pragma HLS INTERFACE axis register both port=ack_cclo
#pragma HLS INTERFACE ap_ctrl_none port=return

for(int i=0; i<NCLIENTS; i++){
	if(!STREAM_IS_EMPTY(cmd_clients[i])){
		//copy a whole command packet from client stream i to the CCLO side
		for(int j=0; j<15; j++){
        	STREAM_WRITE(cmd_cclo, STREAM_READ(cmd_clients[i]));
		}
		//copy a completion flag from the CCLO side to the calling client
		STREAM_WRITE(ack_clients[i], STREAM_READ(ack_cclo));
    }
}

}