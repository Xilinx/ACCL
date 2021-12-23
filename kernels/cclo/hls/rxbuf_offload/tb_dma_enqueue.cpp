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

#include "dma_queue.h"
#include <iostream>
#include <stdio.h>

using namespace hls;
using namespace std;

int test_even(int use_tcp, int num_spare_buffers=16){
	stream< ap_uint<104> > cmd_dma_tcp, cmd_dma_udp;
	stream< ap_uint<32> > inflight_queue;
	uint max_len = 1000;
	//create a fake memory to hold spare buffers
	ap_uint<32>  buffers [num_spare_buffers*SPARE_BUFFER_FIELDS];
	ap_uint<32>* buffers_addr = buffers;
	cout << "Buffer_in  base addr " << hex << buffers  << endl;

 	//fill buffers with dummy data only even buffer are enqueued
	for(int i = 0; i < num_spare_buffers; i++ ){
		buffers[i*SPARE_BUFFER_FIELDS + STATUS_OFFSET] = ((i % 2) == 0 ? STATUS_IDLE : STATUS_RESERVED);
		buffers[i*SPARE_BUFFER_FIELDS + ADDRL_OFFSET ] = 0xdeadbeef + i ;
		buffers[i*SPARE_BUFFER_FIELDS + ADDRH_OFFSET ] = 0;
		buffers[i*SPARE_BUFFER_FIELDS + MAX_LEN_OFFSET] = max_len +1;
		cout << "Buffer "<< i << " status:" << buffers[i*SPARE_BUFFER_FIELDS + STATUS_OFFSET] << " addr: " << hex << buffers[i*SPARE_BUFFER_FIELDS + ADDRH_OFFSET] << hex << buffers[i*SPARE_BUFFER_FIELDS + ADDRL_OFFSET] << endl;
	}

	dma_enqueue(
				use_tcp,
				num_spare_buffers,
				cmd_dma_udp,
				cmd_dma_tcp,
				inflight_queue,
				buffers_addr
	);

	for (ap_uint<32> i = 0; i < num_spare_buffers; i++)
	{
		//only even buffers are enqueued
		cout << "Buffer "<< i << " status:" << hex << buffers[(i*SPARE_BUFFER_FIELDS) + STATUS_OFFSET] << endl;
		if ( (i % 2 ) == 1 ){
			//check that odd spare buffers are left unmodified
			if( buffers[i*SPARE_BUFFER_FIELDS + STATUS_OFFSET] != STATUS_RESERVED ) { printf("fail status"		); return 1};
			if( buffers[i*SPARE_BUFFER_FIELDS + ADDRL_OFFSET ] != 0xdeadbeef + i  ) { printf("fail addrl"		); return 1};
			if( buffers[i*SPARE_BUFFER_FIELDS + ADDRH_OFFSET ] != 0 			  ) { printf("fail addrh"		); return 1};
			if( buffers[i*SPARE_BUFFER_FIELDS + MAX_LEN_OFFSET] != max_len +1     ) { printf("fail maxlen"		); return 1};
			continue;
		}
		ap_uint <104> dma_cmd ;
		//read dma cmd according to the supplied flag
		if (use_tcp){
			dma_cmd = cmd_dma_tcp.read();
		}else{
			dma_cmd = cmd_dma_udp.read();
		}
		//read next spare buffer enqueued
		ap_uint<32> buffer_idx = inflight_queue.read();
		cout << hex << dma_cmd << endl;
		cout << "inflight queue "  << buffer_idx << endl;
		//check that:
		//status   is enqueued
		//address are unchanged
		//cmd 	   is created as it should be
		//enqueued spare buffer id is what is expected (i)
		if( buffers[i*SPARE_BUFFER_FIELDS + STATUS_OFFSET] != STATUS_ENQUEUED 								) { printf("fail status\n"	  ); return 1};
		if( buffers[i*SPARE_BUFFER_FIELDS + ADDRL_OFFSET ] != 0xdeadbeef + i  								) { printf("fail addrl\n"	  ); return 1};
		if( buffers[i*SPARE_BUFFER_FIELDS + ADDRH_OFFSET ] != 0 				 							) { printf("fail addrh\n"	  ); return 1};
		if( buffers[i*SPARE_BUFFER_FIELDS + MAX_LEN_OFFSET]!= max_len +1     	) { printf("fail maxlen"		); return 1};
		if( dma_cmd.range( 31,  0)  != (0xC0800000 | max_len)					) { printf("fail dma_cmd[0]\n"); return 1};
		if( dma_cmd.range( 63, 32)  != buffers[i*SPARE_BUFFER_FIELDS + ADDRL_OFFSET]	) { printf("fail dma_cmd[1]\n"); return 1};
		if( dma_cmd.range( 95, 64)  != buffers[i*SPARE_BUFFER_FIELDS + ADDRH_OFFSET]	) { printf("fail dma_cmd[2]\n"); return 1};
		if( dma_cmd.range(103, 96)  != 0				 						) { printf("fail dma_cmd[3]\n"); return 1};
		if( buffer_idx 									   != i						 						) { printf("fail buffer_idx\n"); return 1};

	}
	return 0;
	
}

int test_all(int use_tcp, int num_spare_buffers=4){
	stream< ap_uint<104> > cmd_dma_tcp, cmd_dma_udp;
	stream< ap_uint<32> > inflight_queue;
	uint max_len = 1000;
	//create a fake memory to hold spare buffers
	ap_uint<32>  buffers [num_spare_buffers*SPARE_BUFFER_FIELDS];
	//ap_uint<32>  buffers1[num_spare_buffers*SPARE_BUFFER_FIELDS];
	ap_uint<32>* buffers_addr = buffers;
	cout << "Buffer_in  base addr " << hex << buffers  << endl;
	//cout << "Buffer_out base addr " << hex << buffers1 << endl;

 	//fill buffers with dummy data
	for(int i = 0; i < num_spare_buffers; i++ ){
		//all buffers are filled
		buffers[i*SPARE_BUFFER_FIELDS + STATUS_OFFSET ] = STATUS_IDLE ;
		buffers[i*SPARE_BUFFER_FIELDS + ADDRL_OFFSET  ] = 0xdeadbeef + i ;
		buffers[i*SPARE_BUFFER_FIELDS + ADDRH_OFFSET  ] = 0;
		buffers[i*SPARE_BUFFER_FIELDS + MAX_LEN_OFFSET] = max_len +1;
		cout << "Buffer "<< i << " status:" << buffers[i*SPARE_BUFFER_FIELDS + STATUS_OFFSET] << " addr: " << hex << buffers[i*SPARE_BUFFER_FIELDS + ADDRH_OFFSET] << hex << buffers[i*SPARE_BUFFER_FIELDS + ADDRL_OFFSET] << endl;
	}

	dma_enqueue(
				use_tcp,
				num_spare_buffers,
				cmd_dma_udp,
				cmd_dma_tcp,
				inflight_queue,
				buffers_addr
	);
	//check results
	for (ap_uint<32> i = 0; i < num_spare_buffers; i++)
	{
		//all buffers are expected to be reserved
		cout << "Buffer "<< i << " status:" << hex << buffers[(i*SPARE_BUFFER_FIELDS) + STATUS_OFFSET] << endl;
		//get dma cmd according to the flag
		ap_uint <104> dma_cmd ;
		//read dma cmd according to the supplied flag
		if (use_tcp){
			dma_cmd = cmd_dma_tcp.read();
		}else{
			dma_cmd = cmd_dma_udp.read();
		}
		//get enqueued spare buffer id
		ap_uint<32> buffer_idx = inflight_queue.read();
		cout << hex << dma_cmd << endl;
		cout << "inflight queue "  << buffer_idx << endl;
		//check that:
		//status   is enqueued
		//address are unchanged
		//cmd 	   is created as it should be
		//enqueued spare buffer id is what is expected (i)
		if( buffers[i*SPARE_BUFFER_FIELDS + STATUS_OFFSET ] != STATUS_ENQUEUED 								) { printf("fail status\n"	  ); return 1};
		if( buffers[i*SPARE_BUFFER_FIELDS + ADDRL_OFFSET  ] != 0xdeadbeef + i  								) { printf("fail addrl\n"	  ); return 1};
		if( buffers[i*SPARE_BUFFER_FIELDS + ADDRH_OFFSET  ] != 0 				 							) { printf("fail addrh\n"	  ); return 1};
		if( buffers[i*SPARE_BUFFER_FIELDS + MAX_LEN_OFFSET]!= max_len +1    ) { printf("fail maxlen"	  ); return 1};
		if( dma_cmd.range( 31,  0)  != (0xC0800000 | max_len)				) { printf("fail dma_cmd[0]\n"); return 1};
		if( dma_cmd.range( 63, 32)  != 0xdeadbeef + i 						) { printf("fail dma_cmd[1]\n"); return 1};
		if( dma_cmd.range( 95, 64)  != 0 									) { printf("fail dma_cmd[2]\n"); return 1};
		if( dma_cmd.range(103, 96)  != 0				 					) { printf("fail dma_cmd[3]\n"); return 1};
		if( buffer_idx 									   != i						 						) { printf("fail buffer_idx\n"); return 1};

	}
	return 0;
}


int main(){
	int 		 nerrors 		 = 0;
	nerrors += test_all( 1, 16);
	//nerrors += test_even(1, 16);
	//nerrors += test_even(0, 16);
	if (nerrors == 0) {
		cout << "TB passed" << endl;
	}

	return nerrors;
}
