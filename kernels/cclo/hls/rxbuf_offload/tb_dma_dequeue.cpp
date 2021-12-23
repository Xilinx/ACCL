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

int test_dma_error_and_ok_behaviour(int use_tcp, int num_test_spares=16){
	stream<ap_uint<32>> sts_dma_tcp	, sts_dma_udp;
	stream<ap_uint<32>> header_tcp	, header_udp;
	stream<ap_uint<32>> inflight_queue;

	int initial_btt		= 100,  initial_tag     = 0x42, initial_src = 0, initial_sequence_number = 666;

	void * ptr 					 = malloc(sizeof(rx_buffer)*num_test_spares+sizeof(unsigned int));
	//create a fake memory
	*((unsigned int *)ptr) 	=  num_test_spares;
	rx_buffer * buffers 	= ( (rx_buffer *) (((uint *)ptr) + 1 ) );
	cout << "Exchange memory: " << ptr << endl;
	//fill buffers with dummy data 
	for(int i = 0; i < num_test_spares; i++ ){
		//only even buffer are enqueued
		buffers[i].status = (i % 2) == 0 ? STATUS_ENQUEUED : STATUS_RESERVED;
		buffers[i].addrl  = 0xdeadbeef + i ;
		buffers[i].addrh  = 0;
		cout << "Buffer "<< i << " status:" << buffers[i].status << " addr: " << hex << buffers[i].addrh << hex << buffers[i].addrl << endl;
		if ( (i % 2) == 0 ){
			//put spare index in the inflight queue
			inflight_queue.write( i );
			//create header 
			if(use_tcp){
				header_tcp.write((initial_btt 			 + i) & DMA_MAX_BTT);
				header_tcp.write(initial_tag 			 + i);
				header_tcp.write(initial_src 			 + i);
				header_tcp.write(initial_sequence_number + i);
			}else{
				header_udp.write((initial_btt 			 + i) & DMA_MAX_BTT);
				header_udp.write(initial_tag 			 + i);
				header_udp.write(initial_src 			 + i);
				header_udp.write(initial_sequence_number + i);
			}
			//create dma sts
			if( i % 4 == 0){
				//create an ok status
				if(use_tcp){
					sts_dma_tcp.write(0x80000080 | ((initial_btt + i) & DMA_MAX_BTT) << 8 );
				}else{
					sts_dma_udp.write(0x80000080 | ((initial_btt + i) & DMA_MAX_BTT) << 8 );
				}
			}else{
				//create an error status
				if(use_tcp){
					sts_dma_tcp.write(0x00000070 );
				}else{
					sts_dma_udp.write(0x00000070 );
				}
			}
		}
	}

	for (int i =0; i < ((int)(num_test_spares / 2)); i++){
		dma_dequeue( use_tcp, sts_dma_udp, sts_dma_tcp, header_udp, header_tcp, inflight_queue, ((ap_uint<32>*) ptr)+1);
	}

	for(int i = 0; i < num_test_spares; i++ ){
		if ( (i %2) == 1 ){
			//odds remain reserved
			if(buffers[i].status != STATUS_RESERVED	) {printf("status fail\n"); return 1;};
			if(buffers[i].addrl  != 0xdeadbeef + i  ) {printf("addrl  fail\n"); return 1;};
			if(buffers[i].addrh  != 0			    ) {printf("addrh  fail\n"); return 1;};
		}else if( (i % 4) == 0){
			//0, 4, 8, ... become reserved
			if(buffers[i].status != STATUS_RESERVED	) {printf("status fail\n"); return 1;};
			if(buffers[i].addrl  != 0xdeadbeef + i  ) {printf("addrl  fail\n"); return 1;};
			if(buffers[i].addrh  != 0			    ) {printf("addrh  fail\n"); return 1;};
		}else{
			//2, 6, become erroneous
			if(buffers[i].status != STATUS_ERROR	) {printf("status fail\n"); return 1;};
			if(buffers[i].addrl  != 0xdeadbeef + i  ) {printf("addrl  fail\n"); return 1;};
			if(buffers[i].addrh  != 0			    ) {printf("addrh  fail\n"); return 1;};
		}
		
		cout << "Buffer "<< i << " status:" << buffers[i].status << " addr: " << hex << buffers[i].addrh << hex << buffers[i].addrl << endl;
		
	}

	return 0;
}
int main(){
	int errors = 0;
	errors += test_dma_error_and_ok_behaviour(1, 16);
	errors += test_dma_error_and_ok_behaviour(0, 16);
	cout << "Test passed"<< endl;
	return 0;
}
