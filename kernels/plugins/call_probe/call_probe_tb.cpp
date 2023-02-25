#include "call_probe.h"
#include <iostream>

int main() {
	STREAM<command_word>  cmd_cclo("cmd_cclo");
	STREAM<command_word>  ack_cclo("ack_cclo");
	STREAM<command_word>  cmd_client("cmd_client");
	STREAM<command_word>  ack_client("ack_client");
	command_word tmp;
	unsigned count = 10;
	ap_uint<32> mem[16*count];

	for(int i=0; i<count; i++){
		//paint the buffer
		std::cout << "Painting buffer for call " << i << std::endl;
		for(unsigned  j = 0; j < 16; j++) {
			mem[j+16*i] = 0xFFFFFFFF;
		}
		// Feed
		std::cout << "Feeding call " << i << std::endl;
		for(unsigned  j = 0; j < 15; j++) {
			tmp.data = j;
			cmd_client.write(tmp);
		}
		tmp.data = i;
		ack_cclo.write(tmp);
	}

	// Exec
	std::cout << "Running probe" << std::endl;
	call_probe(true, count, mem, cmd_client, ack_client, cmd_cclo, ack_cclo);

	for(int i=0; i<count; i++){
		// flush and check streams
		std::cout << "Checking streams" << std::endl;
		tmp = STREAM_READ(ack_client);
		if(tmp.data != i){
			std::cout << "Unexpected ack value" << std::endl;
			return -1;
		}
	}
	for(int i=0; i<count; i++){
		for(unsigned  j = 0; j < 15; j++) {
			tmp = STREAM_READ(cmd_cclo);
			if(tmp.data != j){
				std::cout << "Unexpected data value" << std::endl;
				return -1;
			}
		}
	}
	for(int i=0; i<count; i++){
		//check memory contents
		for(unsigned  j = 0; j < 15; j++) {
			if(mem[j+16*i] != j){
				std::cout << "Unexpected data value" << std::endl;
				return -1;
			}
		}
		if(mem[15+16*i] == 0xFFFFFFFF){
			std::cout << "Unexpected data value" << std::endl;
			return -1;
		}
	}
}
