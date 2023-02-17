#include "call_probe.h"
#include <iostream>

int main() {
	STREAM<command_word>  cmd_cclo("cmd_cclo");
	STREAM<command_word>  cmd_client("cmd_client");
	STREAM<command_word>  ack_client("ack_client");
	ap_uint<32> mem[16];
	command_word tmp;

	//paint the buffer
	for(unsigned  j = 0; j < 16; j++) {
		mem[j] = 0xFFFFFFFF;
	}

	// Feed
	for(unsigned  j = 0; j < 15; j++) {
		tmp.data = j;
		cmd_client.write(tmp);
	}

	// Exec
	call_probe(cmd_client, ack_client, cmd_cclo, cmd_cclo, mem);

	// flush and check streams
	tmp = STREAM_READ(ack_client);
	if(tmp.data != 0) return -1;
	for(unsigned  j = 1; j < 15; j++) {
		tmp = STREAM_READ(cmd_cclo);
		if(tmp.data != j) return -1;
	}
	//check memory contents
	for(unsigned  j = 0; j < 15; j++) {
		if(mem[j] != j) return -1;
	}
	if(mem[15] == 0xFFFFFFFF) return -1;
}
