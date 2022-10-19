#define NUM_CTRL_STREAMS 3

#include "client_arbiter.h"

#include <random>
#include <iostream>
#include <iomanip>


int main() {
	std::default_random_engine  rnd;
	std::uniform_int_distribution<int>  dist(0, 17*NUM_CTRL_STREAMS);

	STREAM<command_word>  cmd_clients[NUM_CTRL_STREAMS];
	STREAM<command_word>  ack_clients[NUM_CTRL_STREAMS];
	STREAM<command_word>  cmd_cclo("cmd_cclo");
	STREAM<command_word>  ack_cclo("ack_cclo");

	constexpr unsigned  N_RQSTS = 8;
	unsigned  cnt_rqsts[NUM_CTRL_STREAMS] = { 0, };
	unsigned  cnt_cclo = 0;
	unsigned  done = 0;
	while(done < NUM_CTRL_STREAMS) {
		// Feed
		for(unsigned  i = 0; i < NUM_CTRL_STREAMS; i++) {
			if(cnt_rqsts[i] < N_RQSTS) {
				if(dist(rnd) == 0) {
					for(unsigned  j = 0; j < 15; j++) {
						command_word  cmd;
						cmd.data = 256*i + j;
						cmd_clients[i].write(cmd);
					}
					if(++cnt_rqsts[i] == N_RQSTS)  done++;
				}
			}
		}

		// Exec
		client_arbiter(cmd_clients, ack_clients, cmd_cclo, ack_cclo);

		// Ack back
		if(!cmd_cclo.empty()) {
			command_word const  x = cmd_cclo.read();
			std::cout << std::hex << "> " << (x.data/256) << '.' << (x.data&0xFF) << std::endl;
			cnt_cclo++;
			if(cnt_cclo == 14) 	ack_cclo.write(x);
			if(cnt_cclo == 15) 	cnt_cclo = 0;
		}

		for(unsigned  i = 0; i < NUM_CTRL_STREAMS; i++) {
			if(!ack_clients[i].empty()) {
				command_word const  x = ack_clients[i].read();
				std::cout << std::hex << "* " << x.data/256 << std::endl;
			}
		}
	}
}
