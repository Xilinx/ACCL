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

#include "hls_stream.h"
#include "ap_int.h"
#include<iostream>

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

void hostctrl_in(	
                stream<ap_uint<DATA_WIDTH> > & in,
				stream<ap_uint<32> > & out
);

int main(){
	int nerrors = 0;
	ap_uint<64> addrb;
	addrb(31,0) = 5;
	addrb(63,32) = 7;
	ap_uint<DATA_WIDTH> in_data;
	stream<ap_uint<DATA_WIDTH>> testin;
	stream<ap_uint<32>> testout;
	in_data.range(31,0)  = 0;
	in_data.range(63,32) = 1;
	in_data.range(95,64) = 2;
 	in_data.range(127,96) = 3;
    in_data.range(159,128) = 4;
    in_data.range(191,160) = 5;
    in_data.range(223,192) = 6;
    in_data.range(255,224) = 7;
    in_data.range(287,256) = 8;
    in_data.range(319,288) = 9;
	in_data.range(383,320) = 10;
	in_data.range(447,384) = 11;
	testin.write(in_data);
	testout.write(1);
	hostctrl_in(testin, testout);
	in_data = testin.read();
	nerrors += (in_data.range(31,0)   != 0);
	nerrors += (in_data.range(63,32)  != 1);
	nerrors += (in_data.range(95,64)  != 2);
	nerrors += (in_data.range(127,96) != 3);
	nerrors += (in_data.range(159,128) != 4);
	nerrors += (in_data.range(191,160) != 5);
	nerrors += (in_data.range(223,192) != 6);
	nerrors += (in_data.range(255,224) != 7);
	nerrors += (in_data.range(287,256) != 8);
	nerrors += (in_data.range(319,288) != 9);
	nerrors += (in_data.range(383,320) != 10);
	nerrors += (in_data.range(447,384) != 11);

	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "******************************"<< std::endl;
	std::cout << "Found " << nerrors << " errors"<< std::endl;
	std::cout << "******************************"<< std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	
	return 0;
}
