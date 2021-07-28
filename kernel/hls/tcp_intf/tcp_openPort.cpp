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
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

using namespace hls;
using namespace std;

#define DWIDTH512 512
#define DWIDTH256 256
#define DWIDTH128 128
#define DWIDTH64 64
#define DWIDTH32 32
#define DWIDTH16 16
#define DWIDTH8 8

typedef ap_axiu<DWIDTH512, 0, 0, 0> pkt512;
typedef ap_axiu<DWIDTH256, 0, 0, 0> pkt256;
typedef ap_axiu<DWIDTH128, 0, 0, 0> pkt128;
typedef ap_axiu<DWIDTH64, 0, 0, 0> pkt64;
typedef ap_axiu<DWIDTH32, 0, 0, 0> pkt32;
typedef ap_axiu<DWIDTH16, 0, 0, 0> pkt16;
typedef ap_axiu<DWIDTH8, 0, 0, 0> pkt8;

void tcp_openPort(	stream<ap_uint<32> > & cmd,
					stream<ap_uint<32> > & sts,
					stream<pkt16>& m_axis_tcp_listen_port, 
               		stream<pkt8>& s_axis_tcp_port_status)
{
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE axis register both port=cmd
#pragma HLS INTERFACE axis register both port=m_axis_tcp_listen_port
#pragma HLS INTERFACE axis register both port=s_axis_tcp_port_status
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

   	enum listenFsmStateType {OPEN_PORT, WAIT_PORT_STATUS, WR_STS};
   	static listenFsmStateType listenState = OPEN_PORT;
	#pragma HLS RESET variable=listenState

   	pkt16 listen_port_pkt;
   	pkt8 port_status;
   	static ap_uint<32> success;

	switch (listenState)
	{
	case OPEN_PORT:
		if (!cmd.empty())
		{
			ap_uint<16> port = cmd.read();
        	listen_port_pkt.data(15,0) = port;
        	m_axis_tcp_listen_port.write(listen_port_pkt);
			listenState = WAIT_PORT_STATUS;
		}
		break;
	case WAIT_PORT_STATUS:
		if (!s_axis_tcp_port_status.empty())
		{
			s_axis_tcp_port_status.read(port_status);
          	success = port_status.data;
          	listenState = WR_STS;
		}         
		break;
	case WR_STS:
		if (!sts.full())
		{
			sts.write(success);
			listenState = OPEN_PORT;
		}
		break;
	}
}