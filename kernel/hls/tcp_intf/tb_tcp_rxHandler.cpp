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

#define DATA_WIDTH 512

typedef ap_axiu<DWIDTH512, 0, 0, 0> pkt512;
typedef ap_axiu<DWIDTH256, 0, 0, 0> pkt256;
typedef ap_axiu<DWIDTH128, 0, 0, 0> pkt128;
typedef ap_axiu<DWIDTH64, 0, 0, 0> pkt64;
typedef ap_axiu<DWIDTH32, 0, 0, 0> pkt32;
typedef ap_axiu<DWIDTH16, 0, 0, 0> pkt16;
typedef ap_axiu<DWIDTH8, 0, 0, 0> pkt8;

int ntransfers(int nbytes){
	int bytes_per_transfer = 512/8;
	return (nbytes+bytes_per_transfer-1)/bytes_per_transfer;
}

void tcp_rxHandler(   
               hls::stream<pkt128>& s_axis_tcp_notification, 
               hls::stream<pkt32>& m_axis_tcp_read_pkg,
               hls::stream<pkt16>& s_axis_tcp_rx_meta, 
               hls::stream<pkt512>& s_axis_tcp_rx_data,
               hls::stream<pkt512 >& s_data_out
                );

int main()
{
	hls::stream<pkt128> s_axis_tcp_notification;
    hls::stream<pkt32> m_axis_tcp_read_pkg;
    hls::stream<pkt16> s_axis_tcp_rx_meta;
    hls::stream<pkt512> s_axis_tcp_rx_data;
    hls::stream<pkt512> s_data_out;

    stream<ap_axiu<DATA_WIDTH,0,0,0> > golden;

    ap_axiu<DATA_WIDTH,0,0,0> inword;
	ap_axiu<DATA_WIDTH,0,0,0> outword;
	ap_axiu<DATA_WIDTH,0,0,0> goldenword;

    int count;
    ap_uint<16> session;
    ap_uint<16> length;
    ap_uint<32> rxBytes;
    ap_uint<32> byteCnt;
    ap_uint<32> wordCnt;

    count = 0;
    session = 1;
    rxBytes = 16*1024;
    length = 1024;
    byteCnt = 0;
    wordCnt = 0;

    while(count < 1000)
    {
    	if (byteCnt < rxBytes)
    	{
    		pkt128 tcp_notification_pkt;
    		tcp_notification_pkt.data(15,0) = session;
    		if (byteCnt + length <= rxBytes)
    			tcp_notification_pkt.data(31,16) = length;
    		else
    			tcp_notification_pkt.data(31,16) = rxBytes - byteCnt;
    		
    		tcp_notification_pkt.data(80,80) = 0;
    		s_axis_tcp_notification.write(tcp_notification_pkt);
    		byteCnt = byteCnt + length;

    	}

    	if (wordCnt < ntransfers(rxBytes))
		{
			wordCnt ++;
			inword.data = wordCnt;
			inword.last = (wordCnt%(length/64) == 0) | (wordCnt == ntransfers(rxBytes));
			s_axis_tcp_rx_data.write(inword);
			golden.write(inword);
			
		}

    	tcp_rxHandler(   
               s_axis_tcp_notification, 
               m_axis_tcp_read_pkg,
               s_axis_tcp_rx_meta, 
               s_axis_tcp_rx_data,
               s_data_out );

    	if (!m_axis_tcp_read_pkg.empty())
    	{
    		pkt32 readRequest_pkt = m_axis_tcp_read_pkg.read();
    		pkt16 rx_meta_pkt;
    		rx_meta_pkt.data = readRequest_pkt.data(15,0);
    		s_axis_tcp_rx_meta.write(rx_meta_pkt);
    		cout<<"read request: session "<<readRequest_pkt.data(15,0)<<" length "<<readRequest_pkt.data(31,16)<<" at cycle "<<count<<endl;
    	}


    	if (!s_data_out.empty())
    	{
    		outword = s_data_out.read();
			goldenword = golden.read();
			cout<<"rx data "<<outword.data<<" last "<<outword.last<<" goldenword "<<goldenword.data<<" last "<<goldenword.last<<endl;
			if(outword.data != goldenword.data) return 1;
			if(outword.last != goldenword.last) return 1;
    	}


    	count++;
    }


	
	return 0;
}