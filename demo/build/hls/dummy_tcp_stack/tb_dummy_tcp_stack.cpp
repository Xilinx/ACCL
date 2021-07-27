# /*******************************************************************************
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
#include "axi_utils.hpp"
#include "packet.hpp"
#include "toe.hpp"

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

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

void network_krnl (
			   	stream<pkt128>& m_axis_tcp_notification, 
               	stream<pkt32>& s_axis_tcp_read_pkg,
               	stream<pkt16>& m_axis_tcp_rx_meta, 
               	stream<pkt512>& m_axis_tcp_rx_data,
               	stream<pkt32>& s_axis_tcp_tx_meta, 
            	stream<pkt512>& s_axis_tcp_tx_data, 
            	stream<pkt64>& m_axis_tcp_tx_status,
            	stream<pkt64>& s_axis_tcp_open_connection,
            	stream<pkt128>& m_axis_tcp_open_status,
            	stream<pkt16>& s_axis_tcp_listen_port, 
               	stream<pkt8>& m_axis_tcp_port_status,
               	stream<pkt512>& net_rx,
               	stream<pkt512>& net_tx);

int main()
{

	stream<pkt128> m_axis_tcp_notification; 
	stream<pkt32> s_axis_tcp_read_pkg;
	stream<pkt16> m_axis_tcp_rx_meta;
	stream<pkt512> m_axis_tcp_rx_data;
	stream<pkt32> s_axis_tcp_tx_meta;
	stream<pkt512> s_axis_tcp_tx_data; 
	stream<pkt64> m_axis_tcp_tx_status;
	stream<pkt64> s_axis_tcp_open_connection;
	stream<pkt128> m_axis_tcp_open_status;
	stream<pkt16> s_axis_tcp_listen_port;
	stream<pkt8> m_axis_tcp_port_status;
	stream<pkt512> net_rx;
	stream<pkt512> net_tx;

	int count = 0;
	ap_uint<16> port = 5001;
	ap_uint<32> ip = 0xDEADBEEF;
	ap_uint<16> session = 0;
	ap_uint<16> payload_length = 1024;
	ap_uint<16> msg_length = payload_length + 64;
	bool sendPkg = false;
	bool header = false;
	ap_uint<16> byteCnt = 0;
	int net_word_cnt = 0;

	while(count < 1000)
	{

		if (count == 1)
		{
			
			pkt16 listen_port_pkt;
			listen_port_pkt.data = port;
			s_axis_tcp_listen_port.write(listen_port_pkt);
			cout<<"listen port request "<<listen_port_pkt.data<<" at cycle "<<count<<endl;
		}

		if (count == 20)
		{
			pkt64 open_connection_pkt;
			open_connection_pkt.data(31,0) = ip;
			open_connection_pkt.data(47,32) = port;
			s_axis_tcp_open_connection.write(open_connection_pkt);
			cout<<"open connection ip "<<hex<<open_connection_pkt.data(31,0)<<" port "<<dec<<open_connection_pkt.data(47,32)<<" at cycle "<<count<<endl;
		}

		if (count == 30)
		{
			pkt32 tx_meta_pkt;
			tx_meta_pkt.data(15,0) = session;
			tx_meta_pkt.data(31,16) = msg_length;
			s_axis_tcp_tx_meta.write(tx_meta_pkt);
			cout<<"tx meta session "<<session<<" msg length "<<msg_length<<" at cycle"<<count<<endl;
		}

		network_krnl (
			   	 m_axis_tcp_notification, 
               	 s_axis_tcp_read_pkg,
               	 m_axis_tcp_rx_meta, 
               	 m_axis_tcp_rx_data,
               	 s_axis_tcp_tx_meta, 
            	 s_axis_tcp_tx_data, 
            	 m_axis_tcp_tx_status,
            	 s_axis_tcp_open_connection,
            	 m_axis_tcp_open_status,
            	 s_axis_tcp_listen_port, 
               	 m_axis_tcp_port_status,
               	 net_rx,
               	 net_tx);

		if (!m_axis_tcp_port_status.empty())
		{
			pkt8 port_status_pkt = m_axis_tcp_port_status.read();
			cout<<"port status:"<<port_status_pkt.data<<" at cycle "<<count<<endl;
		}

		if (!m_axis_tcp_open_status.empty())
		{
			pkt128 open_status_pkt = m_axis_tcp_open_status.read();
			session = open_status_pkt.data(15,0);
			cout<<"open status: session:"<<open_status_pkt.data(15,0)<<",success:"<<open_status_pkt.data(23,16)<<",port:"<<open_status_pkt.data(71,56)<<", ip:"<<hex<<open_status_pkt.data(55,24)<<" at cycle:"<<dec<<count<<endl;
		}

		if (!m_axis_tcp_tx_status.empty())
		{
			pkt64 tx_status = m_axis_tcp_tx_status.read();
			sendPkg = true;
			header = true;
			cout<<"tx status session "<<tx_status.data(15,0)<<"msg length "<<tx_status.data(31,16)<<" remaining space "<<tx_status.data(61,32)<<" error "<<tx_status.data(63,62)<<" at cycle"<<count<<endl;
		}

		if (sendPkg)
		{
			pkt512 tx_data_pkt;
			if (header)
			{
				tx_data_pkt.data(31,0) = payload_length;
				tx_data_pkt.data(63,32) = 5; //tag
				tx_data_pkt.data(95,64) = 1; //src
				tx_data_pkt.last = 0;
				s_axis_tcp_tx_data.write(tx_data_pkt);
				header = false;
				cout<<"tx data header: payload length "<<tx_data_pkt.data(31,0)<<" tag "<<tx_data_pkt.data(63,32)<<" src "<< tx_data_pkt.data(95,64)<<" at cycle "<<count<<endl;
			}
			else 
			{
				byteCnt = byteCnt + 64;
				tx_data_pkt.data = 0;
				tx_data_pkt.last = 0;

				if (byteCnt >= payload_length)
				{
					tx_data_pkt.last = 1;
					sendPkg = false;
					cout<<"tx data word byte count"<<byteCnt<<" at cycle "<<count<<" last "<<tx_data_pkt.last<<endl;
					byteCnt = 0;
				}
				else 
					cout<<"tx data word byte count"<<byteCnt<<" at cycle "<<count<<" last "<<0<<endl;
				s_axis_tcp_tx_data.write(tx_data_pkt);
				
			}
		}


		if (!net_tx.empty())
		{
			pkt512 net_tx_pkt = net_tx.read();
			net_rx.write(net_tx_pkt);
			net_word_cnt++;
		}

		if (!m_axis_tcp_notification.empty())
		{
			pkt128 tcp_notification_pkt = m_axis_tcp_notification.read();
          	cout<<"noti session:"<<tcp_notification_pkt.data(15,0)<<" msg length "<<tcp_notification_pkt.data(31,16)<<" ip "<<tcp_notification_pkt.data(63,32)<<" port "<<tcp_notification_pkt.data(79,64)<<" closed "<<tcp_notification_pkt.data(80,80)<<" at cycle"<< count<<endl;
          	pkt32 readRequest_pkt;
          	readRequest_pkt.data(15,0) = tcp_notification_pkt.data(15,0);
            readRequest_pkt.data(31,16) = tcp_notification_pkt.data(31,16);
            s_axis_tcp_read_pkg.write(readRequest_pkt);
		}

		if (!m_axis_tcp_rx_meta.empty())
		{
			pkt16 rx_meta_pkt = m_axis_tcp_rx_meta.read();
			cout<<"rx meta: "<<rx_meta_pkt.data<<" at cycle "<<count<<endl;
		}

		if (!m_axis_tcp_rx_data.empty())
		{
			pkt512 rx_data_pkt = m_axis_tcp_rx_data.read();
			cout<<"rx data at cycle "<<count<<endl;
		}


		count++;
	}


	return 0;
}