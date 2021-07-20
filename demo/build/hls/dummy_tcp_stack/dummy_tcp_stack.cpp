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

void tx_handler (
				stream<pkt32>& s_axis_tcp_tx_meta, 
            	stream<pkt512>& s_axis_tcp_tx_data, 
            	stream<pkt64>& m_axis_tcp_tx_status,
            	stream<pkt512>& out)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

	enum txFsmStateType {WAIT_META, WRITE_STATUS, TX_DATA};
	static txFsmStateType txFsmState = WAIT_META;

	static ap_uint<16> session = 0;
	static ap_uint<16> length = 0;
	static ap_uint<16> recvByte = 0;

	switch(txFsmState)
	{
	case WAIT_META:
		if (!s_axis_tcp_tx_meta.empty())
		{
			pkt32 tx_meta_pkt = s_axis_tcp_tx_meta.read();
			session = tx_meta_pkt.data(15,0);
			length = tx_meta_pkt.data(31,16);
			txFsmState = WRITE_STATUS;
		}
	break;
	case WRITE_STATUS:
		if (!m_axis_tcp_tx_status.full())
		{
			pkt64 tx_status_pkt;
			tx_status_pkt.data(15,0) = session;
			tx_status_pkt.data(31,16) = length;
			tx_status_pkt.data(61,32) = 8000;
			tx_status_pkt.data(63,62) = 0;
			m_axis_tcp_tx_status.write(tx_status_pkt);
			txFsmState = TX_DATA;
		}
	break;
	case TX_DATA:
		if (!s_axis_tcp_tx_data.empty())
		{
			out.write(s_axis_tcp_tx_data.read());
			recvByte = recvByte + 64;
			if (recvByte >= length)
			{
				recvByte = 0;
				txFsmState = WAIT_META;
			}
		}
	break;

	}//switch

}

void rx_handler(stream<pkt128>& m_axis_tcp_notification, 
               	stream<pkt32>& s_axis_tcp_read_pkg,
               	stream<pkt16>& m_axis_tcp_rx_meta, 
               	stream<pkt512>& m_axis_tcp_rx_data,
               	stream<pkt512>& in)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

static hls::stream<net_axis<512> >	rxDataBuffer;
#pragma HLS STREAM variable=rxDataBuffer depth=512

	enum rxFsmStateType {BUFFER_HEADER, BUFFER_DATA, SEND_NOTI, WAIT_REQ, SEND_MSG};
	static rxFsmStateType rxFsmState = BUFFER_HEADER;

	static int payload_length = 0;
	static int msg_length = 0;
	static int count = 0;
	net_axis<512> currWord;

	switch (rxFsmState)
	{
	case BUFFER_HEADER:
		if (!in.empty())
		{
			pkt512 inword = in.read();
			payload_length = (inword.data)(31,0);
			msg_length = payload_length + 64;
			currWord.data = inword.data;
			currWord.keep = inword.keep;
			currWord.last = inword.last;
			rxDataBuffer.write(currWord);
			rxFsmState = BUFFER_DATA;
		}
	break;
	case BUFFER_DATA:
		if (!in.empty())
		{
			pkt512 inword = in.read();
			currWord.data = inword.data;
			currWord.keep = inword.keep;
			currWord.last = inword.last;
			rxDataBuffer.write(currWord);
			count = count + 64;
			if (count >= payload_length)
			{
				count = 0;
				rxFsmState = SEND_NOTI;
			}
		}
	break;
	case SEND_NOTI:
		if (!m_axis_tcp_notification.full())
		{
			//we acutally don't care about session, ip, port since the 
		  //rank is encoded in the header
          pkt128 tcp_notification_pkt;
          tcp_notification_pkt.data(15,0) = 0; //session
          tcp_notification_pkt.data(31,16) = msg_length; //length of the data plus header
          tcp_notification_pkt.data(63,32) = 0; //ip
          tcp_notification_pkt.data(79,64) = 0; //port
          tcp_notification_pkt.data(80,80) = 0; //close
          m_axis_tcp_notification.write(tcp_notification_pkt);
          rxFsmState = WAIT_REQ;
		}
	break;
	case WAIT_REQ:
		if (!s_axis_tcp_read_pkg.empty())
		{
			s_axis_tcp_read_pkg.read();
			pkt16 rx_meta_pkt;
			rx_meta_pkt.data = 0; //we don't care about the session id in this dummy
			m_axis_tcp_rx_meta.write(rx_meta_pkt);
			rxFsmState = SEND_MSG;
		}
	break;
	case SEND_MSG:
		if (!rxDataBuffer.empty())
		{
			pkt512 rx_data_pkt;
			net_axis<512> buffer_word;
			buffer_word = rxDataBuffer.read();
			rx_data_pkt.data = buffer_word.data;
			rx_data_pkt.keep = buffer_word.keep;
			rx_data_pkt.last = buffer_word.last;
			m_axis_tcp_rx_data.write(rx_data_pkt);

			count = count + 64;
			if (count >= msg_length)
			{
				count = 0;
				msg_length = 0;
				payload_length = 0;
				rxFsmState = BUFFER_HEADER;
			}
		}
		
	break;

	}//switch
}

//[15:0] session; [23:16] success; [55:24] ip; [71:56] port 
void open_con_handler(
				stream<pkt64>& s_axis_tcp_open_connection,
            	stream<pkt128>& m_axis_tcp_open_status)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

	static int sessionCnt = 0;
	//openConReq and openConRsp
	if (!s_axis_tcp_open_connection.empty())
	{
		pkt64 openConnection_pkt = s_axis_tcp_open_connection.read();
		ap_uint<32> ip = openConnection_pkt.data(31,0);
		ap_uint<16> port = openConnection_pkt.data(47,32);

		pkt128 open_status_pkt;
		open_status_pkt.data(15,0) = sessionCnt;
		open_status_pkt.data(23,16) = 1;
		open_status_pkt.data(55,24) = ip;
		open_status_pkt.data(71,56) = port;
		m_axis_tcp_open_status.write(open_status_pkt);
		sessionCnt ++;
	}
}


void listen_port_handler(stream<pkt16>& s_axis_tcp_listen_port, 
               	stream<pkt8>& m_axis_tcp_port_status)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

	//listen port and listen port status
	if (!s_axis_tcp_listen_port.empty())
	{
		pkt16 listen_port_pkt = s_axis_tcp_listen_port.read();
		pkt8 port_status_pkt;
		port_status_pkt.data = 1;
		m_axis_tcp_port_status.write(port_status_pkt);
	}
}

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
               	stream<pkt512>& net_tx)
{
#pragma HLS INTERFACE axis register both port=m_axis_tcp_notification
#pragma HLS INTERFACE axis register both port=s_axis_tcp_read_pkg
#pragma HLS INTERFACE axis register both port=m_axis_tcp_rx_meta
#pragma HLS INTERFACE axis register both port=m_axis_tcp_rx_data
#pragma HLS INTERFACE axis register both port=s_axis_tcp_tx_meta
#pragma HLS INTERFACE axis register both port=s_axis_tcp_tx_data
#pragma HLS INTERFACE axis register both port=m_axis_tcp_tx_status
#pragma HLS INTERFACE axis register both port=s_axis_tcp_open_connection
#pragma HLS INTERFACE axis register both port=m_axis_tcp_open_status
#pragma HLS INTERFACE axis register both port=s_axis_tcp_listen_port
#pragma HLS INTERFACE axis register both port=m_axis_tcp_port_status
#pragma HLS INTERFACE axis register both port=net_rx
#pragma HLS INTERFACE axis register both port=net_tx
#pragma HLS INTERFACE ap_ctrl_none port=return
	
#pragma HLS DATAFLOW disable_start_propagation
	
open_con_handler(
				s_axis_tcp_open_connection,
            	m_axis_tcp_open_status);

listen_port_handler(s_axis_tcp_listen_port, 
               	m_axis_tcp_port_status);

rx_handler(m_axis_tcp_notification, 
           s_axis_tcp_read_pkg,
           m_axis_tcp_rx_meta, 
           m_axis_tcp_rx_data,
           net_rx);


tx_handler (
			s_axis_tcp_tx_meta, 
            s_axis_tcp_tx_data, 
            m_axis_tcp_tx_status,
            net_tx);



}
