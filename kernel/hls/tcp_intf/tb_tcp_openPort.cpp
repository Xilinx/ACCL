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
               		stream<pkt8>& s_axis_tcp_port_status);


int main()
{
	stream<ap_uint<32> >  cmd;
	stream<ap_uint<32> >  sts;
	stream<pkt16> m_axis_tcp_listen_port; 
    stream<pkt8> s_axis_tcp_port_status;

    int count = 0;

    while(count < 100)
    {
    	if (count == 1)
    	{
    		cmd.write(5001);
    		cout <<"cmd to open port 5001 at cycle "<<count<<endl;
    	}

    	tcp_openPort(cmd,
					sts,
					m_axis_tcp_listen_port, 
               		s_axis_tcp_port_status);

    	if (!m_axis_tcp_listen_port.empty())
    	{
    		pkt16 listen_port_pkt = m_axis_tcp_listen_port.read();
    		pkt8 port_status_pkt;
    		port_status_pkt.data = 1;
    		s_axis_tcp_port_status.write(port_status_pkt);
    		cout<<"listen port "<<listen_port_pkt.data<<" at cycle "<<count<<endl;
    	}

    	if (!sts.empty())
    	{
    		cout<<"sts "<<sts.read()<<" at cycle "<<count<<endl;
    	}


    	count ++;
    }

	return 0;
}
