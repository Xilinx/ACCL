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


void tcp_openConResp(   
            stream<ap_uint<32> > & sts,
            stream<pkt128>& s_axis_tcp_open_status);

int main()
{
	stream<ap_uint<32> >  sts;
    stream<pkt128> s_axis_tcp_open_status;

    int count = 0;
    pkt128 open_status_pkt;

    while(count < 100)
    {
    	if (count == 1)
    	{
            open_status_pkt.data(15,0) = 0; //session
            open_status_pkt.data(23,16) = 1; //success
            open_status_pkt.data(55,24) = 0xDEADBEEF; //ip
            open_status_pkt.data(71,56) = 5001; //port
            s_axis_tcp_open_status.write(open_status_pkt);
            cout <<"open status at cycle "<<count<<endl;
        }

        tcp_openConResp(   
            sts,
            s_axis_tcp_open_status);

        if (!sts.empty())
        {
        	cout<<"sts "<<hex<<sts.read()<<" at  cycle "<<count<<endl;
        }

    	count ++;
    }

	return 0;
}
