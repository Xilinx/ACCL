#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
// #include "communication.hpp"

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
            stream<pkt128>& s_axis_tcp_open_status)
{
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE axis register both port=s_axis_tcp_open_status
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    enum openConRespStateType {WAIT_STATUS, WR_SESSION, WR_IP, WR_PORT, WR_SUCCESS};
    static openConRespStateType openConRespState = WAIT_STATUS;

    pkt128 open_status_pkt;
    static unsigned int session; 
    static unsigned int success; 
    static unsigned int ip; 
    static unsigned int port; 

    switch(openConRespState)
    {   
    case WAIT_STATUS:
        if (!s_axis_tcp_open_status.empty())
        {
            open_status_pkt = s_axis_tcp_open_status.read();
            session = open_status_pkt.data(15,0);
            success = open_status_pkt.data(23,16);
            ip = open_status_pkt.data(55,24);
            port = open_status_pkt.data(71,56);
            openConRespState = WR_SESSION;
        }
    break; 
    case WR_SESSION:
        if (!sts.full())
        {
            sts.write(session);
            openConRespState = WR_IP;
        }
    break;
    case WR_IP:
        if (!sts.full())
        {
            sts.write(ip);
            openConRespState = WR_PORT;
        }
    break;
    case WR_PORT:
        if (!sts.full())
        {
            sts.write(port);
            openConRespState = WR_SUCCESS;
        }
    break;
    case WR_SUCCESS:
        if (!sts.full())
        {
            sts.write(success);
            openConRespState = WAIT_STATUS;
        }
    break;

    }

}