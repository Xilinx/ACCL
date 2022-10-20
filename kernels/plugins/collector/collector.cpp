/************************************************
Copyright (c) 2020, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors 
may be used to endorse or promote products derived from this software 
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.// Copyright (c) 2020 Xilinx, Inc.
************************************************/

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

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

#define CMD_WORD 15
#define STS_WORD 1

void cmdHandler(ap_uint<32> round,
                ap_uint<64>* cmdOut,
                stream<pkt32> &cmdIn,
                stream<pkt64> &cmdTimestamp)
{
    ap_uint<64> cmd_mem_offset = 0;

    for(int i=0; i< round; i++)
    {
    #pragma HLS PIPELINE

        // write sequence number
        cmdOut[cmd_mem_offset] = i;
        cmd_mem_offset++;

        // write cmd and timestamp
        for (int j = 0; j < CMD_WORD; j++)
        {
        #pragma HLS PIPELINE 
            pkt32 cmd = cmdIn.read();
            cmdOut[cmd_mem_offset] = cmd.data;
            cmd_mem_offset++;
        }
        pkt64 cmdTime = cmdTimestamp.read();
        cmdOut[cmd_mem_offset] = cmdTime.data;
        cmd_mem_offset++;
        // write cmd end signal
        cmdOut[cmd_mem_offset] = 0xFFFFFFFFFFFFFFFF;
        cmd_mem_offset++;
    }
}

void stsHandler(ap_uint<32> round,
                ap_uint<64>* stsOut,
                stream<pkt32> &stsIn,
                stream<pkt64> &stsTimestamp)
{
    ap_uint<64> sts_mem_offset = 0;

    for(int i=0; i< round; i++)
    {
    #pragma HLS PIPELINE

        // write sequence number
        stsOut[sts_mem_offset] = i;
        sts_mem_offset++;

        // write sts and timestamp
        for (int j = 0; j < STS_WORD; j++)
        {
        #pragma HLS PIPELINE 
            pkt32 sts = stsIn.read();
            stsOut[sts_mem_offset] = sts.data;
            sts_mem_offset++;
        }
        pkt64 stsTime = stsTimestamp.read();
        stsOut[sts_mem_offset] = stsTime.data;
        sts_mem_offset++;
        // write sts end signal
        stsOut[sts_mem_offset] = 0xFFFFFFFFFFFFFFFF;
        sts_mem_offset++;
    }
}


void collector (
                ap_uint<32> round,
                ap_uint<64>* cmdOut,
                ap_uint<64>* stsOut,
                stream<pkt32> &cmdIn,
                stream<pkt64> &cmdTimestamp,
                stream<pkt32> &stsIn,
                stream<pkt64> &stsTimestamp
                )
{
#pragma HLS INTERFACE s_axilite port=round
#pragma HLS INTERFACE axis register both port=cmdIn
#pragma HLS INTERFACE axis register both port=cmdTimestamp
#pragma HLS INTERFACE axis register both port=stsIn
#pragma HLS INTERFACE axis register both port=stsTimestamp
#pragma HLS INTERFACE m_axi port=cmdOut offset=slave bundle = gmem0
#pragma HLS INTERFACE m_axi port=stsOut offset=slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS dataflow

    cmdHandler( round,
                cmdOut,
                cmdIn,
                cmdTimestamp);

    stsHandler( round,
                stsOut,
                stsIn,
                stsTimestamp);

}



