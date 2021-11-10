/*
 * Copyright (c) 2021, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

using namespace hls;
using namespace std;

//Copied from hlslib by Johannes de Fine Licht https://github.com/definelicht/hlslib/blob/master/include/hlslib/xilinx/Utility.h
constexpr unsigned long ConstLog2(unsigned long val) {
  return val == 1 ? 0 : 1 + ConstLog2(val >> 1);
}

#define NUM_MULT_STREAM 8
const int MULT_DEST_BITS = ConstLog2(NUM_MULT_STREAM);

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

typedef ap_axiu<DWIDTH512, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt512r;
typedef ap_axiu<DWIDTH256, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt256r;
typedef ap_axiu<DWIDTH128, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt128r;
typedef ap_axiu<DWIDTH64, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt64r;
typedef ap_axiu<DWIDTH32, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt32r;
typedef ap_axiu<DWIDTH16, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt16r;
typedef ap_axiu<DWIDTH8, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt8r;

struct lookupRespType
{
    bool lookupHit;
    bool isLastPkt;
    ap_uint<MULT_DEST_BITS> dest;
    ap_uint<16> pktLen;
    ap_uint<16> session;
};

struct lookupUpdType
{
    ap_uint<MULT_DEST_BITS> dest;
    ap_uint<16> session;
    ap_uint<16> pktLen;
    ap_uint<32> msgSize;
};

// For every packet, lookup the dest router specifier according to the session id
// If this is the first packet of the message, return a lookup miss, a free dest
// The processing unit receives a miss and will parse the message header and update the lookup table with the message size
// The following packets of the same message will return a lookup hit. 
// If a full message is consumed, issue the dest with the isLastPkt signal set and clear the entry in the lookup table
void session_dest_lookup(hls::stream<ap_uint<64> >& lookupReq,
                    hls::stream<lookupRespType>& lookupResp,
                    hls::stream<lookupUpdType>& lookupUpd
                    )
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off    

    static bool sessionTableValid [NUM_MULT_STREAM];
    static ap_uint<16> sessionTableSession [NUM_MULT_STREAM];
    static ap_uint<32> sessionTableTotalMsgSize [NUM_MULT_STREAM];
    static ap_uint<32> sessionTableRcvdMsgSize [NUM_MULT_STREAM];

 #pragma HLS ARRAY_PARTITION variable=sessionTableValid complete 
 #pragma HLS ARRAY_PARTITION variable=sessionTableSession complete 
 #pragma HLS ARRAY_PARTITION variable=sessionTableTotalMsgSize complete 
 #pragma HLS ARRAY_PARTITION variable=sessionTableRcvdMsgSize complete 

    static hls::stream<ap_uint<NUM_MULT_STREAM> > freeDest;
    #pragma HLS stream variable=freeDest depth=32

    enum lookupFsmStateTyte {IDLE, REQ, HIT_RESP, MISS_RESP, UPDATE};
    static lookupFsmStateTyte  lookupFsmState = IDLE;

    static lookupRespType resp;
    static lookupUpdType upd;

    static ap_uint<NUM_MULT_STREAM> mask = 0;
    static ap_uint<MULT_DEST_BITS> dest = 0;

    static ap_uint<8> counter = 0;
    static ap_uint<32> session = 0;
    static ap_uint<32> pktLen = 0;
    static ap_uint<32> msgSize = 0;


    switch(lookupFsmState)
    {
    //Populate the freeDest Fifo and initialize the array
    case IDLE:
        freeDest.write(counter);
        sessionTableValid[counter] = 0;
        sessionTableSession[counter] = 0;
        sessionTableTotalMsgSize[counter] = 0;
        sessionTableRcvdMsgSize[counter] = 0;
        counter++;
        if (counter == NUM_MULT_STREAM)
        {
            counter = 0;
            lookupFsmState = REQ;
        }
        
    break;
    case REQ:
        if (!lookupReq.empty())
        {
            ap_uint<64> req = lookupReq.read();
            session = req(31,0);
            pktLen = req(63,32);
            //Parallel compare the session ID with the valid table entry
            for (unsigned int i = 0; i < NUM_MULT_STREAM; i++)
            {
            #pragma HLS UNROLL
                mask(i,i) = sessionTableValid[i] & (sessionTableSession[i] == session);
            }  

            #ifndef __SYNTHESIS__
            std::cout<<std::endl;
            std::cout<<"-----lookup Req, session: "<<session<<", pktLen:"<<pktLen<<std::endl;
            std::cout<<"-----comparison mask: "<<mask<<std::endl;
            std::cout<<"-----lookup table-----"<<std::endl;
            for (size_t i = 0; i < NUM_MULT_STREAM; i++)
            {
                std::cout<<"valid:"<<sessionTableValid[i]<<", session:"<<sessionTableSession[i]<<", msgSize:"<<sessionTableTotalMsgSize[i]<<", rcvdSize:"<<sessionTableRcvdMsgSize[i]<<std::endl;
            }
            #endif

            if (mask != 0)
                lookupFsmState = HIT_RESP;
            else 
                lookupFsmState = MISS_RESP;
        }
    break;
    case HIT_RESP:
        if(mask == 0b00000001)
            dest = 0;
        else if (mask == 0b00000010)
            dest = 1;
        else if (mask == 0b00000100)
            dest = 2;
        else if (mask == 0b00001000)
            dest = 3;
        else if (mask == 0b00010000)
            dest = 4;
        else if (mask == 0b00100000)
            dest = 5;
        else if (mask == 0b01000000)
            dest = 6;
        else if (mask == 0b10000000)
            dest = 7;
               
        // If this is last packet of a message, clear register and write free dest to fifo
        // Write the response
        if ((sessionTableRcvdMsgSize[dest] + pktLen) >= sessionTableTotalMsgSize[dest])
        {
            sessionTableSession[dest] = 0;
            sessionTableRcvdMsgSize[dest] = 0;
            sessionTableTotalMsgSize[dest] = 0;
            sessionTableValid[dest] = 0;
            freeDest.write(dest);
            resp.lookupHit = true;
            resp.isLastPkt = true;
            resp.dest = dest;
            resp.session = session;
            resp.pktLen = pktLen;
            lookupResp.write(resp);
        }
        else 
        {
            sessionTableRcvdMsgSize[dest] = sessionTableRcvdMsgSize[dest] + pktLen;
            resp.lookupHit = true;
            resp.isLastPkt = false;
            resp.dest = dest;
            resp.session = session;
            resp.pktLen = pktLen;
            lookupResp.write(resp);
        }
        #ifndef __SYNTHESIS__
        std::cout<<"-----lookup Hit: dest:"<<dest<<std::endl;
        std::cout<<"-----lookup table-----"<<std::endl;
        for (size_t i = 0; i < NUM_MULT_STREAM; i++)
        {
            std::cout<<"valid:"<<sessionTableValid[i]<<", session:"<<sessionTableSession[i]<<", msgSize:"<<sessionTableTotalMsgSize[i]<<", rcvdSize:"<<sessionTableRcvdMsgSize[i]<<std::endl;
        }
        #endif
        lookupFsmState = REQ;
    break;
    // If miss, read from the free dest fifo, report a lookup miss
    case MISS_RESP:
        if (!freeDest.empty())
        {
            dest = freeDest.read();
            resp.lookupHit = false;
            resp.isLastPkt = false;
            resp.dest = dest;
            resp.session = session;
            resp.pktLen = pktLen;
            lookupResp.write(resp);
            lookupFsmState = UPDATE;
            #ifndef __SYNTHESIS__
            std::cout<<"-----lookup Miss: dest:"<<dest<<std::endl;
            #endif
        }
    break;
    // Upon the miss, wait for the message information to update the lookup table
    // If the message is a single packet message, no need to store it in the lookup table, free the dest
    case UPDATE:
        if (!lookupUpd.empty())
        {
            upd = lookupUpd.read();
            dest = upd.dest;
            pktLen = upd.pktLen;
            session = upd.session;
            msgSize = upd.msgSize;
            if (pktLen >= upd.msgSize)
            {
                sessionTableSession[dest] = 0;
                sessionTableRcvdMsgSize[dest] = 0;
                sessionTableTotalMsgSize[dest] = 0;
                sessionTableValid[dest] = 0;
                freeDest.write(dest);
            }
            else 
            {
                sessionTableValid[dest] = 1;
                sessionTableSession[dest] = session;
                sessionTableTotalMsgSize[dest] = msgSize;
                sessionTableRcvdMsgSize[dest] = pktLen;
            }

            lookupFsmState = REQ;

            #ifndef __SYNTHESIS__
            std::cout<<"-----lookup Upd: dest:"<<dest<<std::endl;
            std::cout<<"-----lookup table-----"<<std::endl;
            for (size_t i = 0; i < NUM_MULT_STREAM; i++)
            {
                std::cout<<"valid:"<<sessionTableValid[i]<<", session:"<<sessionTableSession[i]<<", msgSize:"<<sessionTableTotalMsgSize[i]<<", rcvdSize:"<<sessionTableRcvdMsgSize[i]<<std::endl;
            }
            #endif
        }
    break;

    }
}


void procStream(hls::stream<pkt512 >& pktStreamIn,
                hls::stream<pkt512r >& pktMultStreamOut,
                hls::stream<lookupRespType>& lookupResp,
                hls::stream<lookupUpdType>& lookupUpd
                )
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off   

    enum FsmStateTyte {WAIT_STREAM, CONSUME};
    static FsmStateTyte  FsmState = WAIT_STREAM;

    static lookupRespType resp;
    static lookupUpdType upd;

    pkt512 streamInWord;
    pkt512r streamOutWord;

    switch (FsmState)
    {
    case WAIT_STREAM:
        if (!lookupResp.empty() & !pktStreamIn.empty())
        {
            resp = lookupResp.read();
            streamInWord = pktStreamIn.read();
            if (resp.lookupHit == 0)
            {
                // msgSize includes the header size and the data payload size
                upd.msgSize = streamInWord.data(63,32) + streamInWord.data(191,160);
                upd.session = resp.session;
                upd.dest = resp.dest;
                upd.pktLen = resp.pktLen;
                lookupUpd.write(upd);
            }
            streamOutWord.data = streamInWord.data;
            streamOutWord.keep = streamInWord.keep;
            streamOutWord.last = streamInWord.last & resp.isLastPkt;
            streamOutWord.dest = resp.dest;    
            pktMultStreamOut.write(streamOutWord);
            if (streamInWord.last != 1)
                FsmState = CONSUME;
        }
        break;
    case CONSUME:
        if (!pktStreamIn.empty())
        {
            streamInWord = pktStreamIn.read();
            streamOutWord.data = streamInWord.data;
            streamOutWord.keep = streamInWord.keep;
            streamOutWord.last = streamInWord.last & resp.isLastPkt;
            streamOutWord.dest = resp.dest;    
            pktMultStreamOut.write(streamOutWord);
            if (streamInWord.last)
            {
                FsmState = WAIT_STREAM;
            }
            
        }
    }
}


void pktMultiplexor(
                hls::stream<pkt512 >& pktStreamIn,
                hls::stream<ap_uint<64> >& pktMetaIn,
                hls::stream<pkt512r >& pktMultStreamOut               
                )
{
#pragma HLS INTERFACE axis register  port=pktStreamIn
#pragma HLS INTERFACE axis register  port=pktMetaIn
#pragma HLS INTERFACE axis register  port=pktMultStreamOut
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW disable_start_propagation

// #pragma HLS PIPELINE II=1
// #pragma HLS INLINE off

    static hls::stream<lookupRespType> lookupResp;
    #pragma HLS stream variable=lookupResp depth=8

    static hls::stream<lookupUpdType> lookupUpd;
    #pragma HLS stream variable=lookupUpd depth=8


    session_dest_lookup(pktMetaIn,
                    lookupResp,
                    lookupUpd
                    );

    procStream( pktStreamIn,
                pktMultStreamOut,
                lookupResp,
                lookupUpd
                );


}

