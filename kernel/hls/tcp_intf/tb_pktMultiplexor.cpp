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
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <stdint.h>

using namespace hls;
using namespace std;

//Copied from hlslib by Johannes de Fine Licht https://github.com/definelicht/hlslib/blob/master/include/hlslib/xilinx/Utility.h
constexpr unsigned long ConstLog2(unsigned long val) {
  return val == 1 ? 0 : 1 + ConstLog2(val >> 1);
}

#define NUM_MULT_STREAM 8
const unsigned MULT_DEST_BITS = ConstLog2(NUM_MULT_STREAM);

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

typedef ap_axiu<DWIDTH512, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt512r;
typedef ap_axiu<DWIDTH256, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt256r;
typedef ap_axiu<DWIDTH128, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt128r;
typedef ap_axiu<DWIDTH64, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt64r;
typedef ap_axiu<DWIDTH32, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt32r;
typedef ap_axiu<DWIDTH16, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt16r;
typedef ap_axiu<DWIDTH8, MULT_DEST_BITS, MULT_DEST_BITS, MULT_DEST_BITS> pkt8r;

int ntransfers(int nbytes){
	int bytes_per_transfer = 512/8;
	return (nbytes+bytes_per_transfer-1)/bytes_per_transfer;
}


void pktMultiplexor(
                hls::stream<pkt512 >& pktStreamIn,
                hls::stream<ap_uint<64> >& pktMetaIn,
                hls::stream<pkt512r >& pktMultStreamOut               
                );

int main()
{
	hls::stream<pkt512 > pktStreamIn;
    hls::stream<ap_uint<64> > pktMetaIn;
    hls::stream<pkt512r > pktMultStreamOut;   

    stream<pkt512 > golden;

    pkt512 inword;
	pkt512r outword;
	pkt512 goldenword;

    int count;
    ap_uint<16> session;
    ap_uint<16> pktLen;
    ap_uint<32> totalBytes;
    ap_uint<32> byteCnt;
    ap_uint<32> wordCnt;
    ap_uint<32> numPkt;
    ap_uint<32> msgSize;
    int numMsg;

    ap_uint<32> cmdID; // specifier of different communication primitive
    ap_uint<32> cmdLen; // total byte len of compulsory & optional cmd fields
    ap_uint<32> dst; // either dst rank or communicator ID depends on primitive
    ap_uint<32> src; // src rank
    ap_uint<32> tag; // tag, reserved
    ap_uint<32> dataLen; //total byte len of data to each primitive

    cmdID = 1;
    cmdLen = 64;
    dst = 1;
    src = 0;
    tag = 0;
    
    count = 0;
    session = 1;
    totalBytes = 16*1024;
    pktLen = 1024;
    msgSize = 2*1024;
    dataLen = msgSize - cmdLen;
    numPkt = totalBytes / pktLen;
    numMsg = totalBytes / msgSize;
    byteCnt = 0;
    wordCnt = 0;

    // construct the messages
    hls::stream<pkt512 > message [numMsg];
    for (size_t i = 0; i < numMsg; i++)
    {
        for (size_t j = 0; j < msgSize/64; j++)
        {
            // build the header
            if (j == 0)
            {
                inword.data(31,0) = cmdID;
                inword.data(63,32) = cmdLen;
                inword.data(95,64) = dst;
                inword.data(127,96) = src;
                inword.data(159,128) = tag;
                inword.data(191,160) = dataLen;
                inword.data(511,192) = 0;
            }
            else 
            {
                inword.data = i;
            }
            if (j == msgSize/64 -1)
                inword.last = 1;
            else 
                inword.last = 0;
            
            inword.keep = 0xFFFFFFFFFFFFFFFF;
            message[i].write(inword);
            
        }
    }

    printf("Finished constructing messages\n");

    hls::stream<pkt512 > pktQueue;
    hls::stream<ap_uint<64> > metaQueue;

    /* initialize random seed: */
    srand (0);

    while (wordCnt < ntransfers(totalBytes))
    {
        //randomly select the message for next packet
        ap_uint<32> msgInx = (ap_uint<32>) rand() % numMsg;
        // if the message queue is not empty, read and construct a packet
        if (!message[msgInx].empty())
        {
    
            ap_uint<64> pktMeta;
    		pktMeta(31,0) = msgInx % NUM_MULT_STREAM; //use the msgInx as the session number
    		pktMeta(63,32) = pktLen;
    		metaQueue.write(pktMeta);

            cout<<"metaQueue session "<<std::hex<<pktMeta(31,0)<<" pktLen "<<pktMeta(63,32)<<endl;

            for (size_t i = 0; i < ntransfers(pktLen); i++)
            {
                if (message[msgInx].empty())
                {
                    printf("ERRROR\n");
                }
                
                pkt512 msgWord = message[msgInx].read();
                golden.write(msgWord);
                pkt512 currWord;
                currWord.data = msgWord.data;
                currWord.keep = msgWord.keep;
                if (i == ntransfers(pktLen)-1)
                    currWord.last = 1;
                else 
                    currWord.last = 0;
                pktQueue.write(currWord);
                // cout<<"pktQueue data "<<std::hex<<currWord.data<<" last "<<currWord.last<<endl;
            }

            wordCnt = wordCnt + ntransfers(pktLen);
        }
        
    }

    printf("\n\nFinished queuing meta and data\n\n");
    
    byteCnt = 0;
    wordCnt = 0;
    
    while(count < 10000)
    {
    	if (byteCnt < totalBytes)
    	{
    		pktMetaIn.write(metaQueue.read());
    		byteCnt = byteCnt + pktLen;
    	}

    	if (wordCnt < ntransfers(totalBytes))
		{
			wordCnt ++;
            pkt512 currWord = pktQueue.read();
			pktStreamIn.write(currWord);
			
			
		}



    	pktMultiplexor(
                pktStreamIn,
                pktMetaIn,
                pktMultStreamOut               
                );


    	if (!pktMultStreamOut.empty())
    	{
    		outword = pktMultStreamOut.read();
			goldenword = golden.read();
			cout<<"pktMultStreamOut data "<<std::hex<<outword.data<<" last "<<outword.last<<" dest "<<outword.dest<<endl;
			if(outword.data != goldenword.data) return 1;
            if(outword.last != goldenword.last) return 1;
    	}


    	count++;
    }


	
	return 0;
}