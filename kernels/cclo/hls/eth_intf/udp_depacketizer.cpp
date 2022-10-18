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

#include "eth_intf.h"
#include <iostream>

using namespace std;

// This block parses input data and splits it into messages and fragments, signalling up to the RX offload
//
// Definitions:
// Message - a contiguously-sent sequence of data on a connection, prepended by a header as defined in eth_intf.h
// Fragment - a contiguously-received sequence of data on a connection
//
// A message is always sent in one chunk but at the receiver it may interleave with another message
// on a different connection. In this case, the receiver will see data from each incoming message 
// in interleaved fragments. The first fragment of each message carries the message's header.
//
// The ingress pipeline signals the connection ID in side-band AXI Stream signal TDEST
// Each fragment signals its own end with AXI Stream TLAST
//
// Each message is 1 or more fragments at the receiver. Fragments are received as an integer number
// of AXI Stream transactions. 
// This means any one AXI Stream word can only have data from one message, not more.
// The sender can ensure this by padding messages to a multiple of the AXI Stream width
//
// In response to incoming fragments, the depacketizer performs the following functions:
// - keep track of active messages on each connection. On initialization, no messages are active
// - register start of a message on incoming data to a connection without an active message
// - strip header and forward it to RX buffer offload logic (RXBO) with a start of message (SOM) notification
// - get size of active message from header and store it
// - signal start of fragment (SOF) notification to RXBO
// - forward message data to the RX DMA and keep track of how much data was forwarded
// - on TLAST or end of message (calculated from message size), signal end of fragment (EOF) notification to RXBO

//how this works:
//the UDP/ROCE POE pushes B bytes into the input stream for session S
//B is not known ahead of time; end of message/packet signaled with TLAST and TKEEP
//read one word of input and get S from TDEST
//check how many bytes remaining for any ongoing messages on session S
//if remaining[S] == 0
//    it means we're getting the start of a new message, so
//        get the header from the input data stream, indicating how many bytes M are in the message, and destination stream strm
//        check strm in header; if strm is zero:
//            put a notification in the notif_out stream; the notification signals session S and length M
//            copy the header to sts
//        copy from in to out, with dest = strm, decrementing M along the way, until TLAST or M reaches zero; we keep track how much bytes we've copied (B)
//        if strm is zero:
//            put a notification in the notif_out stream; the notification signals S and remaining[S]
//else
//    it means we're continuing a previous message on this session, so
//    write the input word to output (it's not a header in this case)
//    if strm is zero:
//        put a notification in the notif_out stream; the notification signals session S and length remaining[S]
//    copy from in to out, with dest = strm, decrementing remaining[S] along the way, until TLAST or remaining[S] reaches 0; we keep track how much bytes we've copied (B)
//    if strm is zero:
//        put a notification in the notif_out stream; the notification signals S and remaining[S]
//remaining[S] = {M, strm}

void udp_depacketizer(
    STREAM<stream_word > & in,
    STREAM<stream_word > & out,
    STREAM<eth_header> & sts,
    STREAM<eth_notification> &notif_out
) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE axis register both port=notif_out
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS PIPELINE II=1 style=flp

	unsigned const bytes_per_word = DATA_WIDTH/8;
	static unsigned int remaining[1024] = {0};//one RAMB36 for max 1024 sessions
	static ap_uint<DEST_WIDTH> target_strm[1024] = {0};//one RAMB18 for max 1024 sessions

	static eth_notification notif;
	static bool continue_notif = false;

	static unsigned int message_rem = 0;
	static ap_uint<DEST_WIDTH> strm = 0;
	static unsigned int prev_session_id = 0;

	stream_word inword;
	eth_header hdr;

	if(STREAM_IS_EMPTY(in)) return;
	inword = STREAM_READ(in);
	notif.session_id = inword.dest;
	notif.length = 0xffff;

#ifndef ACCL_SYNTHESIS
	std::stringstream ss;
	ss << "UDP Depacketizer: Processing incoming fragment\n";
	std::cout << ss.str();
#endif

	//get remaining message bytes, from local storage
	//TODO: cache latest accessed value
	if(prev_session_id != notif.session_id){
		message_rem = remaining[notif.session_id];
	}
	
	if(message_rem == 0){//if remaining bytes is zero, then this is the start of a new message
		//get header and some important info from it
		hdr = eth_header(inword.data(HEADER_LENGTH-1,0));
		message_rem = hdr.count;//length of upcoming message (excluding the header itself)
		strm = hdr.strm;//target of message (0 is targeting memory so managed, everything else is  stream so unmanaged)
		if(strm == 0){
			//put notification, header in output streams
			STREAM_WRITE(sts, hdr);
		}
		//decrement the length to reflect the fact that we have removed the 64B header
		//Note: the rxHandler must make sure to not give us fragments less than 64B
		notif.length -= bytes_per_word;
		target_strm[notif.session_id] = strm;
	} else{//if remaining bytes is not zero, then this is a continuation of an old message
		strm = target_strm[notif.session_id];
		STREAM_WRITE(out, inword);
	}
	//write out notification
	//in case the fragment spans the end of the current message and beginning of another,
	//only notify for the part up to the end of the current message
	if(strm == 0){
		eth_notification downstream_notif;
		downstream_notif.session_id = notif.session_id;
		downstream_notif.length = (message_rem < notif.length) ? message_rem : (unsigned int)notif.length;
		STREAM_WRITE(notif_out, downstream_notif);
	}
	//copy data in -> out
	do{
		#pragma HLS PIPELINE II=1
		inword = STREAM_READ(in);
		inword.dest = strm;
		STREAM_WRITE(out, inword);
		notif.length = (notif.length < bytes_per_word) ? 0u : (unsigned int)notif.length-bytes_per_word;//floor at zero
		message_rem = (message_rem < bytes_per_word) ? 0u : message_rem-bytes_per_word;//slight problem here if the message doesnt end on a 64B boundary...
	} while(inword.last != 0 && message_rem > 0);
	//update session info (remaining bytes and target of currently processing message)
	remaining[notif.session_id] = message_rem;
	//if we're not finished with this fragment, skip notification read on the next run
	continue_notif = (notif.length > 0);
	//update session id for caching
	prev_session_id = notif.session_id;
}