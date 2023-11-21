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
#include "ap_int.h"
#include "eth_intf.h"
#ifndef ACCL_SYNTHESIS
#include "log.hpp"

extern Log logger;
#endif

using namespace std;

//how this works:
//read a notification from notif_in which says we'll get B bytes for session S
//check how many bytes remaining for any ongoing messages on session S
//if remaining[S] == 0
//    it means we're getting the start of a new message, so
//        get the header from the input data stream, indicating how many bytes M are in the message, and destination stream strm
//        check strm in header; if strm is zero:
//            copy the notification to the notif_out stream
//            copy the header to sts
//            subtract 64 from B
//        copy all B remaining bytes from in to out, with dest = strm, decrementing M along the way
//else
//    it means we're continuing a previous message on this session, so
//    copy min(remaining[S], B) from in to out, with dest = strm, decrementing remaining[S] along the way
//    if strm is zero:
//        copy the notification to the notif_out stream
//remaining[S] = {M, strm}
//if bytes still remaining for this notification, start over without reading notif (keep old notif, with adjusted B)

void tcp_depacketizer(
	STREAM<stream_word > & in,
	STREAM<stream_word > & out,
	STREAM<eth_header > & sts,
    STREAM<eth_notification> &notif_in,
    STREAM<eth_notification> &notif_out
) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE axis register both port=notif_in
#pragma HLS INTERFACE axis register both port=notif_out
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS PIPELINE II=1 style=flp
	
	unsigned constexpr bytes_per_word = DATA_WIDTH/8;
	unsigned const max_dma_bytes = 8*1024*1024-1;//TODO: convert to argument?

	static unsigned int remaining[1024] = {0};//one RAMB36 for max 1024 sessions
	static ap_uint<DEST_WIDTH> target_strm[1024] = {0};//one RAMB18 for max 1024 sessions

	static eth_notification notif;
	static bool continue_notif = false;
	static bool continue_message = false;

	static unsigned int message_rem = 0;
	static ap_uint<DEST_WIDTH> message_strm = 0;
	static unsigned int prev_session_id = 0;
	unsigned int current_bytes = 0;

	stream_word inword;
	eth_header hdr;
	eth_notification downstream_notif;

	if(STREAM_IS_EMPTY(notif_in) && STREAM_IS_EMPTY(in)) return;

	//get new notification unless we're continuing an old one or a message
	if(!continue_notif && !continue_message){
		notif = STREAM_READ(notif_in);
	}

#ifndef ACCL_SYNTHESIS
	std::stringstream ss;
	ss << "TCP Depacketizer: Processing incoming fragment count=" << notif.length << " for session " << notif.session_id << "\n";
	logger << log_level::verbose << ss.str();
#endif

	//get remaining message bytes, from local storage
	//TODO: cache latest accessed value
	if(prev_session_id != notif.session_id){
		message_rem = remaining[notif.session_id];
	}
	
	downstream_notif.session_id = notif.session_id;
	if(message_rem == 0){//if remaining bytes is zero, then this is the start of a new message
		//get header and some important info from it
		inword = STREAM_READ(in);
		hdr = eth_header(inword.data(HEADER_LENGTH-1,0));
		message_rem = hdr.count;//length of upcoming message (excluding the header itself)
		message_strm = hdr.strm;//target of message (0 is targeting memory so managed, everything else is  stream so unmanaged)
		if(message_strm == 0){
			//put notification, header in output streams
			STREAM_WRITE(sts, hdr);
			downstream_notif.type = 0; //for SOM
			downstream_notif.length = hdr.count;
			STREAM_WRITE(notif_out, downstream_notif);
		}
		//decrement the length to reflect the fact that we have removed the 64B header
		//Note: the rxHandler must make sure to not give us fragments less than 64B
		notif.length -= bytes_per_word;
		target_strm[notif.session_id] = message_strm;
	} else{//if remaining bytes is not zero, then this is a continuation of an old message
		message_strm = target_strm[notif.session_id];
	}
	//write out notifications
	//in case the fragment spans the end of the current message and beginning of another,
	//only notify for the part up to the end of the current message
	if(message_strm == 0){
		downstream_notif.type = 1; //for SOF
		downstream_notif.length = notif.length;
		STREAM_WRITE(notif_out, downstream_notif);
	}
	//copy data in -> out
	do{
		#pragma HLS PIPELINE II=1
		inword = STREAM_READ(in);
		inword.dest = message_strm;
		STREAM_WRITE(out, inword);
		notif.length = (notif.length < bytes_per_word) ? 0u : (unsigned int)notif.length-bytes_per_word;//floor at zero
		current_bytes += (message_rem < bytes_per_word) ? message_rem : bytes_per_word;
		message_rem = (message_rem < bytes_per_word) ? 0u : message_rem-bytes_per_word;//slight problem here if the message doesnt end on a 64B boundary...
	} while(notif.length > 0 && message_rem > 0 && current_bytes < (max_dma_bytes-bytes_per_word));
	if(message_strm == 0){
		downstream_notif.type = 2; //for EOF
		downstream_notif.length = current_bytes;
		STREAM_WRITE(notif_out, downstream_notif);
	}
	//update session info (remaining bytes and target of currently processing message)
	remaining[notif.session_id] = message_rem;
	//if we're not finished with this fragment, skip notification read on the next run
	continue_notif = (notif.length > 0);
	continue_message = (notif.length > 0) && (message_rem > 0);
	//update session id for caching
	prev_session_id = notif.session_id;
}