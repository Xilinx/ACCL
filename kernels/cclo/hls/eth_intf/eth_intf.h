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

#pragma once

#include "ap_int.h"
#include "streamdefines.h"

#define DWIDTH512 512
#define DWIDTH256 256
#define DWIDTH128 128
#define DWIDTH64 64
#define DWIDTH32 32
#define DWIDTH16 16
#define DWIDTH8 8

#define RATE_CONTROL

typedef ap_axiu<DWIDTH512, 0, 0, 0> pkt512;
typedef ap_axiu<DWIDTH256, 0, 0, 0> pkt256;
typedef ap_axiu<DWIDTH128, 0, 0, 0> pkt128;
typedef ap_axiu<DWIDTH64, 0, 0, 0> pkt64;
typedef ap_axiu<DWIDTH32, 0, 0, 0> pkt32;
typedef ap_axiu<DWIDTH16, 0, 0, 0> pkt16;
typedef ap_axiu<DWIDTH8, 0, 0, 0> pkt8;

#define HEADER_COUNT_START 0
#define HEADER_COUNT_END   31
#define HEADER_TAG_START   HEADER_COUNT_END+1
#define HEADER_TAG_END	   HEADER_TAG_START+31
#define HEADER_SRC_START   HEADER_TAG_END+1
#define HEADER_SRC_END	   HEADER_SRC_START+31
#define HEADER_SEQ_START   HEADER_SRC_END+1
#define HEADER_SEQ_END	   HEADER_SEQ_START+31
#define HEADER_STRM_START  HEADER_SEQ_END+1
#define HEADER_STRM_END	   HEADER_STRM_START+31
#define HEADER_DST_START   HEADER_STRM_END+1
#define HEADER_DST_END	   HEADER_DST_START+15
#define HEADER_LENGTH      HEADER_DST_END+1

struct eth_header{
	ap_uint<32> count;
	ap_uint<32> tag;
	ap_uint<32> src;
	ap_uint<32> seqn;
	ap_uint<32> strm;
	ap_uint<16> dst;
	eth_header() : count(0), tag(0), src(0), seqn(0), strm(0), dst(0) {}
	eth_header(ap_uint<HEADER_LENGTH> in) : 
		count(in(HEADER_COUNT_END, HEADER_COUNT_START)),
		tag(in(HEADER_TAG_END, HEADER_TAG_START)),
		src(in(HEADER_SRC_END, HEADER_SRC_START)),
		seqn(in(HEADER_SEQ_END, HEADER_SEQ_START)),
		strm(in(HEADER_STRM_END, HEADER_STRM_START)),
		dst(in(HEADER_DST_END, HEADER_DST_START)) {}
	operator ap_uint<HEADER_LENGTH>(){
		ap_uint<HEADER_LENGTH> ret;
		ret(HEADER_COUNT_END, HEADER_COUNT_START) = count;
        ret(HEADER_TAG_END, HEADER_TAG_START) = tag;
        ret(HEADER_SRC_END, HEADER_SRC_START) = src;
        ret(HEADER_SEQ_END, HEADER_SEQ_START) = seqn;
        ret(HEADER_STRM_END, HEADER_STRM_START) = strm;
		ret(HEADER_DST_END, HEADER_DST_START) = dst;
		return ret;
	}
};

typedef struct{
	ap_uint<16> session_id;
	ap_uint<16> length;
} eth_notification;

void udp_packetizer(
	STREAM<stream_word > & in,
	STREAM<stream_word > & out,
	STREAM<eth_header> & cmd,
	STREAM<ap_uint<32> > & sts,
	unsigned int max_pktsize
);

void tcp_packetizer(
	STREAM<stream_word > & in,
	STREAM<stream_word > & out,
	STREAM<eth_header> & cmd,
	STREAM<ap_uint<96> > & cmd_txHandler,
	STREAM<ap_uint<32> > & sts,
	unsigned int max_pktsize
);

void udp_depacketizer(
	STREAM<stream_word > & in,
	STREAM<stream_word > & out,
	STREAM<eth_header> & sts
);

void tcp_depacketizer(
	STREAM<stream_word > & in,
	STREAM<stream_word > & out,
	STREAM<eth_header> & sts,
    STREAM<eth_notification> &notif_in,
    STREAM<eth_notification> &notif_out
);

void tcp_sessionHandler(
    STREAM<ap_axiu<32,0,0,0> > & port_cmd,
    STREAM<ap_axiu<32,0,0,0> > & port_sts,
	STREAM<ap_axiu<32,0,0,0> > & con_cmd,
    STREAM<ap_axiu<32,0,0,0> > & con_sts,
    STREAM<pkt16>& m_axis_tcp_listen_port, 
    STREAM<pkt8>& s_axis_tcp_port_status,
	STREAM<pkt64>& m_axis_tcp_open_connection,
    STREAM<pkt128>& s_axis_tcp_open_status
);

void tcp_txHandler(
    STREAM<stream_word>& s_data_in,
    STREAM<ap_uint<96> >& cmd_txHandler,
    STREAM<pkt32>& m_axis_tcp_tx_meta, 
    STREAM<stream_word>& m_axis_tcp_tx_data, 
    STREAM<pkt64>& s_axis_tcp_tx_status
);

void tcp_rxHandler(   
    STREAM<pkt128>& s_axis_tcp_notification, 
    STREAM<pkt32>& m_axis_tcp_read_pkg,
    STREAM<pkt16>& s_axis_tcp_rx_meta, 
    STREAM<stream_word>& s_axis_tcp_rx_data,
    STREAM<stream_word>& m_data_out,
    STREAM<eth_notification>& m_notif_out
);
