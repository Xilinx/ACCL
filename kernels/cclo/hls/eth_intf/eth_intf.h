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

#include "accl_hls.h"

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


#define EGR_PROTOC      0
#define RNDZVS_PROTOC   1

// Different param message types
#define PARAMS_EGR_HEADER       0
#define PARAMS_RNDZVS_INIT      1
#define PARAMS_RNDZVS_WR_DONE   2

// Coyote RDMA Opcode
#define RDMA_READ    0
#define RDMA_WRITE   1
#define RDMA_SEND    2
#define RDMA_IMMED   3

// Coyote rdma_req_t structs
#define RDMA_OPCODE_BITS	5
#define RDMA_MSG_BITS 		448
#define RDMA_OFFS_BITS 	 	4
#define RDMA_QPN_BITS 		10
#define RDMA_MSN_BITS 		24
#define RDMA_RSRVD_BITS     46
#define RDMA_REQ_BITS 	 	RDMA_RSRVD_BITS+RDMA_MSG_BITS+RDMA_OFFS_BITS+RDMA_MSN_BITS+4+RDMA_QPN_BITS+RDMA_OPCODE_BITS

#define RDMA_VADDR_BITS     64
#define RDMA_LEN_BITS       32
#define RDMA_PARAMS_BITS    288

struct rdma_req_t{
    ap_uint<RDMA_RSRVD_BITS> rsrvd;
    ap_uint<RDMA_MSG_BITS> msg;
    ap_uint<RDMA_OFFS_BITS> offs;
    ap_uint<RDMA_MSN_BITS> ssn;
    ap_uint<1> cmplt;
    ap_uint<1> last;
    ap_uint<1> mode;
    ap_uint<1> host;
    ap_uint<RDMA_QPN_BITS> qpn;
    ap_uint<RDMA_OPCODE_BITS> opcode;

    rdma_req_t() : rsrvd(0), msg(0), offs(0), ssn(0), cmplt(0), last(0), mode(0), host(0), qpn(0), opcode(0) {}
    rdma_req_t(ap_uint<RDMA_REQ_BITS> in) {
        rsrvd = in(RDMA_RSRVD_BITS - 1, 0);
        msg = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS - 1, RDMA_RSRVD_BITS);
        offs = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS - 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS);
        ssn = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS - 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS);
        cmplt = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS);
        last = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 1);
        mode = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 2, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 2);
        host = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 3, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 3);
        qpn = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 4 + RDMA_QPN_BITS - 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 4);
        opcode = in(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 4 + RDMA_QPN_BITS + RDMA_OPCODE_BITS - 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 4 + RDMA_QPN_BITS);
    }
    operator ap_uint<RDMA_REQ_BITS>() {
        ap_uint<RDMA_REQ_BITS> ret;
        ret(RDMA_RSRVD_BITS - 1, 0) = rsrvd;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS - 1, RDMA_RSRVD_BITS) = msg;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS - 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS) = offs;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS - 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS) = ssn;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS) = cmplt;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 1) = last;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 2, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 2) = mode;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 3, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 3) = host;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 4 + RDMA_QPN_BITS - 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 4) = qpn;
        ret(RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 4 + RDMA_QPN_BITS + RDMA_OPCODE_BITS - 1, RDMA_RSRVD_BITS + RDMA_MSG_BITS + RDMA_OFFS_BITS + RDMA_MSN_BITS + 4 + RDMA_QPN_BITS) = opcode;
        return ret;
    }
} ;


struct rdma_req_msg_t{
    ap_uint<RDMA_VADDR_BITS> lvaddr;
    ap_uint<RDMA_VADDR_BITS> rvaddr;
    ap_uint<RDMA_LEN_BITS> len;
    ap_uint<RDMA_PARAMS_BITS> params;
    
    rdma_req_msg_t() : lvaddr(0), rvaddr(0), len(0), params(0) {}
    rdma_req_msg_t(ap_uint<RDMA_MSG_BITS> in) {
        lvaddr = in(RDMA_VADDR_BITS - 1, 0);
        rvaddr = in(2 * RDMA_VADDR_BITS - 1, RDMA_VADDR_BITS);
        len = in(2 * RDMA_VADDR_BITS + RDMA_LEN_BITS - 1, 2 * RDMA_VADDR_BITS);
        params = in(RDMA_MSG_BITS - 1, 2 * RDMA_VADDR_BITS + RDMA_LEN_BITS);
    }
    operator ap_uint<RDMA_MSG_BITS>() {
        ap_uint<RDMA_MSG_BITS> ret;
        ret(RDMA_VADDR_BITS - 1, 0) = lvaddr;
        ret(2 * RDMA_VADDR_BITS - 1, RDMA_VADDR_BITS) = rvaddr;
        ret(2 * RDMA_VADDR_BITS + RDMA_LEN_BITS - 1, 2 * RDMA_VADDR_BITS) = len;
        ret(RDMA_MSG_BITS - 1, 2 * RDMA_VADDR_BITS + RDMA_LEN_BITS) = params;
        return ret;
    }
};


typedef struct{
	ap_uint<16> dst;
	ap_uint<32> seqn;
} rdma_meta_req_t;

typedef struct{
    ap_uint<1> hit;
    ap_uint<RDMA_VADDR_BITS> rvaddr;
} rdma_meta_rsp_t;

typedef struct{
	ap_uint<16> dst;
	ap_uint<32> seqn;
    ap_uint<RDMA_VADDR_BITS> rvaddr;
} rdma_meta_upd_t;


#define HEADER_COUNT_START  0
#define HEADER_COUNT_END    31
#define HEADER_TAG_START    HEADER_COUNT_END+1
#define HEADER_TAG_END	    HEADER_TAG_START+31
#define HEADER_SRC_START    HEADER_TAG_END+1
#define HEADER_SRC_END	    HEADER_SRC_START+31
#define HEADER_SEQ_START    HEADER_SRC_END+1
#define HEADER_SEQ_END	    HEADER_SEQ_START+31
#define HEADER_STRM_START   HEADER_SEQ_END+1
#define HEADER_STRM_END	    HEADER_STRM_START+31
#define HEADER_DST_START    HEADER_STRM_END+1
#define HEADER_DST_END	    HEADER_DST_START+15
#define HEADER_PROTOC_START HEADER_DST_END+1
#define HEADER_PROTOC_END   HEADER_PROTOC_START+15
#define HEADER_RVADDR_START HEADER_PROTOC_END+1
#define HEADER_RVADDR_END   HEADER_RVADDR_START+RDMA_VADDR_BITS
#define HEADER_LENGTH       HEADER_RVADDR_END+1

struct eth_header{
	ap_uint<32> count;
	ap_uint<32> tag;
	ap_uint<32> src;
	ap_uint<32> seqn;
	ap_uint<32> strm;
	ap_uint<16> dst;
    ap_uint<16> protoc;
    ap_uint<RDMA_VADDR_BITS> rvaddr;
	eth_header() : count(0), tag(0), src(0), seqn(0), strm(0), dst(0), protoc(0), rvaddr() {}
	eth_header(ap_uint<HEADER_LENGTH> in) : 
		count(in(HEADER_COUNT_END, HEADER_COUNT_START)),
		tag(in(HEADER_TAG_END, HEADER_TAG_START)),
		src(in(HEADER_SRC_END, HEADER_SRC_START)),
		seqn(in(HEADER_SEQ_END, HEADER_SEQ_START)),
		strm(in(HEADER_STRM_END, HEADER_STRM_START)),
		dst(in(HEADER_DST_END, HEADER_DST_START)),
        protoc(in(HEADER_PROTOC_END, HEADER_PROTOC_START)),
        rvaddr(in(HEADER_RVADDR_END, HEADER_RVADDR_START)) {}
	operator ap_uint<HEADER_LENGTH>(){
		ap_uint<HEADER_LENGTH> ret;
		ret(HEADER_COUNT_END, HEADER_COUNT_START) = count;
        ret(HEADER_TAG_END, HEADER_TAG_START) = tag;
        ret(HEADER_SRC_END, HEADER_SRC_START) = src;
        ret(HEADER_SEQ_END, HEADER_SEQ_START) = seqn;
        ret(HEADER_STRM_END, HEADER_STRM_START) = strm;
		ret(HEADER_DST_END, HEADER_DST_START) = dst;
        ret(HEADER_PROTOC_END, HEADER_PROTOC_START) = protoc;
        ret(HEADER_RVADDR_END, HEADER_RVADDR_START) = rvaddr;
		return ret;
	}
};

typedef struct{
	ap_uint<2>  type; //0 = Start of Message (SOM); 1 = Start of Fragment (SOF); 2 = End of Fragment (EOF); 3 = Reserved
	ap_uint<16> session_id; //all types set session ID
	ap_uint<23> length; //applicable to SOM and EOF; SOF does not set length
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
	STREAM<eth_header> & sts,
	STREAM<eth_notification> &notif_out
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
    STREAM<pkt16>& m_axis_tcp_close_connection,
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
