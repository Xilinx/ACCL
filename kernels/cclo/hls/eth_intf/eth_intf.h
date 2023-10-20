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


// Different message types
#define EGR_MSG       	0
#define RNDZVS_MSG		1	
#define RNDZVS_INIT     2
#define RNDZVS_WR_DONE 	3

// RDMA Opcode
#define RDMA_WRITE   1
#define RDMA_SEND    2

// rdma_req_t structs
#define RDMA_VADDR_BITS     64
#define RDMA_LEN_BITS       32
#define RDMA_QPN_BITS 		16
#define RDMA_OPCODE_BITS	8
#define RDMA_HOST_BITS      8
#define RDMA_REQ_BITS		RDMA_VADDR_BITS+RDMA_LEN_BITS+RDMA_HOST_BITS+RDMA_QPN_BITS+RDMA_OPCODE_BITS

struct rdma_req_t {
    ap_uint<RDMA_VADDR_BITS> vaddr; // only needed in WRITE // needs to be resolved in rdzv handshake
    ap_uint<RDMA_LEN_BITS> len;
    ap_uint<RDMA_HOST_BITS> host; // 0-device memory, 1-host memory; not used in SEND // needs to be resolved in rdzv handshake
    ap_uint<RDMA_QPN_BITS> qpn;
    ap_uint<RDMA_OPCODE_BITS> opcode;

    rdma_req_t() : vaddr(0), len(0), host(0), qpn(0), opcode(0) {}
    
    // Constructor for initializing all members
    rdma_req_t(ap_uint<RDMA_VADDR_BITS> vaddr, ap_uint<RDMA_LEN_BITS> len, ap_uint<RDMA_HOST_BITS> host, ap_uint<RDMA_QPN_BITS> qpn, ap_uint<RDMA_OPCODE_BITS> opcode)
        : vaddr(vaddr), len(len), host(host), qpn(qpn), opcode(opcode) {}

    // Constructor for converting an existing ap_uint<RDMA_REQ_BITS> to rdma_req_t
    rdma_req_t(ap_uint<RDMA_REQ_BITS> in) {
        vaddr = in(RDMA_VADDR_BITS - 1, 0);
        len = in(RDMA_VADDR_BITS + RDMA_LEN_BITS - 1, RDMA_VADDR_BITS);
        host = in(RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS - 1, RDMA_VADDR_BITS + RDMA_LEN_BITS);
        qpn = in(RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS + RDMA_QPN_BITS - 1, RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS);
        opcode = in(RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS + RDMA_QPN_BITS + RDMA_OPCODE_BITS - 1, RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS + RDMA_QPN_BITS);
    }

    // Conversion operator for converting rdma_req_t to ap_uint<RDMA_REQ_BITS>
    operator ap_uint<RDMA_REQ_BITS>() {
        ap_uint<RDMA_REQ_BITS> ret;
        ret(RDMA_VADDR_BITS - 1, 0) = vaddr;
        ret(RDMA_VADDR_BITS + RDMA_LEN_BITS - 1, RDMA_VADDR_BITS) = len;
        ret(RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS - 1, RDMA_VADDR_BITS + RDMA_LEN_BITS) = host;
        ret(RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS + RDMA_QPN_BITS - 1, RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS) = qpn;
        ret(RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS + RDMA_QPN_BITS + RDMA_OPCODE_BITS - 1, RDMA_VADDR_BITS + RDMA_LEN_BITS + RDMA_HOST_BITS + RDMA_QPN_BITS) = opcode;
        return ret;
    }
};


#define HEADER_COUNT_START  0
#define HEADER_COUNT_END    31
#define HEADER_TAG_START    HEADER_COUNT_END + 1
#define HEADER_TAG_END      HEADER_TAG_START + 31
#define HEADER_SRC_START    HEADER_TAG_END + 1
#define HEADER_SRC_END      HEADER_SRC_START + 31
#define HEADER_SEQ_START    HEADER_SRC_END + 1
#define HEADER_SEQ_END      HEADER_SEQ_START + 31
#define HEADER_STRM_START   HEADER_SEQ_END + 1
#define HEADER_STRM_END     HEADER_STRM_START + 31
#define HEADER_DST_START    HEADER_STRM_END + 1
#define HEADER_DST_END      HEADER_DST_START + 15
#define HEADER_MSG_TYPE_START   HEADER_DST_END + 1
#define HEADER_MSG_TYPE_END     HEADER_MSG_TYPE_START + 7
#define HEADER_HOST_START       HEADER_MSG_TYPE_END + 1
#define HEADER_HOST_END         HEADER_HOST_START + 7
#define HEADER_VADDR_START  HEADER_HOST_END + 1
#define HEADER_VADDR_END    HEADER_VADDR_START + RDMA_VADDR_BITS
#define HEADER_LENGTH       HEADER_VADDR_END + 1

struct eth_header {
    ap_uint<32> count;
    ap_uint<32> tag;
    ap_uint<32> src; // source rank number
    ap_uint<32> seqn;
    ap_uint<32> strm;
    ap_uint<16> dst; // encode either session or qpn; // provided by DMP
    ap_uint<8> msg_type; // message types, default EGR_MSG;
    ap_uint<8> host; // 0-device memory, 1-host memory; not used in SEND // needs to be resolved in rdzv handshake // provided by DMP
    ap_uint<RDMA_VADDR_BITS> vaddr; // virual address, not used in SEND // provided by DMP

    eth_header() : count(0), tag(0), src(0), seqn(0), strm(0), dst(0), msg_type(0), host(0), vaddr(0) {}

    eth_header(ap_uint<HEADER_LENGTH> in) :
        count(in(HEADER_COUNT_END, HEADER_COUNT_START)),
        tag(in(HEADER_TAG_END, HEADER_TAG_START)),
        src(in(HEADER_SRC_END, HEADER_SRC_START)),
        seqn(in(HEADER_SEQ_END, HEADER_SEQ_START)),
        strm(in(HEADER_STRM_END, HEADER_STRM_START)),
        dst(in(HEADER_DST_END, HEADER_DST_START)),
        msg_type(in(HEADER_MSG_TYPE_END, HEADER_MSG_TYPE_START)),
        host(in(HEADER_HOST_END, HEADER_HOST_START)),
        vaddr(in(HEADER_VADDR_END, HEADER_VADDR_START)) {}

    operator ap_uint<HEADER_LENGTH>() {
        ap_uint<HEADER_LENGTH> ret;
        ret(HEADER_COUNT_END, HEADER_COUNT_START) = count;
        ret(HEADER_TAG_END, HEADER_TAG_START) = tag;
        ret(HEADER_SRC_END, HEADER_SRC_START) = src;
        ret(HEADER_SEQ_END, HEADER_SEQ_START) = seqn;
        ret(HEADER_STRM_END, HEADER_STRM_START) = strm;
        ret(HEADER_DST_END, HEADER_DST_START) = dst;
        ret(HEADER_MSG_TYPE_END, HEADER_MSG_TYPE_START) = msg_type;
        ret(HEADER_HOST_END, HEADER_HOST_START) = host;
        ret(HEADER_VADDR_END, HEADER_VADDR_START) = vaddr;
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

void rdma_depacketizer(
	STREAM<stream_word > & in,
	STREAM<stream_word > & out,
	STREAM<eth_header > & sts,
    STREAM<eth_notification> &notif_in,
    STREAM<eth_notification> &notif_out,
	STREAM<ap_axiu<32,0,0,0> > & ub_notif_out
);

void rdma_packetizer(
	STREAM<stream_word > & in,
	STREAM<stream_word > & out,
	STREAM<eth_header> & cmd,
	STREAM<ap_uint<32> > & sts,
	unsigned int max_pktsize
);

void rdma_sq_handler(
	STREAM<rdma_req_t> & rdma_sq,
    STREAM<ap_axiu<32,0,0,0> > & ub_sq,
	STREAM<eth_header> & cmd_in,
	STREAM<eth_header> & cmd_out
);
