/*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
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
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "accl_hls.h"

using namespace std;

#define CYT_VADDR_BITS 48
#define CYT_LEN_BITS 28
#define CYT_DEST_BITS 4
#define CYT_PID_BITS 6
#define CYT_N_REGIONS_BITS 1
#define CYT_RSRVD_BITS 96-4-CYT_N_REGIONS_BITS-CYT_VADDR_BITS-CYT_LEN_BITS-CYT_DEST_BITS-CYT_PID_BITS

struct cyt_req_t{
    ap_uint<CYT_RSRVD_BITS> rsrvd;
	ap_uint<CYT_N_REGIONS_BITS> vfid;
    ap_uint<CYT_PID_BITS> pid;
    ap_uint<CYT_DEST_BITS> dest;
	ap_uint<1> host;
    ap_uint<1> ctl;
	ap_uint<1> sync;
	ap_uint<1> stream;
    ap_uint<CYT_LEN_BITS> len;
    ap_uint<CYT_VADDR_BITS> vaddr;

    cyt_req_t() : rsrvd(0), vfid(0), pid(0), dest(0), host(0), ctl(0), sync(0), stream(0), len(0), vaddr(0) {}

	cyt_req_t(ap_uint<CYT_RSRVD_BITS> rsrvd_arg, ap_uint<CYT_N_REGIONS_BITS> vfid_arg, ap_uint<CYT_PID_BITS> pid_arg,
          ap_uint<CYT_DEST_BITS> dest_arg, ap_uint<1> host_arg, ap_uint<1> ctl_arg, ap_uint<1> sync_arg,
          ap_uint<1> stream_arg, ap_uint<CYT_LEN_BITS> len_arg, ap_uint<CYT_VADDR_BITS> vaddr_arg)
        : rsrvd(rsrvd_arg),
          vfid(vfid_arg),
          pid(pid_arg),
          dest(dest_arg),
          host(host_arg),
          ctl(ctl_arg),
          sync(sync_arg),
          stream(stream_arg),
          len(len_arg),
          vaddr(vaddr_arg) {}

    cyt_req_t(ap_uint<96> in) {
        rsrvd = in(CYT_RSRVD_BITS - 1, 0);
        vfid = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS-1, CYT_RSRVD_BITS);
        pid = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS-1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS);
        dest = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS-1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS);
		host = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS);
        ctl = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+1);
		sync = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+2,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+2);
		stream = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+3,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+3);
        len = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+4+CYT_LEN_BITS-1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+4);
        vaddr = in(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+4+CYT_LEN_BITS+CYT_VADDR_BITS-1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+4+CYT_LEN_BITS);
    }
	
    operator ap_uint<96>() {
        ap_uint<96> ret;
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+4+CYT_LEN_BITS+CYT_VADDR_BITS-1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+4+CYT_LEN_BITS) = vaddr; //vaddr
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+4+CYT_LEN_BITS-1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+4) = len; //len
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+3,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+3) = stream; //stream
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+2,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+2) = sync; //sync
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS+1) = ctl; //ctl
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS) = host; //host
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS+CYT_DEST_BITS-1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS) = dest; //dest
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS+CYT_PID_BITS-1,CYT_RSRVD_BITS+CYT_N_REGIONS_BITS) = pid; //pid
		ret(CYT_RSRVD_BITS+CYT_N_REGIONS_BITS-1,CYT_RSRVD_BITS) = vfid; //vfid
		ret(CYT_RSRVD_BITS-1,0) = rsrvd; //rsrvd, disregard
        return ret;
    }
};


// Coyote RDMA Opcode
#define CYT_RDMA_READ    0
#define CYT_RDMA_WRITE   1
#define CYT_RDMA_SEND    2
#define CYT_RDMA_IMMED   3

// Coyote cyt_rdma_req_t structs
#define CYT_RDMA_OPCODE_BITS	5
#define CYT_RDMA_MSG_BITS 		448
#define CYT_RDMA_OFFS_BITS 	 	4
#define CYT_RDMA_QPN_BITS 		10
#define CYT_RDMA_MSN_BITS 		24
#define CYT_RDMA_RSRVD_BITS     17
#define CYT_RDMA_REQ_BITS 	 	CYT_RDMA_RSRVD_BITS+CYT_RDMA_MSG_BITS+CYT_RDMA_OFFS_BITS+CYT_RDMA_MSN_BITS+4+CYT_RDMA_QPN_BITS+CYT_RDMA_OPCODE_BITS

#define CYT_RDMA_VADDR_BITS     64
#define CYT_RDMA_LEN_BITS       32
#define CYT_RDMA_PARAMS_BITS    288

struct cyt_rdma_req_t{
    ap_uint<CYT_RDMA_RSRVD_BITS> rsrvd;
    ap_uint<CYT_RDMA_MSG_BITS> msg;
    ap_uint<CYT_RDMA_OFFS_BITS> offs;
    ap_uint<CYT_RDMA_MSN_BITS> ssn;
    ap_uint<1> cmplt;
    ap_uint<1> last;
    ap_uint<1> mode;
    ap_uint<1> host;
    ap_uint<CYT_RDMA_QPN_BITS> qpn;
    ap_uint<CYT_RDMA_OPCODE_BITS> opcode;

    cyt_rdma_req_t() : rsrvd(0), msg(0), offs(0), ssn(0), cmplt(0), last(0), mode(0), host(0), qpn(0), opcode(0) {}
    cyt_rdma_req_t(ap_uint<CYT_RDMA_REQ_BITS> in) {
        rsrvd = in(CYT_RDMA_RSRVD_BITS - 1, 0);
        msg = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS - 1, CYT_RDMA_RSRVD_BITS);
        offs = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS - 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS);
        ssn = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS - 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS);
        cmplt = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS);
        last = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 1);
        mode = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 2, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 2);
        host = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 3, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 3);
        qpn = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 4 + CYT_RDMA_QPN_BITS - 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 4);
        opcode = in(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 4 + CYT_RDMA_QPN_BITS + CYT_RDMA_OPCODE_BITS - 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 4 + CYT_RDMA_QPN_BITS);
    }
    operator ap_uint<CYT_RDMA_REQ_BITS>() {
        ap_uint<CYT_RDMA_REQ_BITS> ret;
        ret(CYT_RDMA_RSRVD_BITS - 1, 0) = rsrvd;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS - 1, CYT_RDMA_RSRVD_BITS) = msg;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS - 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS) = offs;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS - 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS) = ssn;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS) = cmplt;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 1) = last;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 2, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 2) = mode;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 3, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 3) = host;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 4 + CYT_RDMA_QPN_BITS - 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 4) = qpn;
        ret(CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 4 + CYT_RDMA_QPN_BITS + CYT_RDMA_OPCODE_BITS - 1, CYT_RDMA_RSRVD_BITS + CYT_RDMA_MSG_BITS + CYT_RDMA_OFFS_BITS + CYT_RDMA_MSN_BITS + 4 + CYT_RDMA_QPN_BITS) = opcode;
        return ret;
    }
} ;


struct cyt_rdma_req_msg_t{
    ap_uint<CYT_RDMA_VADDR_BITS> lvaddr;
    ap_uint<CYT_RDMA_VADDR_BITS> rvaddr;
    ap_uint<CYT_RDMA_LEN_BITS> len;
    ap_uint<CYT_RDMA_PARAMS_BITS> params;
    
    cyt_rdma_req_msg_t() : lvaddr(0), rvaddr(0), len(0), params(0) {}
    cyt_rdma_req_msg_t(ap_uint<CYT_RDMA_MSG_BITS> in) {
        lvaddr = in(CYT_RDMA_VADDR_BITS - 1, 0);
        rvaddr = in(2 * CYT_RDMA_VADDR_BITS - 1, CYT_RDMA_VADDR_BITS);
        len = in(2 * CYT_RDMA_VADDR_BITS + CYT_RDMA_LEN_BITS - 1, 2 * CYT_RDMA_VADDR_BITS);
        params = in(CYT_RDMA_MSG_BITS - 1, 2 * CYT_RDMA_VADDR_BITS + CYT_RDMA_LEN_BITS);
    }
    operator ap_uint<CYT_RDMA_MSG_BITS>() {
        ap_uint<CYT_RDMA_MSG_BITS> ret;
        ret(CYT_RDMA_VADDR_BITS - 1, 0) = lvaddr;
        ret(2 * CYT_RDMA_VADDR_BITS - 1, CYT_RDMA_VADDR_BITS) = rvaddr;
        ret(2 * CYT_RDMA_VADDR_BITS + CYT_RDMA_LEN_BITS - 1, 2 * CYT_RDMA_VADDR_BITS) = len;
        ret(CYT_RDMA_MSG_BITS - 1, 2 * CYT_RDMA_VADDR_BITS + CYT_RDMA_LEN_BITS) = params;
        return ret;
    }
};



void cyt_dma_adapter(
	//DM command streams
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma0_s2mm_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma1_s2mm_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma0_mm2s_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma1_mm2s_cmd,
	//DM status streams
	hls::stream<ap_axiu<32,0,0,0>> &dma0_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma0_mm2s_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_mm2s_sts,
#ifdef ACCL_RDMA
	//RDMA rd_req and wr_req
	hls::stream<ap_uint<96>> & rdma_wr_req,
	hls::stream<ap_uint<96>> & rdma_rd_req,
#endif
	//Coyote Bypass interface command and status
	hls::stream<ap_uint<96>> &cyt_byp_wr_cmd,
	hls::stream<ap_uint<16>> &cyt_byp_wr_sts,
	hls::stream<ap_uint<96>> &cyt_byp_rd_cmd,
	hls::stream<ap_uint<16>> &cyt_byp_rd_sts
);