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

#define CYT_OFFS_BITS 6
#define CYT_VADDR_BITS 48
#define CYT_LEN_BITS 28
#define CYT_DEST_BITS 4
#define CYT_PID_BITS 6
#define CYT_STRM_BITS 2
#define CYT_OPCODE_BITS 5
#define CYT_REQ_RSRVD_BITS (128 - CYT_OFFS_BITS - 2 - CYT_VADDR_BITS - CYT_LEN_BITS - 1 - 2 * CYT_DEST_BITS - CYT_PID_BITS - 3 - CYT_STRM_BITS - CYT_OPCODE_BITS)
#define CYT_ACK_RSRVD_BITS (32 - CYT_OPCODE_BITS - CYT_STRM_BITS - 2 - CYT_DEST_BITS - CYT_PID_BITS - CYT_DEST_BITS)

// Coyote RDMA Opcode
#define CYT_RDMA_READ    0
#define CYT_RDMA_WRITE   1
#define CYT_RDMA_SEND    2
#define CYT_RDMA_IMMED   3

// Coyote STRM Opcode
#define CYT_STRM_CARD 0
#define CYT_STRM_HOST 1
#define CYT_STRM_RDMA 2
#define CYT_STRM_TCP 3

struct cyt_req_t{
    ap_uint<CYT_REQ_RSRVD_BITS> rsrvd;    // 19 bits
    ap_uint<CYT_OFFS_BITS> offs;       // 6 bits
    ap_uint<1> host;                   // 1 bit
    ap_uint<1> actv;                   // 1 bit

    ap_uint<CYT_LEN_BITS> len;         // 28 bits
    ap_uint<CYT_VADDR_BITS> vaddr;     // 48 bits

    ap_uint<1> last;                   // 1 bit

    ap_uint<CYT_DEST_BITS> dest;       // 4 bits
    ap_uint<CYT_PID_BITS> pid;         // 6 bits
    ap_uint<CYT_DEST_BITS> vfid;       // 4 bits
	
    ap_uint<1> remote;                 // 1 bit
    ap_uint<1> rdma;                   // 1 bit
    ap_uint<1> mode;                   // 1 bit
    ap_uint<CYT_STRM_BITS> strm;       // 2 bits
    ap_uint<CYT_OPCODE_BITS> opcode;   // 5 bits
    
    // Default constructor
    cyt_req_t()
        : rsrvd(0), offs(0), host(0), actv(0), len(0), vaddr(0), last(0),
          dest(0), pid(0), vfid(0), remote(0), rdma(0), mode(0), strm(0), opcode(0) {}

    // Parameterized constructor
    cyt_req_t(ap_uint<CYT_REQ_RSRVD_BITS> rsrvd_arg, ap_uint<CYT_OFFS_BITS> offs_arg, ap_uint<1> host_arg, ap_uint<1> actv_arg,
              ap_uint<CYT_LEN_BITS> len_arg, ap_uint<CYT_VADDR_BITS> vaddr_arg, ap_uint<1> last_arg,
              ap_uint<CYT_DEST_BITS> dest_arg, ap_uint<CYT_PID_BITS> pid_arg, ap_uint<CYT_DEST_BITS> vfid_arg,
              ap_uint<1> remote_arg, ap_uint<1> rdma_arg, ap_uint<1> mode_arg, ap_uint<CYT_STRM_BITS> strm_arg, ap_uint<CYT_OPCODE_BITS> opcode_arg)
        : rsrvd(rsrvd_arg), offs(offs_arg), host(host_arg), actv(actv_arg), len(len_arg), vaddr(vaddr_arg),
          last(last_arg), dest(dest_arg), pid(pid_arg), vfid(vfid_arg), remote(remote_arg), rdma(rdma_arg),
          mode(mode_arg), strm(strm_arg), opcode(opcode_arg) {}

    // Constructor from a single ap_uint<128> argument
    cyt_req_t(ap_uint<128> in) {
        rsrvd = in(CYT_REQ_RSRVD_BITS - 1, 0);
        offs = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS - 1, CYT_REQ_RSRVD_BITS);
        host = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS);
        actv = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 1);
        len = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 1 + CYT_LEN_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2);
        vaddr = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS);
        last = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS);
        dest = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + 1);
        pid = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS);
        vfid = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS);
        remote = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS);
        rdma = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 1);
        mode = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 2, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 2);
        strm = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 3 + CYT_STRM_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 3);
        opcode = in(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 3 + CYT_STRM_BITS + CYT_OPCODE_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 3 + CYT_STRM_BITS);
    }
	
    operator ap_uint<128>() {
        ap_uint<128> ret;

        // Assigning fields to the appropriate bit positions in the 128-bit return value.
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 3 + CYT_STRM_BITS + CYT_OPCODE_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 3 + CYT_STRM_BITS) = opcode; // opcode
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 3 + CYT_STRM_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 3) = strm; // strm
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 2, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 2) = mode; // mode
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS + 1) = rdma; // rdma
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS) = remote; // remote
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS + CYT_DEST_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS) = vfid; // vfid
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS + CYT_PID_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS) = pid; // pid
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + CYT_DEST_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS + 1) = dest; // dest
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS) = last; // last
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS + CYT_VADDR_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS) = vaddr; // vaddr
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2 + CYT_LEN_BITS - 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 2) = len; // len
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 1, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS + 1) = actv; // actv
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS, CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS) = host; // host
        ret(CYT_REQ_RSRVD_BITS + CYT_OFFS_BITS - 1, CYT_REQ_RSRVD_BITS) = offs; // offs
        ret(CYT_REQ_RSRVD_BITS - 1, 0) = rsrvd; // rsrvd, disregard

        return ret;
    }
};

struct cyt_ack_t {
    ap_uint<CYT_ACK_RSRVD_BITS> rsrvd; // 9 bits
    ap_uint<CYT_DEST_BITS> vfid;       // 4 bits
    ap_uint<CYT_PID_BITS> pid;         // 6 bits
    ap_uint<CYT_DEST_BITS> dest;       // 4 bits
    ap_uint<1> host;                   // 1 bit
    ap_uint<1> remote;                 // 1 bit
    ap_uint<CYT_STRM_BITS> strm;       // 2 bits
    ap_uint<CYT_OPCODE_BITS> opcode;   // 5 bits

    // Default constructor
    cyt_ack_t()
        : rsrvd(0), vfid(0), pid(0), dest(0), host(0), remote(0), strm(0), opcode(0) {}

    // Parameterized constructor
    cyt_ack_t(ap_uint<CYT_ACK_RSRVD_BITS> rsrvd_arg,
              ap_uint<CYT_DEST_BITS> vfid_arg,
              ap_uint<CYT_PID_BITS> pid_arg,
              ap_uint<CYT_DEST_BITS> dest_arg,
              ap_uint<1> host_arg,
              ap_uint<1> remote_arg,
              ap_uint<CYT_STRM_BITS> strm_arg,
              ap_uint<CYT_OPCODE_BITS> opcode_arg)
        : rsrvd(rsrvd_arg), vfid(vfid_arg), pid(pid_arg), dest(dest_arg),
          host(host_arg), remote(remote_arg), strm(strm_arg), opcode(opcode_arg) {}

    // Constructor from a single ap_uint<32> argument
    cyt_ack_t(ap_uint<32> in) {
        opcode = in(31, 31 - CYT_OPCODE_BITS + 1);
        strm = in(31 - CYT_OPCODE_BITS, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS + 1);
        remote = in(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 1, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 1);
        host = in(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 2, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 2);
        dest = in(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 3, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - CYT_DEST_BITS - 2);
        pid = in(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - CYT_DEST_BITS - 3, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - CYT_DEST_BITS - CYT_PID_BITS - 2);
        vfid = in(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - CYT_DEST_BITS - CYT_PID_BITS - 3, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 2 * CYT_DEST_BITS - CYT_PID_BITS - 2);
        rsrvd = in(CYT_ACK_RSRVD_BITS - 1, 0); // Remaining bits for reserved
    }

    // Conversion operator to ap_uint<32>
    operator ap_uint<32>() {
        ap_uint<32> ret;

        ret(31, 31 - CYT_OPCODE_BITS + 1) = opcode;
        ret(31 - CYT_OPCODE_BITS, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS + 1) = strm;
        ret(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 1, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 1) = remote;
        ret(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 2, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 2) = host;
        ret(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 3, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - CYT_DEST_BITS - 2) = dest;
        ret(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - CYT_DEST_BITS - 3, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - CYT_DEST_BITS - CYT_PID_BITS - 2) = pid;
        ret(31 - CYT_OPCODE_BITS - CYT_STRM_BITS - CYT_DEST_BITS - CYT_PID_BITS - 3, 31 - CYT_OPCODE_BITS - CYT_STRM_BITS - 2 * CYT_DEST_BITS - CYT_PID_BITS - 2) = vfid;
        ret(CYT_ACK_RSRVD_BITS - 1, 0) = rsrvd;

        return ret;
    }
};
