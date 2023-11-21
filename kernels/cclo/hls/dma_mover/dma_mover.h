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

#include "streamdefines.h"
#include "accl_hls.h"
#include "ap_int.h"
#include "eth_intf.h"
#include "stream_segmenter.h"
#include "rxbuf_offload.h"
#include "ccl_offload_control.h"

typedef struct {
    //13+4 bits indicating what we're doing
    ap_uint<3> op0_opcode;
    ap_uint<3> op1_opcode;
    ap_uint<3> res_opcode;
    bool res_is_remote;
    bool res_is_rendezvous;
    bool op0_is_compressed;
    bool op1_is_compressed;
    bool res_is_compressed;

    bool op0_is_host;
    bool op1_is_host;
    bool res_is_host;

    ap_uint<4> func_id;//up to 16 functions

    //count
    unsigned int count;

    //parameters required only if sending offchip
    unsigned int comm_offset;
    //parameters required for arithmetic
    unsigned int arcfg_offset;

    ap_uint<64> op0_addr;
    ap_uint<64> op1_addr;
    ap_uint<64> res_addr;

    int op0_stride;
    int op1_stride;
    int res_stride;

    unsigned int rx_tag;
    unsigned int rx_src;

    unsigned int mpi_tag;//required only on remote result
    unsigned int dst_rank;//required only on remote result
} move_instruction;

typedef struct{
    bool check_dma0_rx;
    bool check_dma1_rx;
    bool check_dma1_tx;
    bool check_eth_tx;
    bool check_strm_tx;
    bool release_rxbuf;
    ap_uint<32> release_count = 0;
} move_ack_instruction;

typedef struct{
    unsigned int total_bytes;
    ap_uint<64> addr;
    ap_uint<DEST_WIDTH> mem_id;
    bool last = true;
} datamover_instruction;

typedef struct{
    unsigned int ncommands;
    bool last = true;
} datamover_ack_instruction;

typedef struct{
    unsigned int c_nwords;
    unsigned int u_nwords;
    bool stream_in;
    bool use_secondary_op_lane;
    bool stream_out;
    bool eth_out;
    bool op0_compressed;
    bool op1_compressed;
    bool res_compressed;
    bool use_arithmetic;
    bool arith_compressed;
    ap_uint<16> arith_func;
    ap_uint<16> compress_func;
    ap_uint<16> decompress_func;
    ap_uint<16> krnl_id;
} router_instruction;

typedef struct{
    unsigned int dst_sess_id;
    unsigned int src_rank;
    unsigned int seqn;
    unsigned int len;
    unsigned int mpi_tag;
    unsigned int max_seg_len;
    bool to_stream;
    bool to_host;
    bool rendezvous;
    ap_uint<64> addr;
} packetizer_instruction;

typedef struct{
    unsigned int expected_seqn;
    bool last = true;
} packetizer_ack_instruction;

void dma_mover(
    //interfaces to processor
    unsigned int * exchange_mem,
    STREAM<ap_axiu<32,0,0,0> > &command,
    STREAM<ap_axiu<32,0,0,0> > &error,
    //interfaces to rx buffer seek offload
    STREAM<rxbuf_signature> &rxbuf_req,
    STREAM<rxbuf_seek_result> &rxbuf_ack,
    STREAM<ap_uint<32> > &rxbuf_release_req,
    //interfaces to data movement engines
    STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &dma0_read_cmd,
    STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &dma1_read_cmd,
    STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &dma1_write_cmd,
    STREAM<ap_uint<32> > &dma0_read_sts,
    STREAM<ap_uint<32> > &dma1_read_sts,
    STREAM<ap_uint<32> > &dma1_write_sts,
    STREAM<eth_header> &eth_cmd,
    STREAM<ap_uint<32> > &eth_sts,
    //interfaces to segmenters in the routing path
    STREAM<segmenter_cmd> &dma0_read_seg_cmd,
    STREAM<segmenter_cmd> &dma1_read_seg_cmd,
    STREAM<segmenter_cmd> &krnl_in_seg_cmd,
    STREAM<segmenter_cmd> &krnl_out_seg_cmd,
    STREAM<segmenter_cmd> &arith_op0_seg_cmd,
    STREAM<segmenter_cmd> &arith_op1_seg_cmd,
    STREAM<segmenter_cmd> &arith_res_seg_cmd,
    STREAM<segmenter_cmd> &clane0_op_seg_cmd,
    STREAM<segmenter_cmd> &clane0_res_seg_cmd,
    STREAM<segmenter_cmd> &clane1_op_seg_cmd,
    STREAM<segmenter_cmd> &clane1_res_seg_cmd,
    STREAM<segmenter_cmd> &clane2_op_seg_cmd,
    STREAM<segmenter_cmd> &clane2_res_seg_cmd,
    STREAM<ap_uint<32> >  &krnl_out_seg_sts
);