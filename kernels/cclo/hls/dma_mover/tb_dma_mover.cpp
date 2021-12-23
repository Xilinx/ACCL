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

#include "dma_mover.h"
#include <iostream>
#include <stdio.h>

using namespace hls;
using namespace hlslib;

// typedef struct {
//     //the following 5 fields can be compressed into a single opcode field
//     ap_uint<2> opcode;//from user/mb - 8b probably enough
//     ap_uint<3> which_dm;//from scenario
//     ap_uint<1> remote_flags;//from scenario
//     ap_uint<3> compression_flags;//from user - 4b
//     ap_uint<2> stream_flags;//from user - 2b

//     //count
//     unsigned int count;

//     //parameters required only if sending offchip
//     unsigned int comm_offset;

//     //parameters required for arithmetic
//     unsigned int arcfg_offset;
//     unsigned int func_id;

//     ap_uint<64> op0_addr;//from user; required in all scenarios except when source is stream
//     ap_uint<64> op1_addr;//from user; required only in arithmetic scenarios
//     ap_uint<64> res_addr;//from user; required in all scenarios except when destination is stream or offchip

//     unsigned int mpi_tag;//from user
//     unsigned int dst_rank;//from user
// } move_instruction;

int test_dmas(){
	//set up instruction to transfer locally (no packetizer involved)
	move_instruction insn;
	insn.opcode = MOVE_IMMEDIATE;
	insn.remote_flags = NO_REMOTE;
	insn.compression_flags = NO_COMPRESSION;
	insn.stream_flags = NO_STREAM;
	insn.which_dm = USE_OP0_DM | USE_OP1_DM | USE_RES_DM;
	insn.count = 100;
	insn.arcfg_offset = 0;
	insn.op0_addr = 0;
	insn.op1_addr = 1024;
	insn.res_addr = 2048;
	insn.func_id = 0;

	datapath_arith_config arcfg;
	arcfg.uncompressed_elem_bits = 32;
    arcfg.compressed_elem_bits = 64;
    arcfg.elem_ratio = 16;
    arcfg.compressor_tdest = 1;
    arcfg.decompressor_tdest = 2;
    arcfg.arith_nfunctions = 1;
    arcfg.arith_is_compressed = 0;
    arcfg.arith_tdest[0] = 0;

    //interfaces to processor
    stream<move_instruction> instruction;
    stream<ap_uint<32> > error;
    //interfaces to data movement engines
    stream<axi::Command<64, 23> > dma0_read_cmd;
    stream<axi::Command<64, 23> > dma1_read_cmd;
    stream<axi::Command<64, 23> > dma1_write_cmd;
    stream<axi::Status> dma0_read_sts;
    stream<axi::Status> dma1_read_sts;
    stream<axi::Status> dma1_write_sts;
    stream<eth_header> eth_cmd;
    stream<ap_uint<32> > eth_sts;
    //interfaces to segmenters in the routing path
    stream<segmenter_cmd> dma0_read_seg_cmd;
    stream<segmenter_cmd> dma1_read_seg_cmd;
    stream<segmenter_cmd> dma1_write_seg_cmd;
    stream<segmenter_cmd> krnl_in_seg_cmd;
    stream<segmenter_cmd> krnl_out_seg_cmd;
    stream<segmenter_cmd> arith_op_seg_cmd;
    stream<segmenter_cmd> arith_res_seg_cmd;
    stream<segmenter_cmd> clane0_op_seg_cmd;
    stream<segmenter_cmd> clane0_res_seg_cmd;
    stream<segmenter_cmd> clane1_op_seg_cmd;
    stream<segmenter_cmd> clane1_res_seg_cmd;
    stream<segmenter_cmd> clane2_op_seg_cmd;
    stream<segmenter_cmd> clane2_res_seg_cmd;
    stream<ap_uint<32> > krnl_out_seg_sts;

	instruction.write(insn);
	dma0_read_sts.write(0);
	dma1_read_sts.write(0);
	dma1_write_sts.write(0);

	dma_mover(
		(unsigned int *)(&arcfg), 1024,
		instruction,
		error,
		dma0_read_cmd,
		dma1_read_cmd,
		dma1_write_cmd,
		dma0_read_sts,
		dma1_read_sts,
		dma1_write_sts,
		eth_cmd,
		eth_sts,
		dma0_read_seg_cmd,
		dma1_read_seg_cmd,
		dma1_write_seg_cmd,
		krnl_in_seg_cmd,
		krnl_out_seg_cmd,
		arith_op_seg_cmd,
		arith_res_seg_cmd,
		clane0_op_seg_cmd,
		clane0_res_seg_cmd,
		clane1_op_seg_cmd,
		clane1_res_seg_cmd,
		clane2_op_seg_cmd,
		clane2_res_seg_cmd,
		krnl_out_seg_sts
	);

	if(error.read() != NO_ERROR){
		return 1;
	} else{
		return 0;
	}
}

int main(){
    int nerrors  = 0;
    nerrors += test_dmas();
    return (nerrors == 0 ? 0 : 1);
}
