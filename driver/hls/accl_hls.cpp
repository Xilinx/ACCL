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
#
*******************************************************************************/
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_utils.h"
#include "hls_collectives.h"
#include "hostctrl_in.h"

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

ap_uint<32> send_in(
    unsigned int comm,
    unsigned int len,
    unsigned int tag,
    unsigned int dst_rank,
    uint64_t buf_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return 

    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_SEND;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_ROOT_SRC_DST,START_ROOT_SRC_DST) = dst_rank;
    in_data.range(END_TAG,START_TAG) = tag;
    in_data.range(END_ADDR_A,START_ADDR_A) = buf_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> recv_in(
    unsigned int comm,
    unsigned int len,
    unsigned int tag,
    unsigned int src_rank,
    uint64_t buf_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return 
    
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_RECV;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_ROOT_SRC_DST,START_ROOT_SRC_DST) = src_rank;
    in_data.range(END_TAG,START_TAG) = tag;
    in_data.range(END_ADDR_A,START_ADDR_A) = buf_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> broadcast_in(
    unsigned int comm,
    unsigned int len,
    unsigned int src_rank,
    uint64_t buf_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_BCAST;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_ROOT_SRC_DST,START_ROOT_SRC_DST) = src_rank;
    in_data.range(END_ADDR_A,START_ADDR_A) = buf_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> scatter_in(
    unsigned int comm,
    unsigned int len,
    unsigned int src_rank,
    uint64_t src_buf_addr,
    uint64_t dst_buf_addr
)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_SCATTER;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_ROOT_SRC_DST,START_ROOT_SRC_DST) = src_rank;
    in_data.range(END_ADDR_A,START_ADDR_A) = src_buf_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = dst_buf_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> gather_in(
    unsigned int comm,
    unsigned int len,
    unsigned int root_rank,
    uint64_t src_buf_addr,
    uint64_t dst_buf_addr
){
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_GATHER;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_ROOT_SRC_DST,START_ROOT_SRC_DST) = root_rank;
    in_data.range(END_ADDR_A,START_ADDR_A) = src_buf_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = dst_buf_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> allgather_in(
    unsigned int comm,
    unsigned int len,
    uint64_t src_buf_addr,
    uint64_t dst_buf_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_ALLGATHER;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_ADDR_A,START_ADDR_A) = src_buf_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = dst_buf_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> reduce_in(
    unsigned int comm,
    unsigned int len,
    unsigned int function,
    unsigned int root_rank,
    uint64_t src_addr,
    uint64_t dst_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_REDUCE;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_ROOT_SRC_DST,START_ROOT_SRC_DST) = root_rank;
    in_data.range(END_FUNCTION,START_FUNCTION) = function;
    in_data.range(END_ADDR_A,START_ADDR_A) = src_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = dst_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> allreduce_in(
    unsigned int comm,
    unsigned int len,
    unsigned int function,
    uint64_t src_addr,
    uint64_t dst_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_ALLREDUCE;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_FUNCTION,START_FUNCTION) = function;
    in_data.range(END_ADDR_A,START_ADDR_A) = src_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = dst_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> config_in(unsigned int function)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_CONFIG;
    in_data.range(END_FUNCTION,START_FUNCTION) = function;
    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> accumulate_in(
    unsigned int len,
    unsigned int function,
    uint64_t op0_addr,
    uint64_t op1_addr)//OP1_ADDR = DST_ADDR
    // uint64_t dst_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_ACC;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_FUNCTION,START_FUNCTION) = function;
    in_data.range(END_ADDR_A,START_ADDR_A) = op0_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = op1_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> copy_in(
    unsigned int len,
    uint64_t src_addr,
    uint64_t dst_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_COPY;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_ADDR_A,START_ADDR_A) = src_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = dst_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> ext_kernel_stream_in(
    unsigned int len,
    uint64_t src_addr,
    uint64_t dst_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_EXT_STREAM_KRNL;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_ADDR_A,START_ADDR_A) = src_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = dst_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> reduce_ext_in(
    unsigned int len,
    uint64_t op1_addr,
    uint64_t op2_addr,
    uint64_t dst_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return    
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_EXT_REDUCE;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_ADDR_A,START_ADDR_A) = op1_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = op2_addr;
    in_data.range(END_ADDR_C,START_ADDR_C) = dst_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}

ap_uint<32> scatter_reduce_in(
    unsigned int comm,
    unsigned int len,
    unsigned int function,
    uint64_t src_addr,
    uint64_t dst_addr)
{
    #pragma HLS INTERFACE axis port=cmd_in
    #pragma HLS INTERFACE axis port=cmd_out
    #pragma HLS INTERFACE axis port=sts_in
    #pragma HLS INTERFACE axis port=sts_out
    #pragma HLS INTERFACE ap_ctrl_none  port=return    
    ap_uint<DATA_WIDTH> in_data = 0;
    in_data.range(END_SCENARIO,START_SCENARIO)  = ACCL_REDUCE_SCATTER;
    in_data.range(END_LEN,START_LEN) = len;
    in_data.range(END_COMM,START_COMM) = comm;
    in_data.range(END_FUNCTION,START_FUNCTION) = function;
    in_data.range(END_ADDR_A,START_ADDR_A) = src_addr;
    in_data.range(END_ADDR_B,START_ADDR_B) = dst_addr;

    stream <ap_uint<DATA_WIDTH> > cmd_in;
    cmd_in.write(in_data);
    stream <ap_uint<32> > cmd_out;
    stream <ap_uint<32> > sts_in;
    stream <ap_uint<32> > sts_out;
    hostctrl_in(cmd_in,cmd_out,sts_in,sts_out);
    return sts_out.read();
}