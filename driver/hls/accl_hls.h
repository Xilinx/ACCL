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
#pragma once

#include "ap_int.h"
#include "ap_utils.h"
#include "ap_axi_sdata.h"

#define DATA_WIDTH 512
#define DEST_WIDTH 8

typedef ap_axiu<DATA_WIDTH, 0, 0, DEST_WIDTH> stream_word;
typedef ap_axiu<32, 0, 0, 0> command_word;

//this is a work-around for hlslib streams not synthesizing
//with vitis hls. Instead, we test a macro definition to select
//between simulation behaviour (use hlslib streams) and
//synthesis behaviour (use hls streams)
//all code using these macros should make sure it doesnt use features
//that aren't supported by both types of streams. For example,
//hlslib stream depths and storage types can't be defined on declaration
#ifdef ACCL_SYNTHESIS
//use hls streams
#include "hls_stream.h"
#define STREAM hls::stream 
#define STREAM_IS_EMPTY(s) s.empty()
#define STREAM_IS_FULL(s) s.full()
#define STREAM_READ(s) s.read()
#define STREAM_WRITE(s, val) s.write(val)
#else
//use hlslib streams
#include "Stream.h"
// #include <sstream>
// #include <iostream>
#define STREAM hlslib::Stream 
#define STREAM_IS_EMPTY(s) s.IsEmpty()
#define STREAM_IS_FULL(s) s.IsFull()
#define STREAM_READ(s) s.Pop()
#define STREAM_WRITE(s, val) s.Push(val)
#endif

namespace accl_hls {

//Primitives
#define ACCL_COPY           1
#define ACCL_COMBINE        2
#define ACCL_SEND           3 
#define ACCL_RECV           4
//Collectives
#define ACCL_BCAST          5
#define ACCL_SCATTER        6
#define ACCL_GATHER         7
#define ACCL_REDUCE         8
#define ACCL_ALLGATHER      9
#define ACCL_ALLREDUCE      10
#define ACCL_REDUCE_SCATTER 11

/**
 * @brief Class encapsulating ACCL command streams
 * 
 */
class ACCLCommand{
    public:
        /**
         * @brief Construct a new ACCLCommand object
         * 
         * @param cmd Reference to command stream to CCLO
         * @param sts Reference to status stream to CCLO
         * @param comm_adr Communicator ID
         * @param dpcfg_adr Address of datapath configuration in CCLO memory
         * @param cflags Compression flags
         * @param sflags Stream flags
         */
        ACCLCommand(STREAM<command_word > &cmd, STREAM<command_word > &sts,
                    ap_uint<32> comm_adr, ap_uint<32> dpcfg_adr,
                    ap_uint<32> cflags, ap_uint<32> sflags) : 
                    cmd(cmd), sts(sts), comm_adr(comm_adr), dpcfg_adr(dpcfg_adr), cflags(cflags), sflags(sflags) {}

        /**
         * @brief Construct a new ACCLCommand object
         * 
         * @param cmd Reference to command stream to CCLO
         * @param sts Reference to status stream to CCLO
         */
        ACCLCommand(STREAM<command_word > &cmd, STREAM<command_word > &sts) : 
                    ACCLCommand(cmd, sts, 0, 0, 0, 0) {}

    protected:
        STREAM<command_word > &cmd;
        STREAM<command_word > &sts;
        ap_uint<32> comm_adr;
        ap_uint<32> dpcfg_adr;
        ap_uint<32> cflags;
        ap_uint<32> sflags;

    public:

        /**
         * @brief Launch an ACCL call
         * 
         * @param scenario Indicates type of call (see defines)
         * @param len Length of buffers involved in call, in elements (not bytes)
         * @param comm ID of communicator
         * @param root_src_dst Either root, source or destination rank, depending on scenario
         * @param function Function ID for reduction-type scenarios
         * @param msg_tag Message tag
         * @param datapath_cfg Address of datapath configuration structure
         * @param compression_flags Compression flags
         * @param stream_flags Stream flags
         * @param addra Address of first operand, or zero if not in use
         * @param addrb Address of second operand, or zero if not in use
         * @param addrc Address of result, or zero if not in use
         */
        void start_call(
            ap_uint<32> scenario,
            ap_uint<32> len,
            ap_uint<32> comm,
            ap_uint<32> root_src_dst,
            ap_uint<32> function,
            ap_uint<32> msg_tag,
            ap_uint<32> datapath_cfg,
            ap_uint<32> compression_flags,
            ap_uint<32> stream_flags,
            ap_uint<64> addra,
            ap_uint<64> addrb,
            ap_uint<64> addrc
        ){
            command_word tmp;
            tmp.keep = 0xf;
            io_section:{
                #pragma HLS protocol fixed
                tmp.data=scenario; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=len; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=comm; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=root_src_dst; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=function; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=msg_tag; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=datapath_cfg; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=compression_flags; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=stream_flags; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addra(31,0); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addra(63,32); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addrb(31,0); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addrb(63,32); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addrc(31,0); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addrc(63,32); tmp.last=1;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
            }  
        }

        /**
         * @brief Wait for a previously-launched call to finish
         * 
         */
        void finalize_call(){
            STREAM_READ(sts);
        }

        /**
         * @brief Perform ACCL local array copy
         * 
         * @param len Number of array elements
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void copy(  ap_uint<32> len,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_COPY, len, 0, 0, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Perform ACCL local array combine
         * 
         * @param len Number of array elements
         * @param op0_addr Address of first operand array
         * @param op1_addr Address of second operand array
         * @param res_addr Address of result array
         */
        void combine(   ap_uint<32> len,
                        ap_uint<64> op0_addr,
                        ap_uint<64> op1_addr,
                        ap_uint<64> res_addr
        ){
            start_call(
                ACCL_COMBINE, len, 0, 0, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                op0_addr, op1_addr, res_addr
            );
            finalize_call();
        }

        /**
         * @brief Send data to a remote peer. Two-sided, i.e. requires a recv on remote end.
         * 
         * @param len Number of array elements
         * @param tag Message tag
         * @param dst_rank Rank ID of destination
         * @param src_addr Source array address
         */
        void send(  ap_uint<32> len,
                    ap_uint<32> tag,
                    ap_uint<32> dst_rank,
                    ap_uint<64> src_addr
        ){
            start_call(
                ACCL_SEND, len, comm_adr, dst_rank, 0, tag, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief One-sided data transfer to a stream on a remote peer. 
         * 
         * @param len Number of array elements
         * @param stream_id Stream ID at destination. IDs >=247 are reserved, call will not execute if set in this range
         * @param dst_rank Rank ID of destination
         * @param src_addr Source array address
         */
        void stream_put(ap_uint<32> len,
                        ap_uint<32> stream_id,
                        ap_uint<32> dst_rank,
                        ap_uint<64> src_addr
        ){
            if(stream_id > 246) return;
            start_call(
                ACCL_SEND, len, comm_adr, dst_rank, 0, stream_id, 
                dpcfg_adr, cflags, sflags | 0x2, 
                src_addr, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief Receive data send from a remote peer.
         * 
         * @param len Number of array elements
         * @param tag Message tag
         * @param src_rank Rank ID of sender
         * @param dst_addr Destination array address
         */
        void recv(  ap_uint<32> len,
                    ap_uint<32> tag,
                    ap_uint<32> src_rank,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_RECV, len, comm_adr, src_rank, 0, tag, 
                dpcfg_adr, cflags, sflags,
                0, dst_addr, 0
            );
            finalize_call();
        }

        /**
         * @brief Broadcast data to members of the communicator
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param src_addr Source array address
         */
        void bcast( ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<64> src_addr
        ){
            start_call(
                ACCL_BCAST, len, comm_adr, root, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief Scatter data to members of the communicator
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void scatter( ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_SCATTER, len, comm_adr, root, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Gather data from members of the communicator
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void gather(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_GATHER, len, comm_adr, root, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief All-gather data in the communicator. Equivalent to gather followed by broadcast
         * 
         * @param len Number of array elements
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void all_gather(ap_uint<32> len,
                        ap_uint<64> src_addr,
                        ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_ALLGATHER, len, comm_adr, 0, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Reduce data from members of the communicator
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param function Reduction function ID
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void reduce(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<32> function,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_REDUCE, len, comm_adr, root, function, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Reduce-scatter data in the communicator. Equivalent to reduce followed by scatter
         * 
         * @param len Number of array elements
         * @param function Reduction function ID
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void reduce_scatter(ap_uint<32> len,
                            ap_uint<32> function,
                            ap_uint<64> src_addr,
                            ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_REDUCE_SCATTER, len, comm_adr, 0, function, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief All-reduce data in the communicator. Equivalent to reduce followed by broadcast
         * 
         * @param len Number of array elements
         * @param function Reduction function ID
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void all_reduce(ap_uint<32> len,
                        ap_uint<32> function,
                        ap_uint<64> src_addr,
                        ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_ALLREDUCE, len, comm_adr, 0, function, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }
};

/**
 * @brief Class encapsulating ACCL data streams
 * 
 */
class ACCLData{
    public:
        /**
         * @brief Construct a new ACCLData object
         * 
         * @param krnl2cclo Reference to data stream from user kernel to CCLO
         * @param cclo2krnl Reference to data stream from CCLO to user kernel
         */
        ACCLData(STREAM<stream_word> &krnl2cclo, STREAM<stream_word> &cclo2krnl) : 
                    cclo2krnl(cclo2krnl), krnl2cclo(krnl2cclo){}

    protected:
        STREAM<stream_word> &krnl2cclo;
        STREAM<stream_word> &cclo2krnl;

    public:
        /**
         * @brief Push user data to the CCLO
         * 
         * @param data Data word (64B)
         * @param dest Destination value (potentially used in routing)
         */
        void push(ap_uint<DATA_WIDTH> data, ap_uint<DEST_WIDTH> dest){
            stream_word tmp;
            tmp.data = data;
            tmp.dest = dest;
            tmp.last = 1;
            tmp.keep = -1;
            krnl2cclo.write(tmp);
        }

        /**
         * @brief Pull data from CCLO stream
         * 
         * @return stream_word
         */
        stream_word pull(){
            return cclo2krnl.read();   
        }
};

}
