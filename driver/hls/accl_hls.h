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

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_utils.h"

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

class ACCLCommand{
    public:
        ACCLCommand(hls::stream<ap_uint<32> > &cmd, hls::stream<ap_uint<32> > &sts,
                    ap_uint<32> comm_adr, ap_uint<32> dpcfg_adr,
                    ap_uint<32> cflags, ap_uint<32> sflags) : 
                    cmd(cmd), sts(sts), comm_adr(comm_adr), dpcfg_adr(dpcfg_adr), cflags(cflags), sflags(sflags) {}
        ACCLCommand(hls::stream<ap_uint<32> > &cmd, hls::stream<ap_uint<32> > &sts) : 
                    cmd(cmd), sts(sts), comm_adr(0), dpcfg_adr(0), cflags(0), sflags(0) {}

    protected:
        hls::stream<ap_uint<32> > &cmd;
        hls::stream<ap_uint<32> > &sts;
        ap_uint<32> comm_adr;
        ap_uint<32> dpcfg_adr;
        ap_uint<32> cflags;
        ap_uint<32> sflags;

    public:
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
            io_section:{
                #pragma HLS protocol fixed
                cmd.write(scenario);
                ap_wait();
                cmd.write(len);
                ap_wait();
                cmd.write(comm);
                ap_wait();
                cmd.write(root_src_dst);
                ap_wait();
                cmd.write(function);
                ap_wait();
                cmd.write(msg_tag);
                ap_wait();
                cmd.write(datapath_cfg);
                ap_wait();
                cmd.write(compression_flags);
                ap_wait();
                cmd.write(stream_flags);
                ap_wait();
                cmd.write(addra(31,0));
                ap_wait();
                cmd.write(addra(63,32));
                ap_wait();
                cmd.write(addrb(31,0));
                ap_wait();
                cmd.write(addrb(63,32));
                ap_wait();
                cmd.write(addrc(31,0));
                ap_wait();
                cmd.write(addrc(63,32));
                ap_wait();
            }  
        }

        void finalize_call(){
            sts.read(); 
        }

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

}
