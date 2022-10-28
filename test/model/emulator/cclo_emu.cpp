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

#include "streamdefines.h"
#include "Stream.h"
#include "Simulation.h"
#include "Axi.h"
#include <pthread.h>
#include <thread>
#include <iostream>
#include <sstream>
#include <fstream>
#include "ap_int.h"
#include <stdint.h>
#include "reduce_ops.h"
#include "hp_compression.h"
#include "eth_intf.h"
#include "dummy_tcp_stack.h"
#include "stream_segmenter.h"
#include "client_arbiter.h"
#include "hostctrl.h"
#include "rxbuf_offload.h"
#include "dma_mover.h"
#include "ccl_offload_control.h"
#include <zmqpp/zmqpp.hpp>
#include <string>
#include <chrono>
#include <numeric>
#include <mpi.h>
#include "zmq_server.h"
#include "log.hpp"

#ifndef DEFAULT_LOG_LEVEL
    #define DEFAULT_LOG_LEVEL 3
#endif

using namespace std;
using namespace hlslib;

namespace {
    Log logger;
}

void dma_read(vector<char> &mem, Stream<ap_uint<104> > &cmd, Stream<ap_uint<32> > &sts, Stream<stream_word > &rdata){
    axi::Command<64, 23> command = axi::Command<64, 23>(cmd.Pop());
    axi::Status status;
    stream_word tmp;
    logger << log_level::verbose << "DMA Read: Command popped. length: " << command.length << " offset: " << command.address << endl;
    int byte_count = 0;
    while(byte_count < command.length){
        tmp.keep = 0;
        for(int i=0; i<64 && byte_count < command.length; i++){
            tmp.data(8*(i+1)-1, 8*i) = mem.at(command.address+byte_count);
            tmp.keep(i,i) = 1;
            byte_count++;
        }
        tmp.last = (byte_count >= command.length);
        rdata.Push(tmp);
    }
    status.okay = 1;
    status.tag = command.tag;
    sts.Push(status);
    logger("DMA Read: Status pushed\n", log_level::verbose);
}

void dma_write(vector<char> &mem, Stream<ap_uint<104> > &cmd, Stream<ap_uint<32> > &sts, Stream<stream_word > &wdata){
    axi::Command<64, 23> command = axi::Command<64, 23>(cmd.Pop());
    axi::Status status;
    stream_word tmp;
    logger << log_level::verbose << "DMA Write: Command popped. length: " << command.length << " offset: " << command.address << endl;
    int byte_count = 0;
    while(byte_count<command.length){
        tmp = wdata.Pop();
        for(int i=0; i<64; i++){
            if(tmp.keep(i,i) == 1){
                mem.at(command.address+byte_count) = tmp.data(8*(i+1)-1, 8*i);
                byte_count++;
            }
        }
        //end of packet
        if(tmp.last){
            status.endOfPacket = 1;
            break;
        }
    }
    //TODO: flush?
    status.okay = 1;
    status.tag = command.tag;
    status.bytesReceived = byte_count;
    sts.Push(status);
    logger << log_level::verbose << "DMA Write: Status pushed endOfPacket=" << status.endOfPacket << " btt=" << status.bytesReceived << endl;
}

template <unsigned int INW, unsigned int OUTW, unsigned int DESTW>
void dwc(Stream<ap_axiu<INW, 0, 0, DESTW> > &in, Stream<ap_axiu<OUTW, 0, 0, DESTW> > &out){
    ap_axiu<INW, 0, 0, DESTW> inword;
    ap_axiu<OUTW, 0, 0, DESTW> outword;

    //3 scenarios:
    //1:N (up) conversion - read N times from input, write 1 times to output
    //N:1 (down) conversion - read 1 times from input, write N times to output
    //N:M conversion - up-conversion to least common multiple, then down-conversion
    if(INW < OUTW && OUTW%INW == 0){
        //1:N case
        outword.keep = 0;
        outword.last = 0;
        outword.data = 0;
        outword.dest = inword.dest;
        for(int i=0; i<OUTW/INW; i++){
            inword = in.Pop();
            outword.data((i+1)*INW-1,i*INW) = inword.data;
            outword.keep((i+1)*INW/8-1,i*INW/8) = inword.keep;
            outword.last = 1;
            if((inword.last == 1) || (inword.keep(INW/8-1,INW/8-1) != 1)) break;
        }
        out.Push(outword);
    } else if(INW > OUTW && INW%OUTW == 0){
        //N:1 case
        inword = in.Pop();
        outword.dest = inword.dest;
        for(int i=0; i<INW/OUTW; i++){
            outword.data = inword.data((i+1)*OUTW-1,i*OUTW);
            outword.keep = inword.keep((i+1)*OUTW/8-1,i*OUTW/8);
            //last if actually at last input read or if any previous input read is incomplete
            outword.last = (i==(INW/OUTW-1)) || (outword.keep(OUTW/8-1,OUTW/8-1) != 1);
            out.Push(outword);
            if(outword.last == 1) break;
        }
    } else{
        unsigned const int inter_width = lcm(INW, OUTW);
        Stream<axi::Stream<ap_axiu<inter_width, 0, 0, 0> > > inter;
        dwc<INW, inter_width>(in, inter);
        dwc<inter_width, OUTW>(inter, out);
    }
}

//emulate an AXI Stream Switch with TDEST routing and arbitrate on TLAST
//NOTE: this implementation will block if the first transfer from a newly activated
//slave hits a full master. In that scenario, the switch will block on all
//inputs until the targeted master becomes not full. This differs from
//a physical switch, which is able to operate other streams when one blocks
template <unsigned int NSLAVES, unsigned int NMASTERS>
void axis_switch( Stream<stream_word> s[NSLAVES], Stream<stream_word> m[NMASTERS]){
    stream_word word;
    static int destination[NSLAVES];
    static bool active[NSLAVES];
    for(int i=0; i<NSLAVES; i++){
        //skip this slave if its destination is full
        if(active[i] && m[destination[i]].IsFull()) continue;
        if(!s[i].IsEmpty()){
            word = s[i].Pop();
            if(!active[i]){
                destination[i] = min(NMASTERS-1, (unsigned int)word.dest);
                active[i] = true;
            }
            m[destination[i]].Push(word);
            logger << log_level::verbose << "Switch arbitrate: S" << i << " -> M" << destination[i] << "(" << (unsigned int)word.dest << ")" << endl;
            if(word.last != 0){
                active[i] = false;
            }
        }
    }
}

//emulate an 2:1 AXI Stream Switch i.e. a AXIS multiplexer
void axis_mux(Stream<stream_word> &s0, Stream<stream_word> &s1, Stream<stream_word> &m){
    stream_word word;
    if(!s0.IsEmpty()){
        do{
            word = s0.Pop();
            m.Push(word);
            logger << log_level::verbose << "Switch mux: S0 -> M (" << (unsigned int)word.dest << ")" << endl;
        } while(word.last == 0);
    }
    if(!s1.IsEmpty()){
        do{
            word = s1.Pop();
            m.Push(word);
            logger << log_level::verbose << "Switch mux: S1 -> M (" << (unsigned int)word.dest << ")" << endl;
        } while(word.last == 0);
    }
}

//wrap a host controller
void controller(Stream<command_word> &cmdin, Stream<command_word> &cmdout,
                Stream<command_word> &stsin, Stream<command_word> &stsout){
    //gather arguments from input command queue
    ap_uint<32> scenario = cmdin.Pop().data;
    ap_uint<32> count = cmdin.Pop().data;
    ap_uint<32> comm = cmdin.Pop().data;
    ap_uint<32> root_src_dst = cmdin.Pop().data;
    ap_uint<32> function = cmdin.Pop().data;
    ap_uint<32> tag = cmdin.Pop().data;
    ap_uint<32> arithcfg = cmdin.Pop().data;
    ap_uint<32> compression_flags = cmdin.Pop().data;
    ap_uint<32> stream_flags = cmdin.Pop().data;
    ap_uint<64> addr_0 = cmdin.Pop().data;
    addr_0(63,32) = cmdin.Pop().data;
    ap_uint<64> addr_1 = cmdin.Pop().data;
    addr_1(63,32) = cmdin.Pop().data;
    ap_uint<64> addr_2 = cmdin.Pop().data;
    addr_2(63,32) = cmdin.Pop().data;
    //execute host controller
    logger << log_level::verbose << "Controller forwarding call with scenario " << scenario << endl;
    hostctrl(	scenario, count, comm, root_src_dst, function, tag, arithcfg, 
				compression_flags, stream_flags,
				addr_0, addr_1, addr_2,
				cmdout, stsin   );
    //signal upstream
    stsout.Push({.data=0, .last=1});
}

//wrap a host controller
void controller_bypass(Stream<command_word> &cmdin, Stream<command_word> &cmdout,
                Stream<command_word> &stsin, Stream<command_word> &stsout){
    //gather arguments from input command queue
    for(int i=0; i<15; i++){
        cmdout.Push(cmdin.Pop());
    }
    logger << log_level::verbose << "Controller bypass: pushed arguments" << endl;
    stsout.Push(stsin.Pop());
    logger << log_level::verbose << "Controller bypass: pushed status" << endl;
}

//emulate a subset-converter based TDEST adjustment (for streaming to PL kernels)
void axis_ssc(Stream<stream_word> &in, Stream<stream_word> &out, int adj){
    stream_word word;
    word = in.Pop();
    word.dest = (unsigned int)((int)word.dest + adj);
    out.Push(word);
}

void sim_bd(zmq_intf_context *ctx, bool use_tcp, unsigned int local_rank, unsigned int world_size) {
    vector<char> devicemem;

    Stream<ap_uint<32>, 32> host_cmd("host_cmd");
    Stream<ap_uint<32>, 32> host_sts("host_sts");

    Stream<stream_word, 32> krnl_to_accl_data("krnl_out");
    Stream<stream_word > accl_to_krnl_data("krnl_in");

    Stream<stream_word, 1024> eth_rx_data("eth_in");
    Stream<stream_word, 1024> eth_tx_data("eth_out");

    Stream<stream_word > arith_op0("arith_op0");
    Stream<stream_word > arith_op1("arith_op1");
    Stream<stream_word > arith_res("arith_res");

    Stream<stream_word > clane0_op("clane0_op");
    Stream<stream_word > clane0_res("clane0_res");

    Stream<stream_word > clane1_op("clane1_op");
    Stream<stream_word > clane1_res("clane1_res");

    Stream<stream_word > clane2_op("clane2_op");
    Stream<stream_word > clane2_res("clane2_res");

    Stream<ap_uint<104>, 32> dma_write_cmd_int[2];
    Stream<ap_uint<104>, 32> dma_read_cmd_int[2];
    Stream<ap_uint<32>, 32> dma_write_sts_int[2];
    Stream<ap_uint<32>, 32> dma_read_sts_int[2];
    Stream<stream_word > dma_read_data[2];

    Stream<stream_word > switch_s[8];
    Stream<stream_word > switch_m[10];
    Stream<segmenter_cmd> seg_cmd[13];
    Stream<ap_uint<32> > seg_sts[13];
    Stream<stream_word > accl_to_krnl_seg;
    Stream<stream_word > accl_to_krnl_ssc;

    Stream<eth_header > eth_tx_cmd;
    Stream<ap_uint<32> > eth_tx_sts;
    Stream<eth_header > eth_rx_sts;
    Stream<eth_header > eth_rx_sts_sess;

    Stream<ap_uint<32>, 32> inflight_rxbuf;
    Stream<ap_uint<32>, 32> inflight_rxbuf_sess;

    Stream<ap_uint<104>, 32> enq2sess_dma_cmd;
    Stream<ap_uint<32>, 32> sess2deq_dma_sts;

    Stream<rxbuf_notification> eth_rx_notif;
    Stream<rxbuf_signature> eth_rx_seek_req;
    Stream<rxbuf_seek_result> eth_rx_seek_ack;
    Stream<ap_uint<32> > rxbuf_release_req;

    Stream<pkt16> eth_listen_port;
    Stream<pkt8> eth_port_status;
    Stream<pkt64> eth_open_connection;
    Stream<pkt16> eth_close_connection;
    Stream<pkt128> eth_open_status;

    Stream<ap_uint<96> > cmd_txHandler;
    Stream<pkt32> eth_tx_meta;
    Stream<pkt64> eth_tx_status;

    Stream<pkt128> eth_notif;
    Stream<pkt32> eth_read_pkg;
    Stream<pkt16> eth_rx_meta;
    Stream<eth_notification> eth_notif_out;
    Stream<eth_notification> eth_notif_out_dpkt;

    Stream<stream_word, 1024> eth_tx_data_int;
    Stream<stream_word, 1024> eth_rx_data_int;
    Stream<stream_word> eth_tx_data_stack;
    Stream<stream_word> eth_rx_data_stack;

    Stream<command_word> callreq_fifos[NUM_CTRL_STREAMS];
    Stream<command_word> callack_fifos[NUM_CTRL_STREAMS];

    Stream<command_word, 512> callreq_arb[NUM_CTRL_STREAMS], callack_arb[NUM_CTRL_STREAMS];
    Stream<command_word, 512> callreq_arb_host, callack_arb_host;

    unsigned int max_words_per_pkt = MAX_PACKETSIZE/DATAPATH_WIDTH_BYTES;

    // Dataflow functions running in parallel
    HLSLIB_DATAFLOW_INIT();
    //DMA0
    HLSLIB_FREERUNNING_FUNCTION(dma_write, devicemem, dma_write_cmd_int[0], dma_write_sts_int[0], switch_m[SWITCH_M_DMA0_WRITE]);
    HLSLIB_FREERUNNING_FUNCTION(dma_read, devicemem, dma_read_cmd_int[0], dma_read_sts_int[0], dma_read_data[0]);
    //DMA1
    HLSLIB_FREERUNNING_FUNCTION(dma_write, devicemem, dma_write_cmd_int[1], dma_write_sts_int[1], switch_m[SWITCH_M_DMA1_WRITE]);
    HLSLIB_FREERUNNING_FUNCTION(dma_read, devicemem, dma_read_cmd_int[1], dma_read_sts_int[1], dma_read_data[1]);
    //RX buffer handling offload
    if(!use_tcp){
        HLSLIB_FREERUNNING_FUNCTION(rxbuf_enqueue, dma_write_cmd_int[0], inflight_rxbuf, cfgmem);
        HLSLIB_FREERUNNING_FUNCTION(rxbuf_dequeue, dma_write_sts_int[0], eth_rx_sts, inflight_rxbuf, eth_rx_notif, cfgmem);
        HLSLIB_FREERUNNING_FUNCTION(rxbuf_seek, eth_rx_notif, eth_rx_seek_req, eth_rx_seek_ack, rxbuf_release_req, cfgmem);
    } else{
        HLSLIB_FREERUNNING_FUNCTION(rxbuf_enqueue, enq2sess_dma_cmd, inflight_rxbuf, cfgmem);
        HLSLIB_FREERUNNING_FUNCTION(rxbuf_dequeue, sess2deq_dma_sts, eth_rx_sts_sess, inflight_rxbuf_sess, eth_rx_notif, cfgmem);
        HLSLIB_FREERUNNING_FUNCTION(rxbuf_seek, eth_rx_notif, eth_rx_seek_req, eth_rx_seek_ack, rxbuf_release_req, cfgmem);
        HLSLIB_FREERUNNING_FUNCTION(
            rxbuf_session,
            enq2sess_dma_cmd, sess2deq_dma_sts,
            inflight_rxbuf, inflight_rxbuf_sess,
            dma_write_cmd_int[0], dma_write_sts_int[0],
            eth_notif_out_dpkt,
            eth_rx_sts, eth_rx_sts_sess
        );
    }
    //move offload
    HLSLIB_FREERUNNING_FUNCTION(
        dma_mover, cfgmem, cmd_fifos[CMD_DMA_MOVE], sts_fifos[STS_DMA_MOVE],
        eth_rx_seek_req, eth_rx_seek_ack, rxbuf_release_req,
        dma_read_cmd_int[0], dma_read_cmd_int[1], dma_write_cmd_int[1],
        dma_read_sts_int[0], dma_read_sts_int[1], dma_write_sts_int[1],
        eth_tx_cmd, eth_tx_sts,
        seg_cmd[0], seg_cmd[1], seg_cmd[2], seg_cmd[3], seg_cmd[4], seg_cmd[5],
        seg_cmd[6], seg_cmd[7], seg_cmd[8], seg_cmd[9],
        seg_cmd[10], seg_cmd[11], seg_cmd[12], seg_sts[3]
    );
    //SWITCH and segmenters
    HLSLIB_FREERUNNING_FUNCTION(axis_switch<8, 10>, switch_s, switch_m);
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, dma_read_data[0],             switch_s[SWITCH_S_DMA0_READ], seg_cmd[0],  seg_sts[0] );   //DMA0 read
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, dma_read_data[1],             switch_s[SWITCH_S_DMA1_READ], seg_cmd[1],  seg_sts[1] );   //DMA1 read
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, krnl_to_accl_data,            switch_s[SWITCH_S_EXT_KRNL],  seg_cmd[2],  seg_sts[2] );   //ext kernel in
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, switch_m[SWITCH_M_EXT_KRNL],  accl_to_krnl_seg,             seg_cmd[3],  seg_sts[3] );   //ext kernel out
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, switch_m[SWITCH_M_ARITH_OP0], arith_op0,                    seg_cmd[4],  seg_sts[4] );   //arith op0
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, switch_m[SWITCH_M_ARITH_OP1], arith_op1,                    seg_cmd[5],  seg_sts[5] );   //arith op1
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, arith_res,                    switch_s[SWITCH_S_ARITH_RES], seg_cmd[6],  seg_sts[6] );   //arith result
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, switch_m[SWITCH_M_CLANE0], clane0_op,                    seg_cmd[7],  seg_sts[7] );   //clane0 op
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, clane0_res,                   switch_s[SWITCH_S_CLANE0], seg_cmd[8],  seg_sts[8] );   //clane0 result
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, switch_m[SWITCH_M_CLANE1], clane1_op,                    seg_cmd[9], seg_sts[9]);   //clane1 op
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, clane1_res,                   switch_s[SWITCH_S_CLANE1], seg_cmd[10], seg_sts[10]);   //clane1 result
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, switch_m[SWITCH_M_CLANE2], clane2_op,                    seg_cmd[11], seg_sts[11]);   //clane2 op
    HLSLIB_FREERUNNING_FUNCTION(stream_segmenter, clane2_res,                   switch_s[SWITCH_S_CLANE2], seg_cmd[12], seg_sts[12]);   //clane2 result
    HLSLIB_FREERUNNING_FUNCTION(axis_ssc, switch_m[SWITCH_M_BYPASS], accl_to_krnl_ssc, -9);
    HLSLIB_FREERUNNING_FUNCTION(axis_mux, accl_to_krnl_seg, accl_to_krnl_ssc, accl_to_krnl_data);
    //ARITH
    HLSLIB_FREERUNNING_FUNCTION(reduce_ops, arith_op0, arith_op1, arith_res);
    //COMPRESS 0, 1, 2
    HLSLIB_FREERUNNING_FUNCTION(hp_compression, clane0_op, clane0_res);
    HLSLIB_FREERUNNING_FUNCTION(hp_compression, clane1_op, clane1_res);
    HLSLIB_FREERUNNING_FUNCTION(hp_compression, clane2_op, clane2_res);
    //network PACK/DEPACK
    if(use_tcp){
        HLSLIB_FREERUNNING_FUNCTION(tcp_packetizer, switch_m[SWITCH_M_ETH_TX], eth_tx_data_int, eth_tx_cmd, cmd_txHandler, eth_tx_sts, max_words_per_pkt);
        HLSLIB_FREERUNNING_FUNCTION(tcp_depacketizer, eth_rx_data_int, switch_s[SWITCH_S_ETH_RX], eth_rx_sts, eth_notif_out, eth_notif_out_dpkt);
        HLSLIB_FREERUNNING_FUNCTION(tcp_rxHandler, eth_notif,  eth_read_pkg, eth_rx_meta,  eth_rx_data_stack, eth_rx_data_int, eth_notif_out);
        HLSLIB_FREERUNNING_FUNCTION(tcp_txHandler, eth_tx_data_int, cmd_txHandler, eth_tx_meta,  eth_tx_data_stack,  eth_tx_status);
        HLSLIB_FREERUNNING_FUNCTION(
            tcp_sessionHandler,
            cmd_fifos[CMD_NET_PORT], sts_fifos[STS_NET_PORT],
            cmd_fifos[CMD_NET_CON], sts_fifos[STS_NET_CON],
            eth_listen_port, eth_port_status,
            eth_open_connection, eth_close_connection, eth_open_status
        );
        //instantiate dummy TCP stack which responds to appropriate comm patterns
        HLSLIB_FREERUNNING_FUNCTION(
            network_krnl,
            eth_notif, eth_read_pkg,
            eth_rx_meta, eth_rx_data_stack,
            eth_tx_meta, eth_tx_data_stack, eth_tx_status,
            eth_open_connection, eth_open_status, eth_close_connection,
            eth_listen_port, eth_port_status,
            eth_rx_data, eth_tx_data
        );
    } else{
        HLSLIB_FREERUNNING_FUNCTION(udp_packetizer, switch_m[SWITCH_M_ETH_TX], eth_tx_data, eth_tx_cmd, eth_tx_sts, max_words_per_pkt);
        HLSLIB_FREERUNNING_FUNCTION(udp_depacketizer, eth_rx_data, switch_s[SWITCH_S_ETH_RX], eth_rx_sts);
    }
    //emulated external kernel
    HLSLIB_FREERUNNING_FUNCTION(krnl_endpoint_egress_port, ctx, accl_to_krnl_data);
    HLSLIB_FREERUNNING_FUNCTION(krnl_endpoint_ingress_port, ctx, krnl_to_accl_data);

    //emulated hostcontrollers
    HLSLIB_FREERUNNING_FUNCTION(controller, callreq_fifos[0], callreq_arb[0], callack_arb[0], callack_fifos[0]);
    HLSLIB_FREERUNNING_FUNCTION(controller, callreq_fifos[1], callreq_arb[1], callack_arb[1], callack_fifos[1]);
    HLSLIB_FREERUNNING_FUNCTION(controller_bypass, callreq_fifos[2], callreq_arb[2], callack_arb[2], callack_fifos[2]);
    HLSLIB_FREERUNNING_FUNCTION(client_arbiter, callreq_arb[0], callack_arb[0], callreq_arb[1], callack_arb[1], callreq_arb_host, callack_arb_host);
    HLSLIB_FREERUNNING_FUNCTION(client_arbiter, callreq_arb_host, callack_arb_host, callreq_arb[2], callack_arb[2], sts_fifos[CMD_CALL], cmd_fifos[STS_CALL]);

    //ZMQ to host process
    HLSLIB_FREERUNNING_FUNCTION(serve_zmq, ctx, cfgmem, devicemem, callreq_fifos, callack_fifos);
    //ZMQ to other nodes process(es)
    HLSLIB_FREERUNNING_FUNCTION(eth_endpoint_egress_port, ctx, eth_tx_data, local_rank, use_tcp && world_size > 1);
    HLSLIB_FREERUNNING_FUNCTION(eth_endpoint_ingress_port, ctx, eth_rx_data);
    //MICROBLAZE
    HLSLIB_DATAFLOW_FUNCTION(run_accl);
    HLSLIB_DATAFLOW_FINALIZE();

    this_thread::sleep_for(chrono::milliseconds(1000));
    logger << log_level::verbose << "Rank " << local_rank << " finished" << endl;
}

int main(int argc, char** argv){
    MPI_Init(NULL, NULL);      // initialize MPI environment

    int world_size; // number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int local_rank; // the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

    char *level_env = getenv("LOG_LEVEL");
    log_level level;
    if (level_env) {
        level = static_cast<log_level>(strtoul(level_env, nullptr, 10));
    } else {
        level = static_cast<log_level>(DEFAULT_LOG_LEVEL);
    }
    logger = Log(level);

    string eth_type = argv[1];
    unsigned int starting_port = atoi(argv[2]);
    
    bool krnl_loopback = false;
    if(argc == 4 && string(argv[3]) == "loopback"){
        krnl_loopback = true;
    }

    zmq_intf_context ctx = zmq_server_intf(starting_port, local_rank, world_size, krnl_loopback, logger);
    sim_bd(&ctx, eth_type == "tcp", local_rank, world_size);
}
