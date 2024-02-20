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

#include <stdlib.h>
#include <string>
#include <iostream>
#include <signal.h>
#include <zmqpp/zmqpp.hpp>
#include <vector>
#include <chrono>
#include <thread>
#include "ccl_offload_control.h"
#include "xsi_dut.h"
#include "Stream.h"
#include "Simulation.h"
#include "Axi.h"
#include "cclo_sim.h"
#include <filesystem>
#include "zmq_server.h"
#include "log.hpp"
#include <tclap/CmdLine.h>

#ifndef DEFAULT_LOG_LEVEL
    #define DEFAULT_LOG_LEVEL 3
#endif

using namespace std;
using namespace hlslib;

namespace {
    const axilite control("s_axi_control");
    const axistream callreq("s_axis_call_req");
    const axistream callack("m_axis_call_ack");
    const axistream eth_tx("m_axis_eth_tx_data");
    const axistream eth_rx("s_axis_eth_rx_data");
    const axistream krnl_tx("m_axis_krnl");
    const axistream krnl_rx("s_axis_krnl");
    const aximm datamem("s_axi_data");
    bool stop = false;
    Log logger;
}

void control_read_fsm(XSI_DUT *dut, Stream<unsigned int> &addr, Stream<unsigned int> &ret){
    static axi_fsm_state state = VALID_ADDR;
    switch(state){
        case VALID_ADDR:
            //set araddr = addr
            if(!addr.IsEmpty()){
                dut->write(control.araddr(), addr.Pop());
                //set arvalid = 1
                dut->set(control.arvalid());
                if(dut->test(control.arready())){
                    state = CLEAR_ADDR;
                } else{
                    state = READY_ADDR;
                }
            }
            return;
        case READY_ADDR:
            //advance 1 clock period until arready == 1
            if(dut->test(control.arready())){
                state = CLEAR_ADDR;
            }
            return;
        case CLEAR_ADDR:
            dut->clear(control.arvalid());
            dut->set(control.rready());
            state = READY_DATA;
            return;
        case READY_DATA:
            if(dut->test(control.rvalid())){
                ret.Push(dut->read(control.rdata()));
                state = CLEAR_DATA;
            }
            return;
        case CLEAR_DATA:
            dut->clear(control.rready());
            state = VALID_ADDR;
            return;
    }
}

void data_read_fsm(XSI_DUT *dut, Stream<ap_uint<64> > &addr, Stream<ap_uint<32> > &len, Stream<ap_uint<512> > &ret){
    static axi_fsm_state state = VALID_ADDR;
    static ap_uint<64> curr_addr = 0;
    static unsigned int curr_nbytes = 0;
    static ap_uint<8> nbeats = 0;
    unsigned int nbytes_to_4k_boundary, nbytes_this_transfer;
    switch(state){
        case VALID_ADDR:
            if(!addr.IsEmpty()){
                curr_addr = addr.Pop();
                curr_nbytes = len.Pop();
                nbytes_to_4k_boundary = (curr_addr/4096+1)*4096 - curr_addr;
                nbytes_this_transfer = std::min(nbytes_to_4k_boundary, curr_nbytes);
                nbeats = (nbytes_this_transfer+63)/64;//number of 64B beats in transfer
                logger << log_level::debug << "Read start addr=" << curr_addr << " len=" << nbytes_this_transfer << " (" << nbeats << ")" << endl;
                //set up bus to transfer one 64-byte word at a time
                dut->write(datamem.arsize(), 6);//64B
                dut->write(datamem.arlen(), nbeats-1);
                dut->write(datamem.arburst(), 1);//INCR
                dut->write<64>(datamem.araddr(), curr_addr);
                dut->set(datamem.arvalid());
                if(dut->test(datamem.arready())){
                    state = CLEAR_ADDR;
                } else {
                    state = READY_ADDR;
                }
                curr_addr += nbytes_this_transfer;
                curr_nbytes -= nbytes_this_transfer;
            }
            return;
        case CONTINUE_ADDR:
            nbytes_to_4k_boundary = (curr_addr/4096+1)*4096 - curr_addr;
            nbytes_this_transfer = std::min(nbytes_to_4k_boundary, curr_nbytes);
            nbeats = (nbytes_this_transfer+63)/64;//number of 64B beats in transfer
            logger << log_level::debug << "Read continue addr=" << curr_addr << " len=" << nbytes_this_transfer << " (" << nbeats << ")" << endl;
            //set up bus to transfer one 64-byte word at a time
            dut->write(datamem.arsize(), 6);//64B
            dut->write(datamem.arlen(), nbeats-1);
            dut->write(datamem.arburst(), 1);//INCR
            dut->write<64>(datamem.araddr(), curr_addr);
            dut->set(datamem.arvalid());
            if(dut->test(datamem.arready())){
                state = CLEAR_ADDR;
            } else {
                state = READY_ADDR;
            }
            curr_addr += nbytes_this_transfer;
            curr_nbytes -= nbytes_this_transfer;
            return;
        case READY_ADDR:
            if(dut->test(datamem.arready())){
                state = CLEAR_ADDR;
            }
            return;
        case CLEAR_ADDR:
            dut->clear(datamem.arvalid());
            dut->set(datamem.rready());
            state = READY_DATA;
            return;
        case READY_DATA:
            if(dut->test(datamem.rvalid())){
                nbeats--;
                ret.Push(dut->read<512>(datamem.rdata()));
                if(nbeats == 0){
                    state = CLEAR_DATA;
                }
            }
            return;
        case CLEAR_DATA:
            dut->clear(datamem.rready());
            if(curr_nbytes == 0){
                state = VALID_ADDR;
            } else {
                state = CONTINUE_ADDR;
            }
            return;
    }
}

void control_write_fsm(XSI_DUT *dut, Stream<unsigned int> &addr, Stream<unsigned int> &val){
    static axi_fsm_state state = VALID_ADDR;
    switch(state){
        case VALID_ADDR:
            //set awaddr = addr
            if(!addr.IsEmpty()){
                dut->write(control.awaddr(), addr.Pop());
                //set awvalid = 1
                dut->set(control.awvalid());
                if(dut->test(control.awready())){
                    state = CLEAR_ADDR;
                } else{
                    state = READY_ADDR;
                }
            }
            return;
        case READY_ADDR:
            if(dut->test(control.awready())){
                state = CLEAR_ADDR;
            }
            return;
        case CLEAR_ADDR:
            dut->clear(control.awvalid());
            state = VALID_DATA;
            return;
        case VALID_DATA:
            if(!val.IsEmpty()){
                dut->write(control.wdata(), val.Pop());
                dut->write(control.wstrb(), 0xF);
                dut->set(control.wvalid());
                if(dut->test(control.wready())){
                    state = CLEAR_DATA;
                } else{
                    state = READY_DATA;
                }
            }
            return;
        case READY_DATA:
            if(dut->test(control.wready())){
                state = CLEAR_DATA;
            }
            return;
        case CLEAR_DATA:
            dut->clear(control.wvalid());
            state = VALID_ACK;
            return;
        case VALID_ACK:
            dut->set(control.bready());
            if(dut->test(control.bvalid())){
                state = CLEAR_ACK;
            } else{
                state = READY_ACK;
            }
            return;
        case READY_ACK:
            if(dut->test(control.bvalid())){
                state = CLEAR_ACK;
            }
            return;
        case CLEAR_ACK:
            dut->clear(control.bready());
            state = VALID_ADDR;
            return;
    }
}

void data_write_fsm(XSI_DUT *dut, Stream<ap_uint<64> > &addr, Stream<ap_uint<32> > &len, Stream<ap_uint<512> > &val, Stream<ap_uint<64> > &strb){
    static axi_fsm_state state = VALID_ADDR;
    static ap_uint<64> curr_addr = 0;
    static unsigned int curr_nbytes = 0;
    static ap_uint<8> nbeats = 0;
    unsigned int nbytes_to_4k_boundary, nbytes_this_transfer;
    switch(state){
        case VALID_ADDR:
            //set awaddr = addr
            if(!addr.IsEmpty()){
                curr_addr = addr.Pop();
                curr_nbytes = len.Pop();
                nbytes_to_4k_boundary = (curr_addr/4096+1)*4096 - curr_addr;
                nbytes_this_transfer = std::min(nbytes_to_4k_boundary, curr_nbytes);
                nbeats = (nbytes_this_transfer+63)/64;//number of 64B beats in transfer
                logger << log_level::debug << "Write start addr=" << curr_addr << " len=" << nbytes_this_transfer << " (" << nbeats << ")" << endl;
                dut->write(datamem.awsize(), 6);//64B width
                dut->write(datamem.awlen(), nbeats-1); 
                dut->write(datamem.awburst(), 1);//INCR
                dut->write<64>(datamem.awaddr(), curr_addr);
                //set awvalid = 1
                dut->set(datamem.awvalid());
                if(dut->test(datamem.awready())){
                    state = CLEAR_ADDR;
                } else{
                    state = READY_ADDR;
                }
                curr_addr += nbytes_this_transfer;
                curr_nbytes -= nbytes_this_transfer;
            }
            return;
        case CONTINUE_ADDR:
            nbytes_to_4k_boundary = (curr_addr/4096+1)*4096 - curr_addr;
            nbytes_this_transfer = std::min(nbytes_to_4k_boundary, curr_nbytes);
            nbeats = (nbytes_this_transfer+63)/64;//number of 64B beats in transfer
            logger << log_level::debug << "Write continue addr=" << curr_addr << " len=" << nbytes_this_transfer << " (" << nbeats << ")" << endl;
            dut->write(datamem.awsize(), 6);//64B width
            dut->write(datamem.awlen(), nbeats-1); 
            dut->write(datamem.awburst(), 1);//INCR
            dut->write<64>(datamem.awaddr(), curr_addr);
            //set awvalid = 1
            dut->set(datamem.awvalid());
            if(dut->test(datamem.awready())){
                state = CLEAR_ADDR;
            } else{
                state = READY_ADDR;
            }
            curr_addr += nbytes_this_transfer;
            curr_nbytes -= nbytes_this_transfer;
            return;
        case READY_ADDR:
            if(dut->test(datamem.awready())){
                state = CLEAR_ADDR;
            }
            return;
        case CLEAR_ADDR:
            dut->clear(datamem.awvalid());
            state = VALID_DATA;
            return;
        case VALID_DATA:
            if(!val.IsEmpty() && !strb.IsEmpty()){
                dut->write<512>(datamem.wdata(), val.Pop());
                dut->write<64>(datamem.wstrb(), strb.Pop());
                dut->write(datamem.wlast(), nbeats==1);
                //set wvalid = 1
                dut->set(datamem.wvalid());
                if(dut->test(datamem.wready())){
                    nbeats--;
                    if(nbeats != 0){
                        state = UPDATE_DATA;
                    } else {
                        state = CLEAR_DATA;
                    }
                } else{
                    state = READY_DATA;
                }
            }
            return;
        case READY_DATA:
            if(dut->test(datamem.wready())){
                nbeats--;
                if(nbeats != 0){
                    state = UPDATE_DATA;
                } else {
                    state = CLEAR_DATA;
                }
            }
            return;
        case UPDATE_DATA:
            if(!val.IsEmpty() && !strb.IsEmpty()){
                dut->write<512>(datamem.wdata(), val.Pop());
                dut->write<64>(datamem.wstrb(), strb.Pop());
                dut->write(datamem.wlast(), nbeats==1);
                if(dut->test(datamem.wready())){
                    nbeats--;
                    if(nbeats != 0){
                        state = UPDATE_DATA;
                    } else {
                        state = CLEAR_DATA;
                    }
                } else{
                    state = READY_DATA;
                }
            }
            return;
        case CLEAR_DATA:
            dut->clear(datamem.wvalid());
            state = VALID_ACK;
            return;
        case VALID_ACK:
            dut->set(datamem.bready());
            if(dut->test(datamem.bvalid())){
                state = CLEAR_ACK;
            } else{
                state = READY_ACK;
            }
            return;
        case READY_ACK:
            if(dut->test(datamem.bvalid())){
                state = CLEAR_ACK;
            }
            return;
        case CLEAR_ACK:
            dut->clear(datamem.bready());
            if(curr_nbytes == 0){
                state = VALID_ADDR;
            } else {
                state = CONTINUE_ADDR;
            }
            return;
    }
}

void call_req_fsm(XSI_DUT *dut, Stream<unsigned int> &val){
    static axi_fsm_state state = VALID_DATA;
    switch(state){
        case VALID_DATA:
            if(!val.IsEmpty()){
                dut->write(callreq.tdata(), val.Pop());
                dut->set(callreq.tvalid());
                if(dut->test(callreq.tready())){
                    state = CLEAR_DATA;
                } else{
                    state = READY_DATA;
                }
            }
            return;
        case READY_DATA:
            if(dut->test(callreq.tready())){
                state = CLEAR_DATA;
            }
            return;
        case CLEAR_DATA:
            dut->clear(callreq.tvalid());
            state = VALID_DATA;
            return;
    }
}

void call_ack_fsm(XSI_DUT *dut, Stream<unsigned int> &ack){
    //not much of a FSM for now since we just wait for valid and put the data in a stream
    dut->set(callack.tready());
    if(dut->test(callack.tvalid())){
        ack.Push(dut->read(callack.tdata()));
    }
}

void eth_ingress_fsm(XSI_DUT *dut, Stream<stream_word> &val){
    static axi_fsm_state state = VALID_DATA;
    stream_word tmp;
    switch(state){
        case VALID_DATA:
            if(!val.IsEmpty()){
                tmp = val.Pop();
                dut->write<512>(eth_rx.tdata(), tmp.data);
                dut->write(eth_rx.tdest(), tmp.dest);
                dut->write<64>(eth_rx.tkeep(), tmp.keep);
                dut->write(eth_rx.tlast(), tmp.last);
                dut->set(eth_rx.tvalid());
                if(dut->test(eth_rx.tready())){
                    if(!val.IsEmpty()){
                        state = VALID_DATA;
                    } else{
                        state = CLEAR_DATA;
                    }
                } else{
                    state = READY_DATA;
                }
            }
            return;
        case READY_DATA:
            if(dut->test(eth_rx.tready())){
                if(!val.IsEmpty()){
                    state = VALID_DATA;
                } else{
                    state = CLEAR_DATA;
                }

            }
            return;
        case CLEAR_DATA:
            dut->clear(eth_rx.tvalid());
            state = VALID_DATA;
            return;
    }
}

void eth_egress_fsm(XSI_DUT *dut, Stream<stream_word> &val){
    stream_word tmp;
    if(val.IsFull()){
        dut->clear(eth_tx.tready());
    } else{
        dut->set(eth_tx.tready());
        if(dut->test(eth_tx.tvalid())){
            tmp.data = dut->read<512>(eth_tx.tdata());
            tmp.dest = dut->read(eth_tx.tdest());
            tmp.last = dut->read(eth_tx.tlast());
            tmp.keep = dut->read<64>(eth_tx.tkeep());
            val.Push(tmp);
        }
    }
}

void krnl_ingress_fsm(XSI_DUT *dut, Stream<stream_word> &val){
    static axi_fsm_state state = VALID_DATA;
    stream_word tmp;
    switch(state){
        case VALID_DATA:
            if(!val.IsEmpty()){
                tmp = val.Pop();
                dut->write<512>(krnl_rx.tdata(), tmp.data);
                dut->write(krnl_rx.tdest(), tmp.dest);
                dut->write<64>(krnl_rx.tkeep(), tmp.keep);
                dut->write(krnl_rx.tlast(), tmp.last);
                dut->set(krnl_rx.tvalid());
                if(dut->test(krnl_rx.tready())){
                    if(!val.IsEmpty()){
                        state = VALID_DATA;
                    } else{
                        state = CLEAR_DATA;
                    }
                } else{
                    state = READY_DATA;
                }
            }
            return;
        case READY_DATA:
            if(dut->test(krnl_rx.tready())){
                if(!val.IsEmpty()){
                    state = VALID_DATA;
                } else{
                    state = CLEAR_DATA;
                }

            }
            return;
        case CLEAR_DATA:
            dut->clear(krnl_rx.tvalid());
            state = VALID_DATA;
            return;
    }
}

void krnl_egress_fsm(XSI_DUT *dut, Stream<stream_word> &val){
    stream_word tmp;
    if(val.IsFull()){
        dut->clear(krnl_tx.tready());
    } else{
        dut->set(krnl_tx.tready());
        if(dut->test(krnl_tx.tvalid())){
            tmp.data = dut->read<512>(krnl_tx.tdata());
            tmp.dest = dut->read(krnl_tx.tdest());
            tmp.last = dut->read(krnl_tx.tlast());
            tmp.keep = dut->read<64>(krnl_tx.tkeep());
            val.Push(tmp);
        }
    }
}

void interface_handler(XSI_DUT *dut, Stream<unsigned int> &axilite_rd_addr, Stream<unsigned int> &axilite_rd_data,
                        Stream<unsigned int> &axilite_wr_addr, Stream<unsigned int> &axilite_wr_data,
                        Stream<ap_uint<64> > &aximm_rd_addr, Stream<ap_uint<32> > &aximm_rd_len, Stream<ap_uint<512> > &aximm_rd_data,
                        Stream<ap_uint<64> > &aximm_wr_addr, Stream<ap_uint<32> > &aximm_wr_len, Stream<ap_uint<512> > &aximm_wr_data, Stream<ap_uint<64> > &aximm_wr_strb,
                        Stream<unsigned int> &callreq, Stream<unsigned int> &callack,
                        Stream<stream_word> &eth_tx_data, Stream<stream_word> &eth_rx_data,
                        Stream<stream_word> &cclo_to_krnl_data, Stream<stream_word> &krnl_to_cclo_data){
    logger << log_level::verbose << "Starting XSI interface server" << endl;
    while(!stop){
        dut->run_ncycles(1);
        control_read_fsm(dut, axilite_rd_addr, axilite_rd_data);
        control_write_fsm(dut, axilite_wr_addr, axilite_wr_data);
        data_read_fsm(dut, aximm_rd_addr, aximm_rd_len, aximm_rd_data);
        data_write_fsm(dut, aximm_wr_addr, aximm_wr_len, aximm_wr_data, aximm_wr_strb);
        call_req_fsm(dut, callreq);
        call_ack_fsm(dut, callack);
        eth_ingress_fsm(dut, eth_rx_data);
        eth_egress_fsm(dut, eth_tx_data);
        krnl_ingress_fsm(dut, krnl_to_cclo_data);
        krnl_egress_fsm(dut, cclo_to_krnl_data);
    }
    logger << log_level::verbose << "Exiting XSI interface server" << endl;
}

//this function copies from the global var stop
//into the zmq context, as the handler can't take arguments
void update_zmq_stop(zmq_intf_context *ctx){
    while(!stop){
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    ctx->stop = true;
}

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void finish(int signum) {
    stop = true;
}

int main(int argc, char **argv)
{
    std::string simengine_libname = "librdi_simulator_kernel.so";
    std::string design_libname;

    TCLAP::CmdLine cmd("ACCL Simulator");
    TCLAP::ValueArg<unsigned int> loglevel("l", "loglevel",
                                          "Verbosity level of logging",
                                          false, DEFAULT_LOG_LEVEL, "positive integer");
    cmd.add(loglevel);
    TCLAP::ValueArg<unsigned int> worldsize("s", "worldsize",
                                          "Total number of ranks",
                                          false, 1, "positive integer");
    cmd.add(worldsize);
    TCLAP::ValueArg<unsigned int> localrank("r", "localrank",
                                          "Index of the local rank",
                                          false, 0, "positive integer");
    cmd.add(localrank);

    TCLAP::ValueArg<std::string> designlib("d", "designlib",
                                           "Name of compiled design library",
                                           false, "xsim.dir/ccl_offload_behav/xsimk.so", "file");
    cmd.add(designlib);

    TCLAP::ValueArg<unsigned int> startport("p", "port",
                                          "Starting ZMQ port",
                                          false, 5500, "positive integer");
    cmd.add(startport);

    TCLAP::SwitchArg loopback_arg("b", "loopback", "Enable kernel loopback", cmd, false);
    TCLAP::SwitchArg wave_en("w", "waveform", "Enable waveform recording", cmd, false);

    try {
        cmd.parse(argc, argv);
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    //set up the logger with the specified log level
    logger = Log(static_cast<log_level>(loglevel.getValue()), localrank.getValue());

    //get world size and local rank from environment
    unsigned int world_size = worldsize.getValue(); // number of processes
    unsigned int local_rank = localrank.getValue(); // the rank of the process
    logger << log_level::info << "World Size: " << world_size << " Local Rank: " << local_rank << endl;


    design_libname = designlib.getValue();

    logger << log_level::verbose << "Design DLL     : " << design_libname << std::endl;
    logger << log_level::verbose << "Sim Engine DLL : " << simengine_libname << std::endl;
    logger << log_level::verbose << "Library path: " << std::getenv("LD_LIBRARY_PATH") << std::endl;
    std::string wdb_name = filesystem::current_path().string() + "/waveform_rank" + std::to_string(local_rank) + ".wdb";
    logger << log_level::verbose << "Waveform : " << wdb_name << std::endl;
    XSI_DUT dut(design_libname, simengine_libname, "ap_rst_n", true, "ap_clk", 4, wdb_name, logger, wave_en.getValue());
    logger << log_level::info << "DUT initialized" << std::endl;

    zmq_intf_context ctx = zmq_server_intf(startport.getValue(), local_rank, world_size, loopback_arg.getValue(), logger);

    int status = 0;

    Stream<unsigned int> axilite_rd_addr;
    Stream<unsigned int> axilite_rd_data;
    Stream<unsigned int> axilite_wr_addr;
    Stream<unsigned int> axilite_wr_data;
    Stream<ap_uint<64> > aximm_rd_addr;
    Stream<ap_uint<32> > aximm_rd_len;
    Stream<ap_uint<512> > aximm_rd_data;
    Stream<ap_uint<64> > aximm_wr_addr;
    Stream<ap_uint<32> > aximm_wr_len;
    Stream<ap_uint<512> > aximm_wr_data;
    Stream<ap_uint<64> > aximm_wr_strb;
    Stream<unsigned int, 16> callreq; //need some capacity for all args
    Stream<unsigned int> callack;

    Stream<stream_word> eth_tx_data;
    Stream<stream_word> eth_rx_data;

    Stream<stream_word> cclo_to_krnl_data;
    Stream<stream_word> krnl_to_cclo_data;

    try {
        // Register signal and signal handler
        signal(SIGINT, finish);
        signal(SIGTERM, finish);//for running under MPI, which sends SIGTERM instead of SIGINT when it itself receives SIGINT

        //reset design and let it run for a while to initialize
        logger << log_level::verbose << "Resetting design" << endl;
        dut.reset_design();
        logger << log_level::verbose << "Reset done" << endl;
        logger << log_level::verbose << "Initializing design" << endl;
        dut.run_ncycles(1000);
        logger << log_level::verbose << "Initialization done" << endl;

        HLSLIB_DATAFLOW_INIT();
        HLSLIB_DATAFLOW_FUNCTION(interface_handler, &dut,
                                    axilite_rd_addr, axilite_rd_data,
                                    axilite_wr_addr, axilite_wr_data,
                                    aximm_rd_addr, aximm_rd_len, aximm_rd_data,
                                    aximm_wr_addr, aximm_wr_len, aximm_wr_data, aximm_wr_strb,
                                    callreq, callack,
                                    eth_tx_data, eth_rx_data,
                                    cclo_to_krnl_data, krnl_to_cclo_data);
        HLSLIB_DATAFLOW_FUNCTION(zmq_cmd_server,  &ctx,
                                    axilite_rd_addr, axilite_rd_data,
                                    axilite_wr_addr, axilite_wr_data,
                                    aximm_rd_addr, aximm_rd_len, aximm_rd_data,
                                    aximm_wr_addr, aximm_wr_len, aximm_wr_data, aximm_wr_strb,
                                    callreq, callack);
        //ZMQ to other nodes process(es)
        HLSLIB_DATAFLOW_FUNCTION(zmq_eth_egress_server, &ctx, eth_tx_data, local_rank);
        HLSLIB_DATAFLOW_FUNCTION(zmq_eth_ingress_server, &ctx, eth_rx_data);
        HLSLIB_DATAFLOW_FUNCTION(zmq_krnl_egress_server, &ctx, cclo_to_krnl_data);
        HLSLIB_DATAFLOW_FUNCTION(zmq_krnl_ingress_server, &ctx, krnl_to_cclo_data);
        HLSLIB_DATAFLOW_FUNCTION(update_zmq_stop, &ctx);
        HLSLIB_DATAFLOW_FINALIZE();
    }
    catch (std::exception& e) {
        logger << log_level::error << "An exception occurred: " << e.what() << std::endl;
        status = 2;
    }
    catch (...) {
        logger << log_level::error << "An unknown exception occurred." << std::endl;
        status = 3;
    }

    if(status == 0) {
        logger << log_level::info << "PASSED test" << std::endl;
    } else {
        logger << log_level::warning << "FAILED test" << std::endl;
    }

    exit(status);
}
