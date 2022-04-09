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
#include <jsoncpp/json/json.h>
#include <vector>
#include <chrono>
#include <thread>
#include <mpi.h>
#include "ccl_offload_control.h"
#include "xsi_dut.h"
#include "Stream.h"
#include "Simulation.h"
#include "Axi.h"
#include "cclo_sim.h"
#include <filesystem>
#include "zmq_intf.h"
#include "log.hpp"

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

void data_read_fsm(XSI_DUT *dut, Stream<ap_uint<64> > &addr, Stream<ap_uint<512> > &ret){
    static axi_fsm_state state = VALID_ADDR;
    switch(state){
        case VALID_ADDR:
            //set up bus to transfer one 64-byte word at a time
            dut->write(datamem.arsize(), 6);//64B
            dut->write(datamem.arlen(), 0);//one word
            dut->write(datamem.arburst(), 1);//INCR
            if(!addr.IsEmpty()){
                dut->write<64>(datamem.araddr(), addr.Pop());
                dut->set(datamem.arvalid());
                if(dut->test(datamem.arready())){
                    state = CLEAR_ADDR;
                } else {
                    state = READY_ADDR;
                }
            }
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
                ret.Push(dut->read<512>(datamem.rdata()));
                state = CLEAR_DATA;
            }
            return;
        case CLEAR_DATA:
            dut->clear(datamem.rready());
            state = VALID_ADDR;
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

void data_write_fsm(XSI_DUT *dut, Stream<ap_uint<64> > &addr, Stream<ap_uint<512> > &val, Stream<ap_uint<64> > &strb){
    static axi_fsm_state state = VALID_ADDR;
    switch(state){
        case VALID_ADDR:
            dut->write(datamem.awsize(), 6);//64B width
            dut->write(datamem.awlen(), 0);//one word
            dut->write(datamem.awburst(), 1);//INCR
            dut->write(datamem.wlast(), 1);//always last transfer
            //set awaddr = addr
            if(!addr.IsEmpty()){
                dut->write<64>(datamem.awaddr(), addr.Pop());
                //set awvalid = 1
                dut->set(datamem.awvalid());
                if(dut->test(datamem.awready())){
                    state = CLEAR_ADDR;
                } else{
                    state = READY_ADDR;
                }
            }
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
                //set wvalid = 1
                dut->set(datamem.wvalid());
                if(dut->test(datamem.wready())){
                    state = CLEAR_DATA;
                } else{
                    state = READY_DATA;
                }
            }
            return;
        case READY_DATA:
            if(dut->test(datamem.wready())){
                state = CLEAR_DATA;
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
            state = VALID_ADDR;
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
                        Stream<ap_uint<64> > &aximm_rd_addr, Stream<ap_uint<512> > &aximm_rd_data,
                        Stream<ap_uint<64> > &aximm_wr_addr, Stream<ap_uint<512> > &aximm_wr_data, Stream<ap_uint<64> > &aximm_wr_strb,
                        Stream<unsigned int> &callreq, Stream<unsigned int> &callack,
                        Stream<stream_word> &eth_tx_data, Stream<stream_word> &eth_rx_data,
                        Stream<stream_word> &krnl_tx_data, Stream<stream_word> &krnl_rx_data){
    logger << log_level::info << "Starting XSI interface server" << endl;
    while(!stop){
        dut->run_ncycles(1);
        control_read_fsm(dut, axilite_rd_addr, axilite_rd_data);
        control_write_fsm(dut, axilite_wr_addr, axilite_wr_data);
        data_read_fsm(dut, aximm_rd_addr, aximm_rd_data);
        data_write_fsm(dut, aximm_wr_addr, aximm_wr_data, aximm_wr_strb);
        call_req_fsm(dut, callreq);
        call_ack_fsm(dut, callack);
        eth_ingress_fsm(dut, eth_rx_data);
        eth_egress_fsm(dut, eth_tx_data);
        krnl_ingress_fsm(dut, krnl_rx_data);
        krnl_egress_fsm(dut, krnl_tx_data);
    }
    logger << log_level::info << "Exiting XSI interface server" << endl;
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

    int world_size; // number of processes
    int local_rank; // the rank of the process

    char *level_env = getenv("LOG_LEVEL");
    log_level level;
    if (level_env) {
        level = static_cast<log_level>(strtoul(level_env, nullptr, 10));
    } else {
        level = static_cast<log_level>(DEFAULT_LOG_LEVEL);
    }
    logger = Log(level);

    try {
        MPI_Init(NULL, NULL);      // initialize MPI environment
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    } catch (std::exception& e) {
        logger << log_level::error << "Error during MPI initialization: " << e.what() << std::endl;
        world_size = 1;
        local_rank = 0;
    }
    logger << log_level::info << "World Size: " << world_size << " Local Rank: " << local_rank << endl;


    design_libname = std::string(argv[3]);

    logger << log_level::verbose << "Design DLL     : " << design_libname << std::endl;
    logger << log_level::verbose << "Sim Engine DLL : " << simengine_libname << std::endl;
    logger << log_level::verbose << "Library path: " << std::getenv("LD_LIBRARY_PATH") << std::endl;
    std::string wdb_name = filesystem::current_path().string() + "/waveform_rank" + std::to_string(local_rank) + ".wdb";
    logger << log_level::verbose << "Waveform : " << wdb_name << std::endl;
    XSI_DUT dut(design_libname, simengine_libname, "ap_rst_n", true, "ap_clk", 4, wdb_name, logger);
    logger << log_level::info << "DUT initialized" << std::endl;

    string eth_type = argv[1];
    unsigned int starting_port = atoi(argv[2]);

    bool krnl_loopback = false;
    if(argc == 5 && string(argv[4]) == "loopback"){
        krnl_loopback = true;
    }

    zmq_intf_context ctx = zmq_intf(starting_port, local_rank, world_size, krnl_loopback, logger);

    int status = 0;

    Stream<unsigned int> axilite_rd_addr;
    Stream<unsigned int> axilite_rd_data;
    Stream<unsigned int> axilite_wr_addr;
    Stream<unsigned int> axilite_wr_data;
    Stream<ap_uint<64> > aximm_rd_addr;
    Stream<ap_uint<512> > aximm_rd_data;
    Stream<ap_uint<64> > aximm_wr_addr;
    Stream<ap_uint<512> > aximm_wr_data;
    Stream<ap_uint<64> > aximm_wr_strb;
    Stream<unsigned int, 16> callreq; //need some capacity for all args
    Stream<unsigned int> callack;

    Stream<stream_word> eth_tx_data;
    Stream<stream_word> eth_rx_data;

    Stream<stream_word> krnl_tx_data;
    Stream<stream_word> krnl_rx_data;

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
                                    aximm_rd_addr, aximm_rd_data,
                                    aximm_wr_addr, aximm_wr_data, aximm_wr_strb,
                                    callreq, callack,
                                    eth_tx_data, eth_rx_data,
                                    krnl_tx_data, krnl_rx_data);
        HLSLIB_DATAFLOW_FUNCTION(zmq_cmd_server,  &ctx,
                                    axilite_rd_addr, axilite_rd_data,
                                    axilite_wr_addr, axilite_wr_data,
                                    aximm_rd_addr, aximm_rd_data,
                                    aximm_wr_addr, aximm_wr_data, aximm_wr_strb,
                                    callreq, callack);
        //ZMQ to other nodes process(es)
        HLSLIB_DATAFLOW_FUNCTION(zmq_eth_egress_server, &ctx, eth_tx_data, local_rank, eth_type == "tcp");
        HLSLIB_DATAFLOW_FUNCTION(zmq_eth_ingress_server, &ctx, eth_rx_data);
        HLSLIB_DATAFLOW_FUNCTION(zmq_krnl_egress_server, &ctx, krnl_tx_data);
        HLSLIB_DATAFLOW_FUNCTION(zmq_krnl_ingress_server, &ctx, krnl_rx_data);
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
