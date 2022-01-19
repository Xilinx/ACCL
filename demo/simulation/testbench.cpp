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
#include <cstring>
#include <iostream>
#include "xsi_loader.h"
#include <signal.h>
#include <zmqpp/zmqpp.hpp>
#include <jsoncpp/json/json.h>
#include <vector>
#include <chrono>
#include <thread>
#include <mpi.h>
#include "ccl_offload_control.h"

using namespace std;

#define CLK_HALF_PERIOD 2000

// See xsi.h header for more details on how Verilog values are stored as aVal/bVal pairs
// constants 
const s_xsi_vlog_logicval logic_one  = {0X00000001, 0X00000000};
const s_xsi_vlog_logicval logic_zero = {0X00000000, 0X00000000};

//global instance of the XSI object
Xsi::Loader *Xsi_Instance;

const s_xsi_vlog_logicval to_xsi_logic_single(unsigned int x){
    const s_xsi_vlog_logicval logic = {x, 0X00000000};
    return logic;
}

void append_logic_val_bit_to_string(std::string& retVal, int aVal, int bVal)
{
     if(aVal == 0) {
        if(bVal == 0) {
           retVal +="0";
        } else {
           retVal +="Z";
        }
     } else { // aVal == 1
        if(bVal == 0) {
           retVal +="1";
        } else {
           retVal +="X";
        }
     }
}

void append_logic_val_to_string(std::string& retVal, int aVal, int bVal, int max_bits)
{
   int bit_mask = 0X00000001;
   int aVal_bit, bVal_bit;
   for(int k=max_bits; k>=0; k--) {
      aVal_bit = (aVal >> k ) & bit_mask;
      bVal_bit = (bVal >> k ) & bit_mask;
      append_logic_val_bit_to_string(retVal, aVal_bit, bVal_bit);
   }
}

std::string logic_val_to_string(s_xsi_vlog_logicval* value, int size)
{
   std::string retVal;

   int num_words = size/32 + 1;
   int max_lastword_bit = size %32 - 1;

   // last word may have unfilled bits
   int  aVal = value[num_words -1].aVal;
   int  bVal = value[num_words -1].bVal;
   append_logic_val_to_string(retVal, aVal, bVal, max_lastword_bit);
   
   // this is for fully filled 32 bit aVal/bVal structs
   for(int k = num_words - 2; k>=0; k--) {
      aVal = value[k].aVal;
      bVal = value[k].bVal;
      append_logic_val_to_string(retVal, aVal, bVal, 31);
   }
   return retVal;
}

unsigned int single_logic_val_to_uint(s_xsi_vlog_logicval* value, unsigned int nbits)
{
    int mask = (nbits >= 32) ? -1 : ((1 << nbits) - 1);
    // last word may have unfilled bits
    unsigned int aVal = value[0].aVal;
    return aVal & mask;
}

//run a specified number of cycles, 
//assuming we're called on negedge
//we also finish on negedge
//if all clock manipulation is done with run_ncycles
//we should get smooth, periodic clocks
//and all data will change on negedge, 
//avoiding race conditions on posedge
void run_ncycles(int ncycles){
    int clk = Xsi_Instance->get_port_number("ap_clk");
    for(int i=0; i<ncycles; i++){
        Xsi_Instance->run(CLK_HALF_PERIOD);
        Xsi_Instance->put_value(clk, &logic_one);
        Xsi_Instance->run(CLK_HALF_PERIOD);
        Xsi_Instance->put_value(clk, &logic_zero);
    }
}

void write_callreq_axis(){

}

void read_callack_axis(){

}

unsigned int cfgmem_read(unsigned int addr){
    run_ncycles(1);//we're now at negedge
    int araddr = Xsi_Instance->get_port_number("s_axi_control_araddr");
    int arvalid = Xsi_Instance->get_port_number("s_axi_control_arvalid");
    int arready = Xsi_Instance->get_port_number("s_axi_control_arready");
    int rready = Xsi_Instance->get_port_number("s_axi_control_rready");
    int rvalid = Xsi_Instance->get_port_number("s_axi_control_rvalid");
    int rdata = Xsi_Instance->get_port_number("s_axi_control_rdata");
    s_xsi_vlog_logicval tmp;
    //set araddr = addr
    tmp = to_xsi_logic_single(addr);
    Xsi_Instance->put_value(araddr, &tmp);
    //set arvalid = 1
    Xsi_Instance->put_value(arvalid, &logic_one);
    //advance 1 clock period until arready == 1
    do{
        Xsi_Instance->get_value(arready, &tmp);
        run_ncycles(1);
    } while(single_logic_val_to_uint(&tmp,1) == 0);
    //set arvalid = 0
    Xsi_Instance->put_value(arvalid, &logic_zero);
    //set rready = 1
    Xsi_Instance->put_value(rready, &logic_one);
    //advance 1 clock period until rvalid == 1
    run_ncycles(1);
    do{
        Xsi_Instance->get_value(rvalid, &tmp);
        if(single_logic_val_to_uint(&tmp,1) == 1){
            Xsi_Instance->get_value(rdata, &tmp);
            //set rready = 0
            Xsi_Instance->put_value(rready, &logic_zero);
            //TODO: check rlast == 1 (?)
            //return value of rdata
            cout << single_logic_val_to_uint(&tmp,32) << endl;
            return single_logic_val_to_uint(&tmp,32);
        }
        run_ncycles(1);
    } while(true);
}

void cfgmem_write(unsigned int addr, unsigned int val){
    run_ncycles(1);//we're now at negedge
    int awaddr = Xsi_Instance->get_port_number("s_axi_control_awaddr");
    int awvalid = Xsi_Instance->get_port_number("s_axi_control_awvalid");
    int awready = Xsi_Instance->get_port_number("s_axi_control_awready");
    int wready = Xsi_Instance->get_port_number("s_axi_control_wready");
    int wvalid = Xsi_Instance->get_port_number("s_axi_control_wvalid");
    int wdata = Xsi_Instance->get_port_number("s_axi_control_wdata");
    int bready = Xsi_Instance->get_port_number("s_axi_control_bready");
    int bvalid = Xsi_Instance->get_port_number("s_axi_control_bvalid");
    int bresp = Xsi_Instance->get_port_number("s_axi_control_bresp");
    s_xsi_vlog_logicval tmp;
    //set awaddr = addr
    tmp = to_xsi_logic_single(addr);
    Xsi_Instance->put_value(awaddr, &tmp);
    //set awvalid = 1
    Xsi_Instance->put_value(awvalid, &logic_one);
    //advance 1 clock period until awready == 1
    do{
        Xsi_Instance->get_value(awready, &tmp);
        run_ncycles(1);
    } while(single_logic_val_to_uint(&tmp,1) == 0);
    //set awvalid = 0
    Xsi_Instance->put_value(awvalid, &logic_zero);
    //set wdata = val
    tmp = to_xsi_logic_single(val);
    Xsi_Instance->put_value(wdata, &tmp);
    //set wvalid = 1
    Xsi_Instance->put_value(wvalid, &logic_one);
    //advance 1 clock period until wready == 1
    do{
        Xsi_Instance->get_value(wready, &tmp);
        run_ncycles(1);
    } while(single_logic_val_to_uint(&tmp,1) == 0);
    //set wvalid = 0
    Xsi_Instance->put_value(wvalid, &logic_zero);
    //set bready = 1
    Xsi_Instance->put_value(bready, &logic_one);
    //advance 1 clock period until bvalid == 1
    run_ncycles(1);
    do{
        Xsi_Instance->get_value(bvalid, &tmp);
        if(single_logic_val_to_uint(&tmp,1) == 1){
            Xsi_Instance->get_value(bresp, &tmp);
            //set bready = 0
            Xsi_Instance->put_value(bready, &logic_zero);
            //TODO: check bresp 
            return;
        }
        run_ncycles(1);
    } while(true);
}

void call_req_push(unsigned int val, bool last){
    run_ncycles(1);
    int tdata = Xsi_Instance->get_port_number("s_axis_call_req_tdata");
    int tvalid = Xsi_Instance->get_port_number("s_axis_call_req_tvalid");
    int tlast = Xsi_Instance->get_port_number("s_axis_call_req_tlast");
    int tready = Xsi_Instance->get_port_number("s_axis_call_req_tready");
    //set valid and last
    Xsi_Instance->put_value(tvalid, &logic_one);
    Xsi_Instance->put_value(tlast, last ? &logic_one : &logic_zero);
    s_xsi_vlog_logicval tmp = to_xsi_logic_single(val);
    Xsi_Instance->put_value(tdata, &tmp);
    //advance until ready == 1
    do{
        Xsi_Instance->get_value(tready, &tmp);
        run_ncycles(1);
    } while(single_logic_val_to_uint(&tmp,1) == 0);
    Xsi_Instance->put_value(tvalid, &logic_zero);
    Xsi_Instance->put_value(tlast, &logic_zero);
}

void call_ack_pop(){
    run_ncycles(1);
    int tvalid = Xsi_Instance->get_port_number("m_axis_call_ack_tvalid");
    int tready = Xsi_Instance->get_port_number("m_axis_call_ack_tready");
    s_xsi_vlog_logicval tmp;
    //advance until valid == 1
    do{
        Xsi_Instance->get_value(tvalid, &tmp);
        run_ncycles(1);
    } while(single_logic_val_to_uint(&tmp,1) == 0);
    //set ready
    Xsi_Instance->put_value(tready, &logic_one);
    run_ncycles(1);
    Xsi_Instance->put_value(tready, &logic_zero);
}

void serve_zmq(zmqpp::socket &socket){

    Json::Reader reader;
    Json::StreamWriterBuilder builder;

    // receive the message
    zmqpp::message message;
    // decompose the message 
    if(!socket.receive(message, true)){
        //run sim for 1us while waiting for a message
#ifdef ZMQ_CALL_VERBOSE
        cout << "No command received, running for 1us" << endl;
#endif
        run_ncycles(250);
        return;
    }

    string msg_text;
    message >> msg_text;//message now is in a string

#ifdef ZMQ_CALL_VERBOSE
    cout << "Received: " << msg_text << endl;
#endif

    //parse msg_text as json
    Json::Value request;
    reader.parse(msg_text, request); // reader can also read strings

    //parse message and reply
    Json::Value response;
    response["status"] = 0;
    int adr, val, len;
    uint64_t dma_addr;
    Json::Value dma_wdata;
    switch(request["type"].asUInt()){
        // MMIO read request  {"type": 0, "addr": <uint>}
        // MMIO read response {"status": OK|ERR, "rdata": <uint>}
        case 0:
            adr = request["addr"].asUInt();
#ifdef ZMQ_CALL_VERBOSE
            cout << "MMIO read " << adr << endl;
#endif
            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
                response["rdata"] = 0;
            } else {
                response["rdata"] = cfgmem_read(adr);
            }
            break;
        // MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
        // MMIO write response {"status": OK|ERR}
        case 1:
            adr = request["addr"].asUInt();
#ifdef ZMQ_CALL_VERBOSE
            cout << "MMIO write " << adr << endl;
#endif
            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
            } else {
                cfgmem_write(adr, request["wdata"].asUInt());
            }
            break;
/*
        // Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>}
        // Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
        case 2:
            adr = request["addr"].asUInt();
            len = request["len"].asUInt();
#ifdef ZMQ_CALL_VERBOSE
            cout << "Mem read " << adr << " len: " << len << endl;
#endif
            if((adr+len) > devicemem.size()){
                response["status"] = 1;
                response["rdata"][0] = 0;
            } else {
                for (int i=0; i<len; i++) 
                { 
                    response["rdata"][i] = devicemem.at(adr+i);
                }
            }
            break;
        // Devicemem write request  {"type": 3, "addr": <uint>, "wdata": <array of uint>}
        // Devicemem write response {"status": OK|ERR}
        case 3:
            adr = request["addr"].asUInt();
            dma_wdata = request["wdata"];
            len = dma_wdata.size();
#ifdef ZMQ_CALL_VERBOSE
            cout << "Mem write " << adr << " len: " << len << endl;
#endif
            if((adr+len) > devicemem.size()){
                devicemem.resize(adr+len);
            }
            for(int i=0; i<len; i++){
                devicemem.at(adr+i) = dma_wdata[i].asUInt();
            }
            break;
*/
        // Call request  {"type": 4, arg names and values}
        // Call response {"status": OK|ERR}
        case 4:
#ifdef ZMQ_CALL_VERBOSE
            cout << "Call with scenario " << request["scenario"].asUInt() << endl;
#endif
            call_req_push(request["scenario"].asUInt(), 0);
            call_req_push(request["count"].asUInt(), 0);
            call_req_push(request["comm"].asUInt(), 0);
            call_req_push(request["root_src_dst"].asUInt(), 0);
            call_req_push(request["function"].asUInt(), 0);
            call_req_push(request["tag"].asUInt(), 0);
            call_req_push(request["arithcfg"].asUInt(), 0);
            call_req_push(request["compression_flags"].asUInt(), 0);
            call_req_push(request["stream_flags"].asUInt(), 0);
            dma_addr = request["addr_0"].asUInt64();
            call_req_push((uint32_t)(dma_addr & 0xffffffff), 0);
            call_req_push((uint32_t)(dma_addr >> 32), 0);
            dma_addr = request["addr_1"].asUInt64();
            call_req_push((uint32_t)(dma_addr & 0xffffffff), 0);
            call_req_push((uint32_t)(dma_addr >> 32), 0);
            dma_addr = request["addr_2"].asUInt64();
            call_req_push((uint32_t)(dma_addr & 0xffffffff), 0);
            call_req_push((uint32_t)(dma_addr >> 32), 1);
            //pop the status queue to wait for call completion
            call_ack_pop();
            break;
        default:
#ifdef ZMQ_CALL_VERBOSE
            cout << "Unrecognized message" << endl;
#endif
            response["status"] = 1;

    }
    //return message to client
    string str = Json::writeString(builder, response);
    socket.send(str);
}

void reset_design(){
    int reset = Xsi_Instance->get_port_number("ap_rst_n");
    int araddr = Xsi_Instance->get_port_number("s_axi_control_araddr");
    int arprot = Xsi_Instance->get_port_number("s_axi_control_arprot");
    int arvalid = Xsi_Instance->get_port_number("s_axi_control_arvalid");
    int rready = Xsi_Instance->get_port_number("s_axi_control_rready");
    int awaddr = Xsi_Instance->get_port_number("s_axi_control_awaddr");
    int awprot = Xsi_Instance->get_port_number("s_axi_control_awprot");
    int awvalid = Xsi_Instance->get_port_number("s_axi_control_awvalid");
    int wvalid = Xsi_Instance->get_port_number("s_axi_control_wvalid");
    int wdata = Xsi_Instance->get_port_number("s_axi_control_wdata");
    int wstrb = Xsi_Instance->get_port_number("s_axi_control_wstrb");
    int bready = Xsi_Instance->get_port_number("s_axi_control_bready");
    int callack_tready = Xsi_Instance->get_port_number("m_axis_call_ack_tready");
    int callreq_tdata = Xsi_Instance->get_port_number("s_axis_call_req_tdata");
    int callreq_tvalid = Xsi_Instance->get_port_number("s_axis_call_req_tvalid");
    int callreq_tlast = Xsi_Instance->get_port_number("s_axis_call_req_tlast");
    //set signals to default values
    Xsi_Instance->put_value(araddr, &logic_zero);
    Xsi_Instance->put_value(arprot, &logic_zero);
    Xsi_Instance->put_value(arvalid, &logic_zero);
    Xsi_Instance->put_value(rready, &logic_zero);
    Xsi_Instance->put_value(awaddr, &logic_zero);
    Xsi_Instance->put_value(awprot, &logic_zero);
    Xsi_Instance->put_value(awvalid, &logic_zero);
    Xsi_Instance->put_value(wvalid, &logic_zero);
    Xsi_Instance->put_value(wdata, &logic_zero);
    Xsi_Instance->put_value(wstrb, &logic_zero);
    Xsi_Instance->put_value(bready, &logic_zero);
    Xsi_Instance->put_value(callack_tready, &logic_zero);
    Xsi_Instance->put_value(callreq_tdata, &logic_zero);
    Xsi_Instance->put_value(callreq_tvalid, &logic_zero);
    Xsi_Instance->put_value(callreq_tlast, &logic_zero);
    //set and clear reset
    Xsi_Instance->put_value(reset, &logic_zero);
    run_ncycles(1000);
    Xsi_Instance->put_value(reset, &logic_one);
}

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void finish(int signum) {
    cout << "Caught signal " << signum << endl;
    // Terminate program
    // Just a check to rewind time to 0
    Xsi_Instance->restart();
    exit(signum);
}

int main(int argc, char **argv)
{ 
    std::string simengine_libname = "librdi_simulator_kernel.so";
    std::string design_libname;
    
    design_libname = argv[3];

    std::cout << "Design DLL     : " << design_libname << std::endl;
    std::cout << "Sim Engine DLL : " << simengine_libname << std::endl;

    try {
        Xsi_Instance = new Xsi::Loader(design_libname, simengine_libname);
        std::cout << "Libs loaded" << std::endl;
        s_xsi_setup_info info;
        memset(&info, 0, sizeof(info));
        info.logFileName = NULL;
        char wdbName[] = "test.wdb";
        info.wdbFileName = wdbName;
        Xsi_Instance->open(&info);
        std::cout << "XSI opened" << std::endl;
        Xsi_Instance->trace_all();
        std::cout << "Initial set-up completed" << std::endl;
    } catch (std::exception& e) {
        std::cerr << "ERROR during XSI initialization: " << e.what() << std::endl;
        return -1;
    }

    int world_size; // number of processes
    int local_rank; // the rank of the process
    try {
        MPI_Init(NULL, NULL);      // initialize MPI environment
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    } catch (std::exception& e) {
        std::cerr << "ERROR during MPI initialization: " << e.what() << std::endl;
        world_size = 1;
        local_rank = 0;
    }
    cout << "World Size: " << world_size << " Local Rank: " << local_rank << endl; 

    string eth_type = argv[1];
    unsigned int starting_port = atoi(argv[2]);
    const string endpoint_base = "tcp://127.0.0.1:";

    string cmd_endpoint = endpoint_base + to_string(starting_port + local_rank);
    cout << cmd_endpoint << endl;
    vector<string> eth_endpoints;

    for(int i=0; i<world_size; i++){
        eth_endpoints.emplace_back(endpoint_base + to_string(starting_port+world_size+i));
        cout << eth_endpoints.at(i) << endl;
    }
    
    //ZMQ for commands
    // initialize the 0MQ context
    zmqpp::context context;
    zmqpp::socket cmd_socket(context, zmqpp::socket_type::reply);
    zmqpp::socket eth_tx_socket(context, zmqpp::socket_type::pub);
    zmqpp::socket eth_rx_socket(context, zmqpp::socket_type::sub);
    // bind to the socket(s)
    cout << "Rank " << local_rank << " binding to " << cmd_endpoint << " and " << eth_endpoints.at(local_rank) << endl;
    cmd_socket.bind(cmd_endpoint);
    eth_tx_socket.bind(eth_endpoints.at(local_rank));

    this_thread::sleep_for(chrono::milliseconds(1000));

    // connect to the sockets
    for(int i=0; i<world_size; i++){
        cout << "Rank " << local_rank << " connecting to " << eth_endpoints.at(i) << endl;
        eth_rx_socket.connect(eth_endpoints.at(i));
    }

    this_thread::sleep_for(chrono::milliseconds(1000));

    cout << "Rank " << local_rank << " subscribing to " << local_rank << endl;
    eth_rx_socket.subscribe(to_string(local_rank));

    this_thread::sleep_for(chrono::milliseconds(1000));


    // Ports
    int reset;
    int clk;

    // my variables 
    int count_success = 0;
    int status = 0;

    try {
        reset = Xsi_Instance->get_port_number("ap_rst_n");
        if(reset <0) {
            std::cerr << "ERROR: reset not found" << std::endl;
            exit(1);
        }
        clk = Xsi_Instance->get_port_number("ap_clk");
        if(clk <0) {
            std::cerr << "ERROR: clk not found" << std::endl;
            exit(1);
        }

        // Register signal and signal handler
        signal(SIGINT, finish);

        //set clock/reset to initial values
        Xsi_Instance->put_value(reset, &logic_zero);
        Xsi_Instance->put_value(clk, &logic_zero);

        //reset design and let it run for a while to initialize
        cout << "Resetting design" << endl;
        reset_design();
        cout << "Running design initialization" << endl;
        run_ncycles(90000);
        cout << "Initialization done" << endl;

        while(true){
            cout << "Waiting for command" << endl;
            serve_zmq(cmd_socket);
        }

    }
    catch (std::exception& e) {
        std::cerr << "ERROR: An exception occurred: " << e.what() << std::endl;
        status = 2;
    }
    catch (...) {
        std::cerr << "ERROR: An unknown exception occurred." << std::endl;
        status = 3;
    }

    if(status == 0) {
        std::cout << "PASSED test\n";
    } else {
        std::cout << "FAILED test\n";
    }

    exit(status);
}


