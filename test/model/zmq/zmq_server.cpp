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

#include "zmq_server.h"
#include <iostream>
#include <chrono>
#include <thread>
#include "ccl_offload_control.h"

using namespace std;
using namespace hlslib;

namespace {
    Log *logger;
}

zmq_intf_context zmq_server_intf(unsigned int starting_port, unsigned int local_rank, unsigned int world_size, bool kernel_loopback, Log &log)
{
    zmq_intf_context ctx;

    logger = &log;
    ctx.cmd_socket = std::make_unique<zmqpp::socket>(ctx.context, zmqpp::socket_type::reply);
    ctx.eth_tx_socket = std::make_unique<zmqpp::socket>(ctx.context, zmqpp::socket_type::pub);
    ctx.eth_rx_socket = std::make_unique<zmqpp::socket>(ctx.context, zmqpp::socket_type::sub);
    ctx.krnl_tx_socket = std::make_unique<zmqpp::socket>(ctx.context, zmqpp::socket_type::pub);
    ctx.krnl_rx_socket = std::make_unique<zmqpp::socket>(ctx.context, zmqpp::socket_type::sub);

    const string endpoint_base = "tcp://127.0.0.1:";

    string cmd_endpoint = endpoint_base + to_string(starting_port + local_rank);
    *logger << log_level::verbose << "Endpoint: " << cmd_endpoint << endl;
    vector<string> eth_endpoints;

    for(int i=0; i<world_size; i++){
        eth_endpoints.emplace_back(endpoint_base + to_string(starting_port+world_size+i));
        *logger << log_level::verbose << "Endpoint rank " << i << ": " << eth_endpoints.at(i) << endl;
    }

    // bind to the socket(s)
    *logger << log_level::verbose << "Rank " << local_rank << " binding to " << cmd_endpoint << " (CMD) and " << eth_endpoints.at(local_rank) << " (ETH)" << endl;
    ctx.cmd_socket->bind(cmd_endpoint);
    ctx.eth_tx_socket->bind(eth_endpoints.at(local_rank));

    this_thread::sleep_for(chrono::milliseconds(1000));

    // connect to the sockets
    for(int i=0; i<world_size; i++){
        *logger << log_level::verbose << "Rank " << local_rank << " connecting to " << eth_endpoints.at(i) << " (ETH)" << endl;
        ctx.eth_rx_socket->connect(eth_endpoints.at(i));
    }

    this_thread::sleep_for(chrono::milliseconds(1000));

    *logger << log_level::verbose << "Rank " << local_rank << " subscribing to " << local_rank << " (ETH)" << endl;
    // Create a padded version of the rank to prevent subscription to
    // ranks that have the same starting digits
    std::stringstream rank_pad;
    rank_pad << std::setw(DEST_PADDING) << std::setfill('0') << local_rank; 
    ctx.eth_rx_socket->subscribe(rank_pad.str());

    this_thread::sleep_for(chrono::milliseconds(1000));

    //kernel interface
    //bind to tx socket
    string krnl_endpoint = endpoint_base + to_string(starting_port+2*world_size+local_rank);
    *logger << log_level::verbose << "Rank " << local_rank << " binding to " << krnl_endpoint << " (KRNL)" << endl;
    ctx.krnl_tx_socket->bind(krnl_endpoint);
    this_thread::sleep_for(chrono::milliseconds(1000));
    //connect to rx socket
    //in case we want to loopback the kernel interface, we connect to the TX sockets for RX
    if(!kernel_loopback){
        krnl_endpoint = endpoint_base + to_string(starting_port+3*world_size+local_rank);
    }
    *logger << log_level::verbose << "Rank " << local_rank << " connecting to " << krnl_endpoint << " (KRNL)" << endl;
    ctx.krnl_rx_socket->connect(krnl_endpoint);
    this_thread::sleep_for(chrono::milliseconds(1000));
    //subscribing to all (for now)
    *logger << log_level::verbose << "Rank " << local_rank << " subscribing to all (KRNL)" << endl;
    ctx.krnl_rx_socket->subscribe("");
    this_thread::sleep_for(chrono::milliseconds(1000));

    *logger << log_level::info << "ZMQ Context established for rank " << local_rank << endl;

    return ctx;
}

void eth_endpoint_egress_port(zmq_intf_context *ctx, Stream<stream_word > &in, unsigned int local_rank){

    zmqpp::message message;
    Json::Value packet;
    Json::StreamWriterBuilder builder;

    if(in.IsEmpty()) return;
    //pop first word in packet
    stringstream ss;
    unsigned int dest;
    stream_word tmp;
    //get the data (bytes valid from tkeep)
    unsigned int idx=0;
    do{
        tmp = in.Pop();
        for(int i=0; i<64; i++){
            if(tmp.keep(i,i) == 1){
                packet["data"][idx++] = (unsigned int)tmp.data(8*(i+1)-1,8*i);
            }
        }
    }while(tmp.last == 0);
    dest = tmp.dest;
    // Create a padded version of he destitnation port ID to ensure unique string
    // for each rank
    std::stringstream dest_pad;
    dest_pad << std::setw(DEST_PADDING) << std::setfill('0') << dest; 
    //first part of the message is the destination port ID
    message << dest_pad.str();
    //second part of the message is the local rank of the sender
    message << to_string(local_rank);
    //finally package the data
    string str = Json::writeString(builder, packet);
    message << str;
    *logger << log_level::verbose << "ETH Send " << idx << " bytes to " << dest << endl;
    *logger << log_level::debug << str << endl;
    ctx->eth_tx_socket->send(message);
    //add some spacing to encourage realistic
    //interleaving between messsages in fabric
    this_thread::sleep_for(chrono::milliseconds(10));
}

void eth_endpoint_ingress_port(zmq_intf_context *ctx, Stream<stream_word > &out){

    Json::Reader reader;

    // receive the message
    zmqpp::message message;
    if(!ctx->eth_rx_socket->receive(message, true)) return;

    // decompose the message
    string msg_text, dst_text, src_text, sender_rank_text;

    //get and check destination ID
    message >> dst_text;
    message >> sender_rank_text;
    message >> msg_text;

    //parse msg_text as json
    Json::Value packet, data;
    reader.parse(msg_text, packet);

    data = packet["data"];
    unsigned int len = data.size();

    stream_word tmp;
    int idx = 0;
    while(idx<len){
        for(int i=0; i<64; i++){
            if(idx<len){
                tmp.data(8*(i+1)-1,8*i) = data[idx++].asUInt();
                tmp.keep(i,i) = 1;
            } else{
                tmp.keep(i,i) = 0;
            }
        }
        tmp.last = (idx == len);
        tmp.dest = stoi(sender_rank_text);
        out.Push(tmp);
    }

    *logger << log_level::verbose << "ETH Receive " << len << " bytes from " << sender_rank_text << endl;
    *logger << log_level::debug << msg_text << endl;

}

void krnl_endpoint_egress_port(zmq_intf_context *ctx, Stream<stream_word > &in){

    zmqpp::message message;
    Json::Value packet;
    Json::StreamWriterBuilder builder;

    if(in.IsEmpty()) return;
    //pop first word in packet
    unsigned int dest;
    stream_word tmp;

    //get the data (bytes valid from tkeep)
    unsigned int idx=0;
    do{
        if(ctx->stop) return;
        tmp = in.Pop();
        for(int i=0; i<64; i++){
            if(tmp.keep(i,i) == 1){
                packet["data"][idx++] = (unsigned int)tmp.data(8*(i+1)-1,8*i);
            }
        }
    }while(tmp.last == 0);

    //first part of the message is the destination port ID
    dest = tmp.dest;
    message << to_string(dest);

    //finally package the data
    string str = Json::writeString(builder, packet);
    message << str;
    *logger << log_level::verbose << "CCLO to user kernel: push " << idx << " bytes to dest = " << dest << endl;
    *logger << log_level::debug << str << endl;
    if(!ctx->stop){
        ctx->krnl_tx_socket->send(message);
    }
}


void krnl_endpoint_ingress_port(zmq_intf_context *ctx, Stream<stream_word > &out){

    Json::Reader reader;

    // receive the message
    zmqpp::message message;
    if(!ctx->krnl_rx_socket->receive(message, true)) return;

    // decompose the message
    string msg_text, dst_text;

    //get and check destination ID
    message >> dst_text;
    message >> msg_text;

    //parse msg_text as json
    Json::Value packet, data;
    reader.parse(msg_text, packet);

    data = packet["data"];
    unsigned int len = data.size();

    stream_word tmp;
    int idx = 0;
    while(idx<len){
        if(ctx->stop) return;
        for(int i=0; i<64; i++){
            if(idx<len){
                tmp.data(8*(i+1)-1,8*i) = data[idx++].asUInt();
                tmp.keep(i,i) = 1;
            } else{
                tmp.keep(i,i) = 0;
            }
        }
        tmp.last = (idx == len);
        tmp.dest = stoi(dst_text);
        out.Push(tmp);
    }

    *logger << log_level::verbose << "User kernel to CCLO: push " << len << " bytes" << endl;
    *logger << log_level::debug << msg_text << endl;

}

void serve_zmq(zmq_intf_context *ctx, uint32_t *cfgmem, vector<char> &devicemem, vector<char> &hostmem, Stream<command_word> cmd[NUM_CTRL_STREAMS], Stream<command_word> sts[NUM_CTRL_STREAMS]){

    Json::Reader reader;
    Json::StreamWriterBuilder builder;

    // receive the message
    zmqpp::message message;
    if(!ctx->cmd_socket->receive(message, true)) return;

    // decompose the message
    string msg_text;
    message >> msg_text; //message now is in a string

    *logger << log_level::debug << "Received: " << msg_text << endl;

    //parse msg_text as json
    Json::Value request;
    reader.parse(msg_text, request); // reader can also read strings

    //parse message and reply
    Json::Value response;
    response["status"] = 0;
    int adr, val, len;
    bool host;
    uint64_t dma_addr;
    Json::Value dma_wdata;
    unsigned int ctrl_id;

    switch(request["type"].asUInt()){
        // MMIO read request  {"type": 0, "addr": <uint>}
        // MMIO read response {"status": OK|ERR, "rdata": <uint>}
        case 0:
            adr = request["addr"].asUInt();
            *logger << log_level::debug << "MMIO read " << adr << endl;

            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
                response["rdata"] = 0;
            } else {
                response["rdata"] = cfgmem[adr/4];
            }
            break;
        // MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
        // MMIO write response {"status": OK|ERR}
        case 1:
            adr = request["addr"].asUInt();
            *logger << log_level::debug << "MMIO write " << adr << endl;
            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
            } else {
                cfgmem[adr/4] = request["wdata"].asUInt();
            }
            break;
        // Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>, "host": <bool>}
        // Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
        case 2:
            adr = request["addr"].asUInt();
            len = request["len"].asUInt();
            host = request["host"].asUInt();
            *logger << log_level::debug << (host ? "Host " : "Device ") << " mem read " << adr << " len: " << len << endl;
            if(host){
                if((adr+len) > hostmem.size()){
                    response["status"] = 1;
                    response["rdata"][0] = 0;
                    *logger << log_level::error << "Host mem read outside allocated range ("<< hostmem.size()/1024 << "KB) at addr: " << adr << " len: " << len << endl;
                } else {
                    for (int i=0; i<len; i++)
                    {
                        response["rdata"][i] = hostmem.at(adr+i);
                    }
                }
            } else {
                if((adr+len) > devicemem.size()){
                    response["status"] = 1;
                    response["rdata"][0] = 0;
                    *logger << log_level::error << "Device mem read outside allocated range ("<< devicemem.size()/1024 << "KB) at addr: " << adr << " len: " << len << endl;
                } else {
                    for (int i=0; i<len; i++)
                    {
                        response["rdata"][i] = devicemem.at(adr+i);
                    }
                }
            }
            break;
        // Devicemem write request  {"type": 3, "addr": <uint>, "wdata": <array of uint>, "host": <bool>}
        // Devicemem write response {"status": OK|ERR}
        case 3:
            adr = request["addr"].asUInt();
            dma_wdata = request["wdata"];
            len = dma_wdata.size();
            host = request["host"].asUInt();
            *logger << log_level::debug << (host ? "Host " : "Device ") << " mem write " << adr << " len: " << len << endl;
            if(host){
                if((adr+len) > hostmem.size()){
                    hostmem.resize(adr+len);
                }
                for(int i=0; i<len; i++){
                    hostmem.at(adr+i) = dma_wdata[i].asUInt();
                }
            } else {
                if((adr+len) > devicemem.size()){
                    devicemem.resize(adr+len);
                }
                for(int i=0; i<len; i++){
                    devicemem.at(adr+i) = dma_wdata[i].asUInt();
                }
            }
            break;
        // Devicemem allocate request  {"type": 4, "addr": <uint>, "len": <uint>, "host": <bool>}
        // Devicemem allocate response {"status": OK|ERR}
        case 4:
            adr = request["addr"].asUInt();
            len = request["len"].asUInt();
            host = request["host"].asUInt();
            *logger << log_level::debug << (host ? "Host " : "Device ") << " mem allocate " << adr << " len: " << len << endl;
            if(host){
                if((adr+len) > hostmem.size()){
                    hostmem.resize(adr+len);
                }
            } else {
                if((adr+len) > devicemem.size()){
                    devicemem.resize(adr+len);
                }
            }
            break;
        // Call request  {"type": 5, arg names and values}
        // Call response {"status": OK|ERR}
        case 5:
            ctrl_id = request["ctrl_id"].asUInt();
            *logger << log_level::verbose << "Call with scenario " << request["scenario"].asUInt() << " on controller " << ctrl_id << endl;
            if(ctrl_id >= NUM_CTRL_STREAMS){
                response["status"] = -1;
            } else{
                cmd[ctrl_id].Push((command_word){.data=request["scenario"].asUInt(), .last=0});
                cmd[ctrl_id].Push((command_word){.data=request["count"].asUInt(), .last=0});
                cmd[ctrl_id].Push((command_word){.data=request["comm"].asUInt(), .last=0});
                cmd[ctrl_id].Push((command_word){.data=request["root_src_dst"].asUInt(), .last=0});
                cmd[ctrl_id].Push((command_word){.data=request["function"].asUInt(), .last=0});
                cmd[ctrl_id].Push((command_word){.data=request["tag"].asUInt(), .last=0});
                cmd[ctrl_id].Push((command_word){.data=request["arithcfg"].asUInt(), .last=0});
                cmd[ctrl_id].Push((command_word){.data=request["compression_flags"].asUInt(), .last=0});
                cmd[ctrl_id].Push((command_word){.data=request["stream_flags"].asUInt(), .last=0});
                dma_addr = request["addr_0"].asUInt64();
                cmd[ctrl_id].Push((command_word){.data=(uint32_t)(dma_addr & 0xffffffff), .last=0});
                cmd[ctrl_id].Push((command_word){.data=(uint32_t)(dma_addr >> 32), .last=0});
                dma_addr = request["addr_1"].asUInt64();
                cmd[ctrl_id].Push((command_word){.data=(uint32_t)(dma_addr & 0xffffffff), .last=0});
                cmd[ctrl_id].Push((command_word){.data=(uint32_t)(dma_addr >> 32), .last=0});
                dma_addr = request["addr_2"].asUInt64();
                cmd[ctrl_id].Push((command_word){.data=(uint32_t)(dma_addr & 0xffffffff), .last=0});
                cmd[ctrl_id].Push((command_word){.data=(uint32_t)(dma_addr >> 32), .last=1});
                response["status"] = 0;
            }
            break;
        // Call wait request  {"type": 6, arg controller id}
        // Call wait response {"status": OK|NOK|ERR}
        case 6:
            ctrl_id = request["ctrl_id"].asUInt();
            *logger << log_level::debug << "Call wait on controller " << ctrl_id << endl;
            //pop the corresponding status queue if possible
            if(ctrl_id >= NUM_CTRL_STREAMS){
                response["status"] = -1;
            } else if(sts[ctrl_id].IsEmpty()){
                response["status"] = 1;
            } else {
                sts[ctrl_id].Pop();
                response["status"] = 0;
            }
            break;
        default:
            (*logger)("Unrecognized message\n", log_level::warning);
            response["status"] = 1;
    }
    //return message to client
    string str = Json::writeString(builder, response);
    ctx->cmd_socket->send(str);
}


void serve_zmq(zmq_intf_context *ctx,
                Stream<unsigned int> &axilite_rd_addr, Stream<unsigned int> &axilite_rd_data,
                Stream<unsigned int> &axilite_wr_addr, Stream<unsigned int> &axilite_wr_data,
                Stream<ap_uint<64> > &aximm_rd_addr, Stream<ap_uint<32> > &aximm_rd_len, Stream<ap_uint<512> > &aximm_rd_data,
                Stream<ap_uint<64> > &aximm_wr_addr, Stream<ap_uint<32> > &aximm_wr_len, Stream<ap_uint<512> > &aximm_wr_data, Stream<ap_uint<64> > &aximm_wr_strb,
                Stream<unsigned int> &callreq, Stream<unsigned int> &callack){

    Json::Reader reader;
    Json::StreamWriterBuilder builder;

    // receive the message
    zmqpp::message message;
    if(!ctx->cmd_socket->receive(message, true)) return;

    // decompose the message
    string msg_text;
    message >> msg_text;//message now is in a string

    *logger << log_level::debug << "Received: " << msg_text << endl;

    //parse msg_text as json
    Json::Value request;
    reader.parse(msg_text, request); // reader can also read strings

    //parse message and reply
    Json::Value response;
    response["status"] = 0;
    int adr, val, len;
    bool host;
    uint64_t dma_addr;
    Json::Value dma_wdata;
    ap_uint<64> mem_addr;
    ap_uint<512> mem_data;
    ap_uint<64> mem_strb;
    unsigned int ctrl_id;
    unsigned int ctrl_offset;

    switch(request["type"].asUInt()){
        // MMIO read request  {"type": 0, "addr": <uint>}
        // MMIO read response {"status": OK|ERR, "rdata": <uint>}
        case 0:
            adr = request["addr"].asUInt();
            *logger << log_level::debug << "MMIO read " << adr << endl;
            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
                response["rdata"] = 0;
            } else {
                axilite_rd_addr.Push(adr);
                while(!ctx->stop){
                    if(!axilite_rd_data.IsEmpty()){
                        response["rdata"] = axilite_rd_data.Pop();
                        break;
                    } else{
                        this_thread::sleep_for(chrono::milliseconds(1));
                    }
                }
            }
            break;
        // MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
        // MMIO write response {"status": OK|ERR}
        case 1:
            adr = request["addr"].asUInt();
            *logger << log_level::debug << "MMIO write " << adr << endl;
            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
            } else {
                while(!ctx->stop){
                    if(!axilite_wr_addr.IsFull() && !axilite_wr_data.IsFull()){
                        axilite_wr_addr.Push(adr);
                        axilite_wr_data.Push(request["wdata"].asUInt());
                        break;
                    } else{
                        this_thread::sleep_for(chrono::milliseconds(1));
                    }
                }
            }
            break;
        // Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>, "host": <bool>}
        // Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
        case 2:
            adr = request["addr"].asUInt();
            len = request["len"].asUInt();
            host = request["host"].asUInt();
            *logger << log_level::debug << "Mem read " << adr << " len: " << len << endl;
            if((adr+len) > (host ? 1 : ACCL_SIM_NUM_BANKS)*ACCL_SIM_MEM_SIZE_KB*1024){
                response["status"] = 1;
                response["rdata"][0] = 0;
                *logger << log_level::error << "Mem read outside available range ("<< ACCL_SIM_MEM_SIZE_KB << "KB) at addr: " << adr << " len: " << len << endl;
            } else {
                adr += host ? ACCL_SIM_NUM_BANKS*ACCL_SIM_MEM_SIZE_KB*1024 : 0;
                aximm_rd_addr.Push(adr);
                aximm_rd_len.Push(len);
                unsigned int idx = 0;
                while(!ctx->stop && len>idx){
                    if(!aximm_rd_data.IsEmpty()){
                        mem_data = aximm_rd_data.Pop();
                        for(int j=0; j<64 && len>idx; j++, idx++){
                            response["rdata"][idx] = (unsigned int)mem_data(8*(j+1)-1, 8*j);
                        }
                    } else{
                        this_thread::sleep_for(chrono::milliseconds(1));
                    }
                }
            }
            break;
        // Devicemem write request  {"type": 3, "addr": <uint>, "wdata": <array of uint>, "host": <bool>}
        // Devicemem write response {"status": OK|ERR}
        case 3:
            adr = request["addr"].asUInt();
            dma_wdata = request["wdata"];
            len = dma_wdata.size();
            host = request["host"].asUInt();
            *logger << log_level::debug << "Mem write " << adr << " len: " << len << endl;
            if((adr+len) > (host ? 1 : ACCL_SIM_NUM_BANKS)*ACCL_SIM_MEM_SIZE_KB*1024){
                response["status"] = 1;
                *logger << log_level::error << "Mem write outside available range ("<< ACCL_SIM_MEM_SIZE_KB << "KB) at addr: " << adr << " len: " << len << endl;
            } else{
                adr += host ? ACCL_SIM_NUM_BANKS*ACCL_SIM_MEM_SIZE_KB*1024 : 0;
                aximm_wr_addr.Push(adr);
                aximm_wr_len.Push(len);
                for(int i=0; i<len; i+=64){
                    mem_strb = 0;
                    for(int j=0; j<64 && (i+j)<len; j++){
                        mem_data(8*(j+1)-1, 8*j) = dma_wdata[i+j].asUInt();
                        mem_strb(j,j) = 1;
                    }
                    while(!ctx->stop){
                        if(!aximm_wr_data.IsFull() && !aximm_wr_strb.IsFull()){
                            aximm_wr_data.Push(mem_data);
                            aximm_wr_strb.Push(mem_strb);
                            break;
                        } else{
                            this_thread::sleep_for(chrono::milliseconds(1));
                        }
                    }
                }
            }
            break;
        // Devicemem allocate request  {"type": 4, "addr": <uint>, "len": <uint>, "host": <bool>}
        // Devicemem allocate response {"status": OK|ERR}
        case 4:
            adr = request["addr"].asUInt();
            len = request["len"].asUInt();
            host = request["host"].asUInt();
            *logger << log_level::debug << (host ? "Host " : "Device ") << " mem allocate " << adr << " len: " << len << endl;
            if((adr+len) > (host ? 1 : ACCL_SIM_NUM_BANKS)*ACCL_SIM_MEM_SIZE_KB*1024){
                response["status"] = 1;
                *logger << log_level::error << "Mem allocate outside available range ("<< ACCL_SIM_MEM_SIZE_KB << "KB) at addr: " << adr << " len: " << len << endl;
            }
            break;
        // Call start request  {"type": 5, controller id, arg values}
        // Call start response {"status": OK|ERR}
        case 5:
            ctrl_id = request["ctrl_id"].asUInt();
            *logger << log_level::verbose << "Call with scenario " << request["scenario"].asUInt() << " on control stream " << ctrl_id << endl;
            if(ctrl_id >= NUM_CTRL_STREAMS){
                response["status"] = -1;
            }else if(ctrl_id < 2){
                //start a hostcontroller
                ctrl_offset = 0x2000 * (ctrl_id+1);
                axilite_wr_addr.Push(ctrl_offset+0x10);
                axilite_wr_data.Push(request["scenario"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x18);
                axilite_wr_data.Push(request["count"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x20);
                axilite_wr_data.Push(request["comm"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x28);
                axilite_wr_data.Push(request["root_src_dst"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x30);
                axilite_wr_data.Push(request["function"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x38);
                axilite_wr_data.Push(request["tag"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x40);
                axilite_wr_data.Push(request["arithcfg"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x48);
                axilite_wr_data.Push(request["compression_flags"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x50);
                axilite_wr_data.Push(request["stream_flags"].asUInt());
                axilite_wr_addr.Push(ctrl_offset+0x58);
                dma_addr = request["addr_0"].asUInt64();
                axilite_wr_data.Push((uint32_t)(dma_addr & 0xffffffff));
                axilite_wr_addr.Push(ctrl_offset+0x5c);
                axilite_wr_data.Push((uint32_t)(dma_addr >> 32));
                axilite_wr_addr.Push(ctrl_offset+0x64);
                dma_addr = request["addr_1"].asUInt64();
                axilite_wr_data.Push((uint32_t)(dma_addr & 0xffffffff));
                axilite_wr_addr.Push(ctrl_offset+0x68);
                axilite_wr_data.Push((uint32_t)(dma_addr >> 32));
                axilite_wr_addr.Push(ctrl_offset+0x70);
                dma_addr = request["addr_2"].asUInt64();
                axilite_wr_data.Push((uint32_t)(dma_addr & 0xffffffff));
                axilite_wr_addr.Push(ctrl_offset+0x74);
                axilite_wr_data.Push((uint32_t)(dma_addr >> 32));
                this_thread::sleep_for(chrono::milliseconds(500));
                //start the controller
                axilite_wr_addr.Push(ctrl_offset);
                axilite_wr_data.Push(1);
                this_thread::sleep_for(chrono::milliseconds(1500));
                response["status"] = 0;
            }else{
                //push a command into the call stream
                callreq.Push(request["scenario"].asUInt());
                callreq.Push(request["count"].asUInt());
                callreq.Push(request["comm"].asUInt());
                callreq.Push(request["root_src_dst"].asUInt());
                callreq.Push(request["function"].asUInt());
                callreq.Push(request["tag"].asUInt());
                callreq.Push(request["arithcfg"].asUInt());
                callreq.Push(request["compression_flags"].asUInt());
                callreq.Push(request["stream_flags"].asUInt());
                dma_addr = request["addr_0"].asUInt64();
                callreq.Push((uint32_t)(dma_addr & 0xffffffff));
                callreq.Push((uint32_t)(dma_addr >> 32));
                dma_addr = request["addr_1"].asUInt64();
                callreq.Push((uint32_t)(dma_addr & 0xffffffff));
                callreq.Push((uint32_t)(dma_addr >> 32));
                dma_addr = request["addr_2"].asUInt64();
                callreq.Push((uint32_t)(dma_addr & 0xffffffff));
                callreq.Push((uint32_t)(dma_addr >> 32));
                response["status"] = 0;
            }
            break;
        // Call wait request  {"type": 6, arg controller id}
        // Call wait response {"status": OK|NOK|ERR}
        case 6:
            ctrl_id = request["ctrl_id"].asUInt();
            *logger << log_level::verbose << "Call wait on control stream " << ctrl_id << endl;
            //pop the corresponding status queue if possible
            if(ctrl_id >= NUM_CTRL_STREAMS){
                response["status"] = -1;
            } else {
                if(ctrl_id < 2){
                    //we handle this request by checking on a hostctrl
                    ctrl_offset = 0x2000 * (ctrl_id+1);
                    *logger << log_level::verbose << "Status read on controller " << ctrl_id << endl;
                    axilite_rd_addr.Push(ctrl_offset);
                    //wait for a response to come back
                    while(!ctx->stop && axilite_rd_data.IsEmpty()){
                        *logger << log_level::debug << "Read wait on controller " << ctrl_id << endl;
                        this_thread::sleep_for(chrono::milliseconds(50));
                    }
                    unsigned int resp;
                    if(!axilite_rd_data.IsEmpty()){
                        resp = axilite_rd_data.Pop();
                        *logger << log_level::debug << "Read response: " << resp << endl;
                    } else{
                        resp = 0;
                    }
                    response["status"] = (resp & 0x2) ? 0 : 1;

                } else {
                    //we handle this request checking the status stream
                    if(callack.IsEmpty()){
                        response["status"] = 1;
                    } else {
                        callack.Pop();
                        response["status"] = 0;
                    }
                }
            }
            break;
        default:
            (*logger)("Unrecognized message\n", log_level::warning);
            response["status"] = -1;

    }
    //return message to client
    string str = Json::writeString(builder, response);
    ctx->cmd_socket->send(str);
}

void zmq_cmd_server(zmq_intf_context *ctx,
                Stream<unsigned int> &axilite_rd_addr, Stream<unsigned int> &axilite_rd_data,
                Stream<unsigned int> &axilite_wr_addr, Stream<unsigned int> &axilite_wr_data,
                Stream<ap_uint<64> > &aximm_rd_addr, Stream<ap_uint<32> > &aximm_rd_len, Stream<ap_uint<512> > &aximm_rd_data,
                Stream<ap_uint<64> > &aximm_wr_addr, Stream<ap_uint<32> > &aximm_wr_len, Stream<ap_uint<512> > &aximm_wr_data, Stream<ap_uint<64> > &aximm_wr_strb,
                Stream<unsigned int> &callreq, Stream<unsigned int> &callack){
    (*logger)("Starting ZMQ server\n", log_level::verbose);
    while(!ctx->stop){
        serve_zmq(ctx,
            axilite_rd_addr, axilite_rd_data,
            axilite_wr_addr, axilite_wr_data,
            aximm_rd_addr, aximm_rd_len, aximm_rd_data,
            aximm_wr_addr, aximm_wr_len, aximm_wr_data, aximm_wr_strb,
            callreq, callack
        );
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ server\n", log_level::verbose);
}

void zmq_eth_egress_server(zmq_intf_context *ctx, Stream<stream_word > &in, unsigned int local_rank){
    (*logger)("Starting ZMQ Eth Egress server\n", log_level::verbose);
    while(!ctx->stop){
        eth_endpoint_egress_port(ctx, in, local_rank);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ Eth Egress server\n", log_level::verbose);
}

void zmq_eth_ingress_server(zmq_intf_context *ctx, Stream<stream_word > &out){
    (*logger)("Starting ZMQ Eth Ingress server\n", log_level::verbose);
    while(!ctx->stop){
        eth_endpoint_ingress_port(ctx, out);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ Eth Ingress server\n", log_level::verbose);
}

void zmq_krnl_egress_server(zmq_intf_context *ctx, Stream<stream_word > &in){
    (*logger)("Starting ZMQ Streaming Kernel Egress server\n", log_level::verbose);
    while(!ctx->stop){
        krnl_endpoint_egress_port(ctx, in);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ Streaming Kernel Egress server\n", log_level::verbose);
}

void zmq_krnl_ingress_server(zmq_intf_context *ctx, Stream<stream_word > &out){
    (*logger)("Starting ZMQ Streaming Kernel Ingress server\n", log_level::verbose);
    while(!ctx->stop){
        krnl_endpoint_ingress_port(ctx, out);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ Streaming Kernel Ingress server\n", log_level::verbose);
}
