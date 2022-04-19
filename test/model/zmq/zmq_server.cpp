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
    ctx.cmd_socket = new zmqpp::socket(ctx.context, zmqpp::socket_type::reply);
    ctx.eth_tx_socket = new zmqpp::socket(ctx.context, zmqpp::socket_type::pub);
    ctx.eth_rx_socket = new zmqpp::socket(ctx.context, zmqpp::socket_type::sub);
    ctx.krnl_tx_socket = new zmqpp::socket(ctx.context, zmqpp::socket_type::pub);
    ctx.krnl_rx_socket = new zmqpp::socket(ctx.context, zmqpp::socket_type::sub);

    const string endpoint_base = "tcp://127.0.0.1:";

    string cmd_endpoint = endpoint_base + to_string(starting_port + local_rank);
    *logger << log_level::verbose << "Endpoint: " << cmd_endpoint << endl;
    vector<string> eth_endpoints;

    for(int i=0; i<world_size; i++){
        eth_endpoints.emplace_back(endpoint_base + to_string(starting_port+world_size+i));
        *logger << log_level::verbose << "Endpoint rank " << i << ": " << eth_endpoints.at(i) << endl;
    }

    // bind to the socket(s)
    *logger << log_level::info << "Rank " << local_rank << " binding to " << cmd_endpoint << " (CMD) and " << eth_endpoints.at(local_rank) << " (ETH)" << endl;
    ctx.cmd_socket->bind(cmd_endpoint);
    ctx.eth_tx_socket->bind(eth_endpoints.at(local_rank));

    this_thread::sleep_for(chrono::milliseconds(1000));

    // connect to the sockets
    for(int i=0; i<world_size; i++){
        *logger << log_level::info << "Rank " << local_rank << " connecting to " << eth_endpoints.at(i) << " (ETH)" << endl;
        ctx.eth_rx_socket->connect(eth_endpoints.at(i));
    }

    this_thread::sleep_for(chrono::milliseconds(1000));

    *logger << log_level::info << "Rank " << local_rank << " subscribing to " << local_rank << " (ETH)" << endl;
    ctx.eth_rx_socket->subscribe(to_string(local_rank));

    this_thread::sleep_for(chrono::milliseconds(1000));

    //kernel interface
    //bind to tx socket
    string krnl_endpoint = endpoint_base + to_string(starting_port+2*world_size+local_rank);
    *logger << log_level::info << "Rank " << local_rank << " binding to " << krnl_endpoint << " (KRNL)" << endl;
    ctx.krnl_tx_socket->bind(krnl_endpoint);
    this_thread::sleep_for(chrono::milliseconds(1000));
    //connect to rx socket
    //in case we want to loopback the kernel interface, we connect to the TX sockets for RX
    if(!kernel_loopback){
        krnl_endpoint = endpoint_base + to_string(starting_port+3*world_size+local_rank);
    }
    *logger << log_level::info << "Rank " << local_rank << " connecting to " << krnl_endpoint << " (KRNL)" << endl;
    ctx.krnl_rx_socket->connect(krnl_endpoint);
    this_thread::sleep_for(chrono::milliseconds(1000));
    //subscribe to dst == local_rank
    string krnl_subscribe = kernel_loopback ? "" : to_string(local_rank);
    *logger << log_level::info << "Rank " << local_rank << " subscribing to " << krnl_subscribe << " (KRNL)" << endl;
    ctx.krnl_rx_socket->subscribe(krnl_subscribe);
    this_thread::sleep_for(chrono::milliseconds(1000));

    *logger << log_level::info << "ZMQ Context established for rank " << local_rank << endl;

    return ctx;
}

void eth_endpoint_egress_port(zmq_intf_context *ctx, Stream<stream_word > &in, unsigned int local_rank, bool remap_dest){

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
    //do a bit of dest translation, because
    //dest is actually a local session number, and sessions are allocated
    //sequentially to ranks, skipping the local rank
    //therefore ranks  0, 1, ... , local_rank, local_rank+1, ... , N-2, N-1
    //map to sesssions 0, 1, ... ,        N/A, local_rank  , ... , N-3, N-2
    if(remap_dest){
        if(dest >= local_rank){
            dest++;
        }
    }
    //first part of the message is the destination port ID
    message << to_string(dest);
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
    ctx->krnl_tx_socket->send(message);
}


void krnl_endpoint_ingress_port(zmq_intf_context *ctx, Stream<stream_word > &out){

    Json::Reader reader;

    // receive the message
    zmqpp::message message;
    if(!ctx->krnl_rx_socket->receive(message, true)) return;

    // decompose the message
    string msg_text, dst_text;;

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

void serve_zmq(zmq_intf_context *ctx, uint32_t *cfgmem, vector<char> &devicemem, Stream<ap_axiu<32,0,0,0> > &cmd, Stream<ap_axiu<32,0,0,0> > &sts){

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
    uint64_t dma_addr;
    Json::Value dma_wdata;
    switch(request["type"].asUInt()){
        // MMIO read request  {"type": 0, "addr": <uint>}
        // MMIO read response {"status": OK|ERR, "rdata": <uint>}
        case 0:
            adr = request["addr"].asUInt();
            *logger << log_level::info << "MMIO read " << adr << endl;

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
            *logger << log_level::info << "MMIO write " << adr << endl;
            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
            } else {
                cfgmem[adr/4] = request["wdata"].asUInt();
            }
            break;
        // Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>}
        // Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
        case 2:
            adr = request["addr"].asUInt();
            len = request["len"].asUInt();
            *logger << log_level::info << "Mem read " << adr << " len: " << len << endl;
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
            *logger << log_level::info << "Mem write " << adr << " len: " << len << endl;
            if((adr+len) > devicemem.size()){
                devicemem.resize(adr+len);
            }
            for(int i=0; i<len; i++){
                devicemem.at(adr+i) = dma_wdata[i].asUInt();
            }
            break;
        // Call request  {"type": 4, arg names and values}
        // Call response {"status": OK|ERR}
        case 4:
            *logger << log_level::info << "Call with scenario " << request["scenario"].asUInt() << endl;
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["scenario"].asUInt(), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["count"].asUInt(), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["comm"].asUInt(), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["root_src_dst"].asUInt(), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["function"].asUInt(), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["tag"].asUInt(), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["arithcfg"].asUInt(), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["compression_flags"].asUInt(), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=request["stream_flags"].asUInt(), .last=0});
            dma_addr = request["addr_0"].asUInt64();
            cmd.Push((ap_axiu<32,0,0,0>){.data=(uint32_t)(dma_addr & 0xffffffff), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=(uint32_t)(dma_addr >> 32), .last=0});
            dma_addr = request["addr_1"].asUInt64();
            cmd.Push((ap_axiu<32,0,0,0>){.data=(uint32_t)(dma_addr & 0xffffffff), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=(uint32_t)(dma_addr >> 32), .last=0});
            dma_addr = request["addr_2"].asUInt64();
            cmd.Push((ap_axiu<32,0,0,0>){.data=(uint32_t)(dma_addr & 0xffffffff), .last=0});
            cmd.Push((ap_axiu<32,0,0,0>){.data=(uint32_t)(dma_addr >> 32), .last=1});
            //pop the status queue to wait for call completion
            sts.Pop();
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
                Stream<ap_uint<64> > &aximm_rd_addr, Stream<ap_uint<512> > &aximm_rd_data,
                Stream<ap_uint<64> > &aximm_wr_addr, Stream<ap_uint<512> > &aximm_wr_data, Stream<ap_uint<64> > &aximm_wr_strb,
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
    uint64_t dma_addr;
    Json::Value dma_wdata;
    ap_uint<64> mem_addr;
    ap_uint<512> mem_data;
    ap_uint<64> mem_strb;
    switch(request["type"].asUInt()){
        // MMIO read request  {"type": 0, "addr": <uint>}
        // MMIO read response {"status": OK|ERR, "rdata": <uint>}
        case 0:
            adr = request["addr"].asUInt();
            *logger << log_level::info << "MMIO read " << adr << endl;
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
            *logger << log_level::info << "MMIO write " << adr << endl;
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
        // Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>}
        // Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
        case 2:
            adr = request["addr"].asUInt();
            len = request["len"].asUInt();
            *logger << log_level::info << "Mem read " << adr << " len: " << len << endl;
            if((adr+len) > 256*1024){
                response["status"] = 1;
                response["rdata"][0] = 0;
            } else {
                for(int i=0; i<len; i+=64){
                    mem_addr = adr+i;
                    aximm_rd_addr.Push(mem_addr);
                    while(!ctx->stop){
                        if(!aximm_rd_data.IsEmpty()){
                            mem_data = aximm_rd_data.Pop();
                            break;
                        } else{
                            this_thread::sleep_for(chrono::milliseconds(1));
                        }
                    }
                    for(int j=0; j<64 && (i+j)<len; j++){
                        response["rdata"][i+j] = (unsigned int)mem_data(8*(j+1)-1, 8*j);
                    }
                }
            }
            break;
        // Devicemem write request  {"type": 3, "addr": <uint>, "wdata": <array of uint>}
        // Devicemem write response {"status": OK|ERR}
        case 3:
            adr = request["addr"].asUInt();
            dma_wdata = request["wdata"];
            len = dma_wdata.size();
            *logger << log_level::info << "Mem write " << adr << " len: " << len << endl;
            if((adr+len) > 256*1024){
                response["status"] = 1;
            } else{
                for(int i=0; i<len; i+=64){
                    mem_strb = 0;
                    mem_addr = adr+i;
                    for(int j=0; j<64 && (i+j)<len; j++){
                        mem_data(8*(j+1)-1, 8*j) = dma_wdata[i+j].asUInt();
                        mem_strb(j,j) = 1;
                    }
                    while(!ctx->stop){
                        if(!aximm_wr_addr.IsFull() && !aximm_wr_data.IsFull() && !aximm_wr_strb.IsFull()){
                            aximm_wr_addr.Push(mem_addr);
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
        // Call request  {"type": 4, arg names and values}
        // Call response {"status": OK|ERR}
        case 4:
            *logger << log_level::info << "Call with scenario " << request["scenario"].asUInt() << endl;
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
            //pop the status queue to wait for call completion
            while(!ctx->stop){
                if(!callack.IsEmpty()){
                    callack.Pop();
                    break;
                } else{
                    this_thread::sleep_for(chrono::milliseconds(1));
                }
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

void zmq_cmd_server(zmq_intf_context *ctx,
                Stream<unsigned int> &axilite_rd_addr, Stream<unsigned int> &axilite_rd_data,
                Stream<unsigned int> &axilite_wr_addr, Stream<unsigned int> &axilite_wr_data,
                Stream<ap_uint<64> > &aximm_rd_addr, Stream<ap_uint<512> > &aximm_rd_data,
                Stream<ap_uint<64> > &aximm_wr_addr, Stream<ap_uint<512> > &aximm_wr_data, Stream<ap_uint<64> > &aximm_wr_strb,
                Stream<unsigned int> &callreq, Stream<unsigned int> &callack){
    (*logger)("Starting ZMQ server\n", log_level::info);
    while(!ctx->stop){
        serve_zmq(ctx,
            axilite_rd_addr, axilite_rd_data,
            axilite_wr_addr, axilite_wr_data,
            aximm_rd_addr, aximm_rd_data,
            aximm_wr_addr, aximm_wr_data, aximm_wr_strb,
            callreq, callack
        );
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ server\n", log_level::info);
}

void zmq_eth_egress_server(zmq_intf_context *ctx, Stream<stream_word > &in, unsigned int local_rank, bool remap_dest){
    (*logger)("Starting ZMQ Eth Egress server\n", log_level::info);
    while(!ctx->stop){
        eth_endpoint_egress_port(ctx, in, local_rank, remap_dest);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ Eth Egress server\n", log_level::info);
}

void zmq_eth_ingress_server(zmq_intf_context *ctx, Stream<stream_word > &out){
    (*logger)("Starting ZMQ Eth Ingress server\n", log_level::info);
    while(!ctx->stop){
        eth_endpoint_ingress_port(ctx, out);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ Eth Ingress server\n", log_level::info);
}

void zmq_krnl_egress_server(zmq_intf_context *ctx, Stream<stream_word > &in){
    (*logger)("Starting ZMQ Streaming Kernel Egress server\n", log_level::info);
    while(!ctx->stop){
        krnl_endpoint_egress_port(ctx, in);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ Streaming Kernel Egress server\n", log_level::info);
}

void zmq_krnl_ingress_server(zmq_intf_context *ctx, Stream<stream_word > &out){
    (*logger)("Starting ZMQ Streaming Kernel Ingress server\n", log_level::info);
    while(!ctx->stop){
        krnl_endpoint_ingress_port(ctx, out);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    (*logger)("Exiting ZMQ Streaming Kernel Ingress server\n", log_level::info);
}
