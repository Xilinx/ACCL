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

#include "zmq_client.h"
#include <iostream>
#include <string>

using namespace std;

zmq_intf_context zmq_client_cmd_intf(unsigned int starting_port, unsigned int local_rank){
    zmq_intf_context ctx;
    const string endpoint_base = "tcp://127.0.0.1:";

    ctx.cmd_socket = new zmqpp::socket(ctx.context, zmqpp::socket_type::request);

    string cmd_endpoint = endpoint_base + to_string(starting_port + local_rank);
    cout << "Endpoint: " << cmd_endpoint << endl;

    // bind to the socket(s)
    cout << "Rank " << local_rank << " binding to " << cmd_endpoint << " (CMD)" << endl;
    ctx.cmd_socket->connect(cmd_endpoint);

    this_thread::sleep_for(chrono::milliseconds(1000));

    cout << "ZMQ Client Command Context established for rank " << local_rank << endl;

    return ctx;
}

zmq_intf_context zmq_client_krnl_intf(unsigned int starting_port, unsigned int local_rank, unsigned int world_size, unsigned int krnl_dest)
{
    zmq_intf_context ctx;
    const string endpoint_base = "tcp://127.0.0.1:";

    ctx.krnl_tx_socket = new zmqpp::socket(ctx.context, zmqpp::socket_type::sub);
    ctx.krnl_rx_socket = new zmqpp::socket(ctx.context, zmqpp::socket_type::pub);

    //bind to tx socket
    string krnl_endpoint = endpoint_base + to_string(starting_port+2*world_size+local_rank);
    cout << "Rank " << local_rank << " connecting to " << krnl_endpoint << " (KRNL)" << endl;
    ctx.krnl_tx_socket->connect(krnl_endpoint);
    this_thread::sleep_for(chrono::milliseconds(1000));
    //subscribe to dst == local_rank
    string krnl_subscribe = (krnl_dest > 0) ? to_string(krnl_dest) : "";
    cout << "Rank " << local_rank << " subscribing to " << ((krnl_dest > 0) ? krnl_subscribe : "all") << " (KRNL)" << endl;
    ctx.krnl_tx_socket->subscribe(krnl_subscribe);
    this_thread::sleep_for(chrono::milliseconds(1000));
    //connect to rx socket
    krnl_endpoint = endpoint_base + to_string(starting_port+3*world_size+local_rank);
    cout << "Rank " << local_rank << " binding to " << krnl_endpoint << " (KRNL)" << endl;
    ctx.krnl_rx_socket->bind(krnl_endpoint);
    this_thread::sleep_for(chrono::milliseconds(1000));

    cout << "ZMQ Client Kernel Context established for rank " << local_rank << endl;

    return ctx;
}

zmq_intf_context zmq_client_intf(unsigned int starting_port, unsigned int local_rank, unsigned int world_size, unsigned int krnl_dest){
    zmq_intf_context ctx = zmq_client_cmd_intf(starting_port, local_rank);
    if(krnl_dest > 0){
        zmq_intf_context krnl_ctx = zmq_client_krnl_intf(starting_port, local_rank, world_size, krnl_dest);
        ctx.krnl_tx_socket = krnl_ctx.krnl_tx_socket;
        ctx.krnl_rx_socket = krnl_ctx.krnl_rx_socket;
    }
    return ctx;
}

void zmq_client_startcall(zmq_intf_context *ctx, unsigned int scenario, unsigned int tag, unsigned int count,
                        unsigned int comm, unsigned int root_src_dst, unsigned int function,
                        unsigned int arithcfg_addr, unsigned int compression_flags, unsigned int stream_flags,
                        uint64_t addr_0, uint64_t addr_1, uint64_t addr_2) {
    Json::Value request_json;

    request_json["type"] = 4;
    request_json["scenario"] = scenario;
    request_json["tag"] = tag;
    request_json["count"] = count;
    request_json["comm"] = comm;
    request_json["root_src_dst"] = root_src_dst;
    request_json["function"] = function;
    request_json["arithcfg"] = arithcfg_addr;
    request_json["compression_flags"] = compression_flags;
    request_json["stream_flags"] = stream_flags;
    request_json["addr_0"] = (Json::Value::UInt64)addr_0;
    request_json["addr_1"] = (Json::Value::UInt64)addr_1;
    request_json["addr_2"] = (Json::Value::UInt64)addr_2;

    zmqpp::message msg;
    to_message(request_json, msg);
    ctx->cmd_socket->send(msg);
}

void zmq_client_retcall(zmq_intf_context *ctx){
    zmqpp::message reply;
    ctx->cmd_socket->receive(reply);
    Json::Value status = to_json(reply);
    if (status["status"] != 0) {
        throw std::runtime_error("ZMQ call error (" + std::to_string(status["status"].asUInt()) + ")");
    }
}

unsigned int zmq_client_cfgread(zmq_intf_context *ctx, unsigned int offset){
    Json::Value request_json;

    request_json["type"] = 0;
    request_json["addr"] = (Json::Value::UInt)offset;
    zmqpp::message request;
    to_message(request_json, request);
    ctx->cmd_socket->send(request);

    zmqpp::message reply;
    ctx->cmd_socket->receive(reply);
    Json::Value reply_json = to_json(reply);
    if (reply_json["status"] != 0) {
        throw std::runtime_error("ZMQ config read error (" + std::to_string(reply_json["status"].asUInt()) + ")");
    }
    return reply_json["rdata"].asUInt();
}

void zmq_client_cfgwrite(zmq_intf_context *ctx, unsigned int offset, unsigned int val) {
    Json::Value request_json;
    request_json["type"] = 1;
    request_json["addr"] = (Json::Value::UInt)offset;
    request_json["wdata"] = (Json::Value::UInt)val;
    zmqpp::message request;
    to_message(request_json, request);
    ctx->cmd_socket->send(request);

    zmqpp::message reply;
    ctx->cmd_socket->receive(reply);
    Json::Value reply_json = to_json(reply);
    if (reply_json["status"] != 0) {
        throw std::runtime_error("ZMQ config write error (" + std::to_string(reply_json["status"].asUInt()) + ")");
    }
}

void zmq_client_memread(zmq_intf_context *ctx, uint64_t adr, unsigned int size, uint8_t *data){
    Json::Value request_json;
    request_json["type"] = 2;
    request_json["addr"] = (Json::Value::UInt64)adr;
    request_json["len"] = (Json::Value::UInt64)size;
    zmqpp::message request;
    to_message(request_json, request);
    ctx->cmd_socket->send(request);

    zmqpp::message reply;
    ctx->cmd_socket->receive(reply);
    Json::Value reply_json = to_json(reply);
    if (reply_json["status"] != 0) {
        throw std::runtime_error("ZMQ mem read error (" + std::to_string(reply_json["status"].asUInt()) + ")");
    }

    size_t array_size = reply_json["rdata"].size();
    for (size_t i = 0; i < array_size; ++i) {
        data[i] = (uint8_t)reply_json["rdata"][(Json::ArrayIndex)i].asInt();
    }
}

void zmq_client_memwrite(zmq_intf_context *ctx, uint64_t adr, unsigned int size, uint8_t *data){
    Json::Value request_json;
    request_json["type"] = 3;
    request_json["addr"] = (Json::Value::UInt64)adr;

    Json::Value array;
    for (size_t i = 0; i < size; ++i) {
        array[(Json::ArrayIndex)i] = (Json::Value::Int)data[i];
    }
    request_json["wdata"] = array;

    zmqpp::message request;
    to_message(request_json, request);
    ctx->cmd_socket->send(request);

    zmqpp::message reply;
    ctx->cmd_socket->receive(reply);
    Json::Value reply_json = to_json(reply);
    if (reply_json["status"] != 0) {
        throw std::runtime_error("ZMQ mem write error (" + std::to_string(reply_json["status"].asUInt()) + ")");
    }
}

std::vector<uint8_t> zmq_client_strmread(zmq_intf_context *ctx){
    zmqpp::message msg;
    ctx->krnl_tx_socket->receive(msg);
    Json::Value msg_json = to_json(msg);
    std::vector<uint8_t> ret;
    size_t array_size = msg_json["data"].size();
    for (size_t i = 0; i < array_size; ++i) {
        ret.push_back((uint8_t)msg_json["data"][(Json::ArrayIndex)i].asInt());
    }
    return ret;
}

void zmq_client_strmwrite(zmq_intf_context *ctx, std::vector<uint8_t> val){
    Json::Value msg_json;
    zmqpp::message msg;
    ctx->krnl_tx_socket->receive(msg);
    for (int i = 0; i < val.size(); ++i) {
        msg_json["data"][i] = val.at(i);
    }
    to_message(msg_json, msg);
    ctx->krnl_rx_socket->send(msg);
}