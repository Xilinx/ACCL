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
#include <mutex>

using namespace std;

mutex cmd_socket_mutex;

zmq_intf_context zmq_client_intf(unsigned int starting_port, unsigned int local_rank, const vector<unsigned int>& krnl_dest, unsigned int world_size){
    zmq_intf_context ctx;
    const string endpoint_base = "tcp://127.0.0.1:";

    ctx.cmd_socket = std::make_unique<zmqpp::socket>(ctx.context, zmqpp::socket_type::request);

    string cmd_endpoint = endpoint_base + to_string(starting_port + local_rank);
    cout << "Endpoint: " << cmd_endpoint << endl;

    // bind to the socket(s)
    cout << "Rank " << local_rank << " binding to " << cmd_endpoint << " (CMD)" << endl;
    ctx.cmd_socket->connect(cmd_endpoint);

    this_thread::sleep_for(chrono::milliseconds(1000));

    cout << "ZMQ Client Command Context established for rank " << local_rank << endl;

    if(krnl_dest.size() > 0){
        ctx.krnl_tx_socket = std::make_unique<zmqpp::socket>(ctx.context, zmqpp::socket_type::sub);
        ctx.krnl_rx_socket = std::make_unique<zmqpp::socket>(ctx.context, zmqpp::socket_type::pub);

        //bind to tx socket
        string krnl_endpoint = endpoint_base + to_string(starting_port+2*world_size+local_rank);
        cout << "Rank " << local_rank << " connecting to " << krnl_endpoint << " (KRNL)" << endl;
        ctx.krnl_tx_socket->connect(krnl_endpoint);
        this_thread::sleep_for(chrono::milliseconds(1000));
        //subscribe to destinations
        for(int i=0; i<(int)krnl_dest.size(); i++){
            string krnl_subscribe = to_string(krnl_dest.at(i));
            cout << "Rank " << local_rank << " subscribing to " << krnl_subscribe << " (KRNL)" << endl;
            ctx.krnl_tx_socket->subscribe(krnl_subscribe);
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
        //connect to rx socket
        krnl_endpoint = endpoint_base + to_string(starting_port+3*world_size+local_rank);
        cout << "Rank " << local_rank << " binding to " << krnl_endpoint << " (KRNL)" << endl;
        ctx.krnl_rx_socket->bind(krnl_endpoint);
        this_thread::sleep_for(chrono::milliseconds(1000));

        cout << "ZMQ Client Kernel Context established for rank " << local_rank << endl;
    }

    return ctx;
}

void zmq_client_startcall(zmq_intf_context *ctx, unsigned int scenario, unsigned int tag, unsigned int count,
                        unsigned int comm, unsigned int root_src_dst, unsigned int function,
                        unsigned int arithcfg_addr, unsigned int compression_flags, unsigned int stream_flags,
                        uint64_t addr_0, uint64_t addr_1, uint64_t addr_2, unsigned int ctrl_id) {
    Json::Value request_json;

    request_json["type"] = 5;
    request_json["ctrl_id"] = ctrl_id;
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

    //send the message out to the CCLO simulator/emulator
    zmqpp::message msg;
    to_message(request_json, msg);
    cmd_socket_mutex.lock();
    ctx->cmd_socket->send(msg);
    //receive confirmation
    zmqpp::message reply;
    ctx->cmd_socket->receive(reply);
    cmd_socket_mutex.unlock();
    Json::Value status = to_json(reply);
    if (status["status"].asInt() < 0) {
        throw std::runtime_error("ZMQ startcall error (" + std::to_string(status["status"].asInt()) + ")");
    }
}

void zmq_client_retcall(zmq_intf_context *ctx, unsigned int ctrl_id){
    while(1){
        //send an update request out to the CCLO simulator/emulator
        Json::Value request_json;
        request_json["type"] = 6;
        request_json["ctrl_id"] = ctrl_id;
        zmqpp::message msg;
        to_message(request_json, msg);
        cmd_socket_mutex.lock();
        ctx->cmd_socket->send(msg);
        //check the reply
        zmqpp::message reply;
        ctx->cmd_socket->receive(reply);
        cmd_socket_mutex.unlock();
        Json::Value status = to_json(reply);
        if (status["status"].asInt() < 0) {
            throw std::runtime_error("ZMQ retcall error (" + std::to_string(status["status"].asInt()) + ")");
        } else if (status["status"] == 0) {
            return;
        }
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

void zmq_client_memread(zmq_intf_context *ctx, uint64_t adr, unsigned int size, uint8_t *data, bool host){
    Json::Value request_json;
    request_json["type"] = 2;
    request_json["addr"] = (Json::Value::UInt64)adr;
    request_json["len"] = (Json::Value::UInt64)size;
    request_json["host"] = host;
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

void zmq_client_memwrite(zmq_intf_context *ctx, uint64_t adr, unsigned int size, uint8_t *data, bool host){
    Json::Value request_json;
    request_json["type"] = 3;
    request_json["addr"] = (Json::Value::UInt64)adr;
    request_json["host"] = host;

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

void zmq_client_memalloc(zmq_intf_context *ctx, uint64_t adr, unsigned int size, bool host){
    Json::Value request_json;
    request_json["type"] = 4;
    request_json["addr"] = (Json::Value::UInt64)adr;
    request_json["len"] = (Json::Value::UInt64)size;
    request_json["host"] = host;

    zmqpp::message request;
    to_message(request_json, request);
    ctx->cmd_socket->send(request);

    zmqpp::message reply;
    ctx->cmd_socket->receive(reply);
    Json::Value reply_json = to_json(reply);
    if (reply_json["status"] != 0) {
        throw std::runtime_error("ZMQ mem alloc error (" + std::to_string(reply_json["status"].asUInt()) + ")");
    }
}

std::vector<uint8_t> zmq_client_strmread(zmq_intf_context *ctx, bool dont_block){
    zmqpp::message msg;
    Json::Reader reader;
    Json::Value msg_json;
    std::string msg_text, dst_text;

    if(!ctx->krnl_tx_socket->receive(msg, dont_block)) return std::vector<uint8_t>();

    // decompose the message
    msg >> dst_text;
    msg >> msg_text;
    reader.parse(msg_text, msg_json);

    std::vector<uint8_t> ret;
    size_t array_size = msg_json["data"].size();
    for (size_t i = 0; i < array_size; ++i) {
        ret.push_back((uint8_t)msg_json["data"][(Json::ArrayIndex)i].asInt());
    }
    return ret;
}

void zmq_client_strmwrite(zmq_intf_context *ctx, std::vector<uint8_t> val, unsigned int dest){
    Json::Value data_packet;
    zmqpp::message msg;
    Json::StreamWriterBuilder builder;
    for (int i = 0; i < static_cast<int>(val.size()); ++i) {
        data_packet["data"][i] = val.at(i);
    }
    //first part of the message is the destination, used to filter at receiver
    msg << to_string(dest);
    //package the data
    msg << Json::writeString(builder, data_packet);
    ctx->krnl_rx_socket->send(msg);
}
