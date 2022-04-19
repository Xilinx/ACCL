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
#pragma once
#include <jsoncpp/json/json.h>
#include <zmqpp/zmqpp.hpp>

struct zmq_intf_context{
    zmqpp::context context;
    zmqpp::socket *cmd_socket;
    zmqpp::socket *eth_tx_socket;
    zmqpp::socket *eth_rx_socket;
    zmqpp::socket *krnl_tx_socket;
    zmqpp::socket *krnl_rx_socket;
    bool stop = false;
    zmq_intf_context() : context() {}
};

Json::Value to_json(zmqpp::message &message);
void to_message(Json::Value &request_json, zmqpp::message &request);