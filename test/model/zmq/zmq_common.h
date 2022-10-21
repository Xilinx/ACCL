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
#include <json/json.h>
#include <zmqpp/zmqpp.hpp>
#include <memory>

/**
 * @brief ZMQ interface context, consisting of ZMQ sockets
 *
 */
struct zmq_intf_context{
    zmqpp::context context;
    std::unique_ptr<zmqpp::socket> cmd_socket;
    std::unique_ptr<zmqpp::socket> eth_tx_socket;
    std::unique_ptr<zmqpp::socket> eth_rx_socket;
    std::unique_ptr<zmqpp::socket> krnl_tx_socket;
    std::unique_ptr<zmqpp::socket> krnl_rx_socket;
    bool stop = false;
    zmq_intf_context() : context() {}
};

/**
 * @brief Convert a ZMQ message to JSON
 *
 * @param message Reference to the ZMQ message, as received from the socket
 * @return Json::Value The JSON equivalent
 */
Json::Value to_json(zmqpp::message &message);

/**
 * @brief Convert a JSON to a ZMQ message
 *
 * @param request_json The JSON input
 * @param request Reference to the ZMQ message, ready for sending
 */
void to_message(Json::Value &request_json, zmqpp::message &request);
