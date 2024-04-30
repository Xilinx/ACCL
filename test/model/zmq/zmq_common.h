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
#include <zmq.hpp>
#include <memory>

/**
 * @brief ZMQ interface context, consisting of ZMQ sockets
 *
 */
struct zmq_intf_context{
    zmq::context_t context;
    std::unique_ptr<zmq::socket_t> cmd_socket;
    std::unique_ptr<zmq::socket_t> eth_tx_socket;
    std::unique_ptr<zmq::socket_t> eth_rx_socket;
    std::unique_ptr<zmq::socket_t> krnl_tx_socket;
    std::unique_ptr<zmq::socket_t> krnl_rx_socket;
    bool stop = false;
    zmq_intf_context() : context() {}
};

/**
 * @brief Convert a ZMQ message to JSON
 *
 * @param message Reference to the ZMQ message, as received from the socket
 * @return Json::Value The JSON equivalent
 */
Json::Value to_json(zmq::message_t &message);

/**
 * @brief Convert a JSON to a ZMQ message
 *
 * @param request_json The JSON input
 * @return zmq::message_t The ZMQ message
 */
zmq::message_t to_message(Json::Value &request_json);
