/*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
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
#
*******************************************************************************/

#pragma once
#include "buffer.hpp"
#include "common.hpp"
#include <cmath>
#include <jsoncpp/json/json.h>
#include <zmq.hpp>

/** @file simbuffer.hpp */

namespace ACCL {
extern addr_t next_free_address;

template <typename dtype> class SimBuffer : public Buffer<dtype> {
private:
  zmq::socket_t *const socket;

  addr_t get_next_free_address(size_t size) {
    addr_t address = next_free_address;
    // allocate on 4K boundaries
    // not sure how realistic this is, but it does help
    // work around some addressing limitations in RTLsim
    next_free_address += ((addr_t)std::ceil(size / 4096.0)) * 4096;

    return address;
  }

public:
  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq::socket_t *const socket, const addr_t physical_address)
      : Buffer<dtype>(buffer, length, type, physical_address), socket(socket) {}

  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq::socket_t *const socket)
      : SimBuffer(buffer, length, type, socket,
                  this->get_next_free_address(length * sizeof(dtype))) {}
  void sync_from_device() override {
    Json::Value request_json;
    request_json["type"] = 2;
    request_json["addr"] = (Json::Value::UInt64)this->_physical_address;
    request_json["len"] = (Json::Value::UInt64)this->_size;
    Json::StreamWriterBuilder builder;
    builder["indentation"] = ""; // minimize output
    const std::string request = Json::writeString(builder, request_json);
    accl_send_log("sync from device", request);
    this->socket->send(zmq::const_buffer(request.c_str(), request.size()),
                       zmq::send_flags::none);

    zmq::message_t reply;
    zmq::recv_result_t result =
        this->socket->recv(reply, zmq::recv_flags::none);
    Json::Value reply_json = parse_json(reply.to_string());
    check_return_status(reply_json["status"]);

    size_t array_size = reply_json["rdata"].size();
    uint8_t *data = static_cast<uint8_t *>(this->_byte_array);
    for (size_t i = 0; i < array_size; ++i) {
      data[i] = (uint8_t)reply_json["rdata"][(Json::ArrayIndex)i].asInt();
    }
  }

  void sync_to_device() override {
    Json::Value request_json;
    request_json["type"] = 3;
    request_json["addr"] = (Json::Value::UInt64)this->_physical_address;

    Json::Value array;
    uint8_t *data = static_cast<uint8_t *>(this->_byte_array);
    for (size_t i = 0; i < this->_size; ++i) {
      array[(Json::ArrayIndex)i] = (Json::Value::Int)data[i];
    }
    request_json["wdata"] = array;
    Json::StreamWriterBuilder builder;
    builder["indentation"] = ""; // minimize output
    const std::string request = Json::writeString(builder, request_json);
    accl_send_log("sync to device", request);
    this->socket->send(zmq::const_buffer(request.c_str(), request.size()),
                       zmq::send_flags::none);

    zmq::message_t reply;
    zmq::recv_result_t result =
        this->socket->recv(reply, zmq::recv_flags::none);
    check_return_status(reply);
  }

  void free_buffer() override { return; }

  std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override {
    return std::unique_ptr<BaseBuffer>(
        new SimBuffer(&this->buffer[start], end - start, this->_type,
                      this->socket, this->_physical_address + start));
  }
};
} // namespace ACCL
