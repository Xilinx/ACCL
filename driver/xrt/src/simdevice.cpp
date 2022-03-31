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

#include "simdevice.hpp"
#include "common.hpp"
#include <jsoncpp/json/json.h>

namespace ACCL {
SimDevice::SimDevice(std::string zmqadr) {
  debug("SimDevice connecting to ZMQ on " + zmqadr);

  this->context = zmq::context_t();
  this->socket = zmq::socket_t(this->context, zmq::socket_type::req);
  this->socket.connect(zmqadr);

  debug("SimDevice connected");
};

void SimDevice::start(const Options &options) {
  int function;
  Json::Value request_json;

  if (options.waitfor.size() != 0) {
    throw std::runtime_error("SimDevice does not support chaining");
  }

  if (options.scenario == operation::config) {
    function = static_cast<int>(options.cfg_function);
  } else {
    function = static_cast<int>(options.reduce_function);
  }

  request_json["type"] = 4;
  request_json["scenario"] = static_cast<int>(options.scenario);
  request_json["tag"] = options.tag;
  request_json["count"] = options.count;
  request_json["comm"] = options.comm;
  request_json["root_src_dst"] = options.root_src_dst;
  request_json["function"] = function;
  request_json["arithcfg"] = (Json::Value::UInt64)options.arithcfg_addr;
  request_json["compression_flags"] =
      static_cast<int>(options.compression_flags);
  request_json["stream_flags"] = static_cast<int>(options.stream_flags);
  request_json["addr_0"] =
      (Json::Value::UInt64)options.addr_0->physical_address();
  request_json["addr_1"] =
      (Json::Value::UInt64)options.addr_1->physical_address();
  request_json["addr_2"] =
      (Json::Value::UInt64)options.addr_2->physical_address();

  Json::StreamWriterBuilder builder;
  builder["indentation"] = ""; // minimize output
  const std::string message = Json::writeString(builder, request_json);
  accl_send_log("device start", message);

  zmq::message_t request(message.c_str(), message.size());
  this->socket.send(request, zmq::send_flags::none);
};

void SimDevice::wait() {
  zmq::message_t reply;
  zmq::recv_result_t result = this->socket.recv(reply, zmq::recv_flags::none);
  check_return_status(reply);
}

void SimDevice::call(const Options &options) {
  this->start(options);
  this->wait();
}

// MMIO read request  {"type": 0, "addr": <uint>}
// MMIO read response {"status": OK|ERR, "rdata": <uint>}
val_t SimDevice::read(addr_t offset) {
  Json::Value request_json;
  request_json["type"] = 0;
  request_json["addr"] = (Json::Value::UInt64)offset;
  Json::StreamWriterBuilder builder;
  builder["indentation"] = ""; // minimize output
  const std::string request = Json::writeString(builder, request_json);
  accl_send_log("device read", request);
  this->socket.send(zmq::const_buffer(request.c_str(), request.size()),
                    zmq::send_flags::none);

  zmq::message_t reply;
  zmq::recv_result_t result = this->socket.recv(reply, zmq::recv_flags::none);
  Json::Value reply_json = parse_json(reply.to_string());
  check_return_status(reply_json["status"]);
  return reply_json["rdata"].asUInt64();
}

// MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
// MMIO write response {"status": OK|ERR}
void SimDevice::write(addr_t offset, val_t val) {
  Json::Value request_json;
  request_json["type"] = 1;
  request_json["addr"] = (Json::Value::UInt64)offset;
  request_json["wdata"] = (Json::Value::UInt)val;
  Json::StreamWriterBuilder builder;
  builder["indentation"] = ""; // minimize output
  const std::string request = Json::writeString(builder, request_json);
  accl_send_log("device write", request);
  this->socket.send(zmq::const_buffer(request.c_str(), request.size()),
                    zmq::send_flags::none);

  zmq::message_t reply;
  zmq::recv_result_t result = this->socket.recv(reply, zmq::recv_flags::none);
  check_return_status(reply);
}
} // namespace ACCL
