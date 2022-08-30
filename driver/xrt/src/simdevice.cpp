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
#include "zmq_client.h"

namespace ACCL {
SimDevice::SimDevice(unsigned int zmqport, unsigned int local_rank) {
  debug("SimDevice connecting to ZMQ on port " + std::to_string(zmqport) +
        " for rank " + std::to_string(local_rank));
  zmq_ctx = zmq_client_intf(zmqport, local_rank);
  debug("SimDevice connected");
};

void SimDevice::start(const Options &options) {
  int function;

  options.addr_0->sync_bo_to_device();
  options.addr_1->sync_bo_to_device();
  options.addr_2->sync_bo_to_device();
  addr_0_cache = options.addr_0;
  addr_1_cache = options.addr_1;
  addr_2_cache = options.addr_2;

  if (options.waitfor.size() != 0) {
    throw std::runtime_error("SimDevice does not support chaining");
  }

  if (options.scenario == operation::config) {
    function = static_cast<int>(options.cfg_function);
  } else {
    function = static_cast<int>(options.reduce_function);
  }

  zmq_client_startcall(
      &(this->zmq_ctx), static_cast<int>(options.scenario), options.tag,
      options.count, options.comm, options.root_src_dst, function,
      options.arithcfg_addr, static_cast<int>(options.compression_flags),
      static_cast<int>(options.stream_flags),
      options.addr_0->physical_address(), options.addr_1->physical_address(),
      options.addr_2->physical_address());
};

void SimDevice::wait() {
  zmq_client_retcall(&(this->zmq_ctx));
  addr_0_cache->sync_bo_from_device();
  addr_1_cache->sync_bo_from_device();
  addr_2_cache->sync_bo_from_device();
}

void SimDevice::call(const Options &options) {
  this->start(options);
  this->wait();
}

// MMIO read request  {"type": 0, "addr": <uint>}
// MMIO read response {"status": OK|ERR, "rdata": <uint>}
val_t SimDevice::read(addr_t offset) {
  return zmq_client_cfgread(&(this->zmq_ctx), offset);
}

// MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
// MMIO write response {"status": OK|ERR}
void SimDevice::write(addr_t offset, val_t val) {
  zmq_client_cfgwrite(&(this->zmq_ctx), offset, val);
}
} // namespace ACCL
