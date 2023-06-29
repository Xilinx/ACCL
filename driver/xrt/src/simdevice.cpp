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

#include "accl/simdevice.hpp"
#include "accl/common.hpp"
#include <cassert>
#include <future>
#include "zmq_client.h"

static void finish_sim_request(ACCL::SimRequest *req) {
  ACCL::SimDevice *cclo = reinterpret_cast<ACCL::SimDevice *>(req->cclo());
  // wait on zmq context
  zmq_client_retcall(cclo->get_context());
  // get ret code before notifying waiting theads
  req->set_retcode(cclo->read(ACCL::RETCODE_OFFSET));
  req->set_status(ACCL::operationStatus::COMPLETED);
  req->notify();
  cclo->complete_request(req);
}

namespace ACCL {
void SimRequest::start() {
  assert(this->get_status() ==  operationStatus::EXECUTING);

  int function;

  options.addr_0->sync_bo_to_device();
  options.addr_1->sync_bo_to_device();
  options.addr_2->sync_bo_to_device();

  if (options.scenario == operation::config) {
    function = static_cast<int>(options.cfg_function);
  } else {
    function = static_cast<int>(options.reduce_function);
  }

  zmq_client_startcall(
      reinterpret_cast<SimDevice *>(cclo_ptr)->get_context(),
      static_cast<int>(options.scenario), options.tag, options.count,
      options.comm, options.root_src_dst, function, options.arithcfg_addr,
      static_cast<int>(options.compression_flags),
      static_cast<int>(options.stream_flags),
      options.addr_0->physical_address(), options.addr_1->physical_address(),
      options.addr_2->physical_address());

  // launch an async wait simulating a callback for completing
  auto f = std::async(std::launch::async, finish_sim_request, this);
}

ACCLRequest *SimDevice::start(const Options &options) {
  if (options.waitfor.size() != 0) {
    throw std::runtime_error("SimDevice does not support chaining");
  }
  
  SimRequest *req = new SimRequest(reinterpret_cast<void *>(this), options);
 
  queue.push(req);
  req->set_status(operationStatus::QUEUED);

  launch_request();

  return req;
};

SimDevice::SimDevice(unsigned int zmqport, unsigned int local_rank) {
  debug("SimDevice connecting to ZMQ on port " + std::to_string(zmqport) +
        " for rank " + std::to_string(local_rank));
  zmq_ctx = zmq_client_intf(zmqport, local_rank);
  debug("SimDevice connected");
};

void SimDevice::wait(ACCLRequest *request) { 
  request->wait(); 
}

timeoutStatus SimDevice::wait(ACCLRequest *request,
                              std::chrono::milliseconds timeout) {
  if (request->wait(timeout))
    return timeoutStatus::no_timeout;

  return timeoutStatus::timeout;
}

bool SimDevice::test(ACCLRequest *request) {
  return request->get_status() == operationStatus::COMPLETED;
}

void SimDevice::free_request(ACCLRequest *request) {
  delete request;
}

ACCLRequest *SimDevice::call(const Options &options) {
  ACCLRequest *req = this->start(options);
  this->wait(req);
  
  return req;
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

void SimDevice::launch_request() {
  // This guarantees permission to only one thread trying to start an operation
  if (queue.run()) {
    SimRequest *req = queue.front();
    assert(req->get_status() == operationStatus::QUEUED);
    req->set_status(operationStatus::EXECUTING);
    req->start();
  }
}

void SimDevice::complete_request(SimRequest *request) {
  // Avoid user from completing requests
  if (request->get_status() == operationStatus::COMPLETED) {
    const Options &options = request->get_options();

    options.addr_0->sync_bo_from_device();
    options.addr_1->sync_bo_from_device();
    options.addr_2->sync_bo_from_device();

    queue.pop();
    launch_request();
  }
}
} // namespace ACCL
