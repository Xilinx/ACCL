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
#include "acclrequest.hpp"
#include "cclo.hpp"
#include "constants.hpp"
#include <string>
#include "zmq_client.h"
#include <experimental/xrt_ip.h>
#include <xrt/xrt_kernel.h>

/** @file simdevice.hpp */

namespace ACCL {
/**
 * Implementation of the derived request class associated with Sim Devices
 * 
 */
class SimRequest : public BaseRequest {
public:
  /**
   * Constructs a new request for Sim Devices
   * 
   * @param cclo      Opaque reference to the main CCLO object
   * @param options   Constant reference to the associated options
   */
  SimRequest(void *cclo, const CCLO::Options &options)
      : BaseRequest(cclo), options(options) {}

  /**
   * Return the operation associated options 
   * 
   * @return const CCLO::Options& Constant reference to the associated options
   */
  const CCLO::Options &get_options() {
    return options;
  }

  /**
   * Effectively starts the call
   * 
   */
  void start();

private:
  const CCLO::Options &options;
};

/**
 * Implementation of CCLO that uses an external CCLO simulator or emulator.
 *
 */
class SimDevice : public CCLO {
public:
  /**
   * Construct a new Simulated Device object.
   *
   * @param zmqport    Port of simulator or emulator to connect to.
   * @param local_rank The local rank of this process.
   */
  SimDevice(unsigned int zmqport, unsigned int local_rank);

  /**
   * Destroy the Simulated Device object
   *
   */
  virtual ~SimDevice() {}

  ACCLRequest *call(const Options &options) override;

  ACCLRequest *start(const Options &options) override;

  val_t read(addr_t offset) override;

  void write(addr_t offset, val_t val) override;

  void wait(ACCLRequest *request) override;

  timeoutStatus wait(ACCLRequest *request,
                     std::chrono::milliseconds timeout) override;
  
  bool test(ACCLRequest *request) override;
  
  uint64_t get_duration(ACCLRequest *request) override;

  void free_request(ACCLRequest *request) override;

  val_t get_retcode(ACCLRequest *request) override;

  addr_t get_base_addr() override { return 0x0; }

  void printDebug() override { return; }

  /**
   * Get the zmq server used by the CCLO emulator or simulator.
   *
   * @return zmq_intf_context* The zmq server used by the CCLO.
   */
  zmq_intf_context *get_context() { return &zmq_ctx; }

  /**
   * Internally completes the request.
   *
   * @param request Associated request to be completed
   */
  void complete_request(SimRequest *request);

  deviceType get_device_type() override;

  xrt::device* get_device() {
    return &device;
  }

private:
  xrt::device device;
  zmq_intf_context zmq_ctx;
  
  FPGAQueue<SimRequest *> queue;
  std::unordered_map<ACCLRequest, SimRequest *> request_map;

  /**
   * Starts the execution of the first request in the queue. To keep queue
   * going, this function is called every time a operation is issue and
   * everytime an ongoing operation finishes
   *
   */
  void launch_request();
};
} // namespace ACCL
