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
#include <experimental/xrt_ip.h>
#include <unordered_map>
#include <string>
#include <xrt/xrt_kernel.h>

/** @file fpgadevice.hpp */

namespace ACCL {
/**
 * Implementation of the request derived class for FPGA devices
 *
 */
class FPGARequest : public BaseRequest {
private:
  xrt::run run;

public:
  const CCLO::Options &options;

public:
  /**
   * Construct a new FPGA request object
   *
   * @param cclo   Opaque reference to the main CCLO object
   * @param kernel Hostctrl kernel to be used in the request
   */
  FPGARequest(void *cclo, xrt::kernel &kernel, const CCLO::Options &options)
      : BaseRequest(cclo), run(xrt::run(kernel)), options(options) {}

  /**
   * Effectively starts the call
   * 
   */
  void start();

  void wait_kernel() {
    run.wait();
  }
};

/**
 * Implementation of CCLO that uses a CCLO kernel on a FPGA.
 *
 */
class FPGADevice : public CCLO {
public:
  /**
   * Construct a new FPGADevice object
   *
   * @param cclo_ip      The CCLO kernel to use.
   * @param hostctrl_ip  The hostctrl kernel to use.
   * @param device       Xrt device;
   */
  FPGADevice(xrt::ip &cclo_ip, xrt::kernel &hostctrl_ip, xrt::device &device);

  /**
   * Destroy the FPGADevice object
   *
   */
  virtual ~FPGADevice() {}

  ACCLRequest *call(const Options &options) override;

  ACCLRequest *start(const Options &options) override;

  val_t read(addr_t offset) override;

  void write(addr_t offset, val_t val) override;

  void wait(ACCLRequest *request) override;

  timeoutStatus wait(ACCLRequest *request,
                     std::chrono::milliseconds timeout) override;
  
  bool test(ACCLRequest *request) override;
  
  uint64_t get_duration(ACCLRequest *request);

  void free_request(ACCLRequest *request) override;

  val_t get_retcode(ACCLRequest *request) override;

  addr_t get_base_addr() override {
    // TODO: Find way to retrieve CCLO base address on FPGA
    return 0x0;
  }

  /**
   * Internally completes the request.
   * 
   * @param request Associated request to be completed
   */
  void complete_request(FPGARequest *request);

  deviceType get_device_type() override;

  void printDebug() override { return; }

  xrt::device* get_device() {
    return &device;
  }

private:
  xrt::ip cclo;
  xrt::kernel hostctrl;
  xrt::device device;

  FPGAQueue<FPGARequest *> queue;
  std::unordered_map<ACCLRequest, FPGARequest *> request_map;

  /**
   * Starts the execution of the first request in the queue. To keep queue
   * going, this function is called every time a operation is issue and
   * everytime an ongoing operation finishes
   *
   */
  void launch_request();
};
} // namespace ACCL
