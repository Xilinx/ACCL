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
#include "cProcess.hpp"
#include "ibvQpConn.hpp"
#include "ibvStructs.hpp"
#include <string>
#include <iostream>
#include <fstream>

constexpr int targetRegion = 0;

/** @file coyotedevice.hpp */

namespace ACCL {

/**
 * Implementation of the request derived class for FPGA devices
 *
 */
class CoyoteRequest : public BaseRequest {
private:
  const size_t OFFSET_HOSTCTRL = 0x2000; 
  const CCLO::Options &options;

public:

  /**
   * Construct a new FPGA request object
   *
   * @param cclo    Opaque reference to the main CCLO object
   * @param options Options for the associated call
   */
  CoyoteRequest(void *cclo, const CCLO::Options &options)
      : BaseRequest(cclo), options(options) {}

  /**
   * Effectively starts the call
   * 
   */
  void start();

  /**
   * Waits for call to end
   * 
   */
  void wait_kernel();
};

/**
 * Implementation of CCLO that uses a CCLO kernel on a FPGA.
 *
 */
class CoyoteDevice : public CCLO {
public:
  CoyoteDevice();
  CoyoteDevice(unsigned int num_qp);
  /**
   * Destroy the CoyoteDevice object
   *
   */
  ~CoyoteDevice();

  ACCLRequest *call(const Options &options) override;

  ACCLRequest *start(const Options &options) override;

  val_t read(addr_t offset) override;

  void write(addr_t offset, val_t val) override;

  void wait(ACCLRequest *request) override;

  timeoutStatus wait(ACCLRequest *request,
                std::chrono::milliseconds timeout) override;

  /**
   * Internally completes the request.
   * 
   * @param request Associated request to be completed
   */
  void complete_request(CoyoteRequest *request);

  addr_t get_base_addr() override {
    return OFFSET_CCLO;
  }

  deviceType get_device_type() override;

  void printDebug() override;

  fpga::cProcess* get_device(){
    return coyote_proc;
  }

  bool test(ACCLRequest *request) override;
  
  uint64_t get_duration(ACCLRequest *request);

  void free_request(ACCLRequest *request) override;

  val_t get_retcode(ACCLRequest *request) override;

  fpga::cProcess* coyote_proc;

  // RDMA related 
  // RDMA requires multiple processes to establish queue pairs
  // The CCLO kernel is still managed by coyote_proc
  unsigned int num_qp;
  std::vector<fpga::cProcess*> coyote_qProc_vec;
private:
  const size_t OFFSET_CCLO = 0x0; 

  FPGAQueue<CoyoteRequest *> queue;
  std::unordered_map<ACCLRequest, CoyoteRequest *> request_map;

  /**
   * Starts the execution of the first request in the queue. To keep queue
   * going, this function is called every time a operation is issue and
   * everytime an ongoing operation finishes
   *
   */
  void launch_request();

};
} // namespace ACCL
