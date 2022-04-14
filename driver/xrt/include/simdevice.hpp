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
#include "cclo.hpp"
#include "constants.hpp"
#include <string>
#include <zmq.hpp>
#include "zmq_client.h"

/** @file simdevice.hpp */

namespace ACCL {
/**
 * Implementation of CCLO that uses an external CCLO simulator or emulator.
 *
 */
class SimDevice : public CCLO {
public:
  /**
   * Construct a new Simulated Device object.
   *
   * @param zmqadr  Address of simulator or emulator to connect to.
   */
  SimDevice(unsigned int zmqport, unsigned int local_rank);

  /**
   * See ACCL::CCLO::call().
   *
   */
  void call(const Options &options) override;

  /**
   * See ACCL::CCLO::start().
   *
   */
  void start(const Options &options) override;

  val_t read(addr_t offset) override;

  void write(addr_t offset, val_t val) override;

  void wait() override;

  addr_t get_base_addr() override { return 0x0; }

  zmq_intf_context *get_context() { return &zmq_ctx; }

private:
  zmq_intf_context zmq_ctx;
};
} // namespace ACCL
