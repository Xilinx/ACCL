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
#ifdef ACCL_HARDWARE_SUPPORT
#include "cclo.hpp"
#include "constants.hpp"
#include <experimental/xrt_ip.h>
#include <string>
#include <xrt/xrt_kernel.h>

/** @file fpgadevice.hpp */

namespace ACCL {
/**
 * Implementation of CCLO that uses an external CCLO simulator or emulator.
 *
 */
class FPGADevice : public CCLO {
public:
  FPGADevice(xrt::ip &cclo_ip, xrt::kernel &hostctrl_ip);

  ~FPGADevice() override {}

  void call(const Options &options) override;

  void start(const Options &options) override;

  val_t read(addr_t offset) override;

  void write(addr_t offset, val_t val) override;

  void wait() override;

  addr_t get_base_addr() override {
    // TODO: Find way to retrieve CCLO base address on FPGA
    return 0x0;
  }

private:
  xrt::ip cclo;
  xrt::kernel hostctrl;
  xrt::run run{};
};
} // namespace ACCL

#endif // ACCL_HARDWARE_SUPPORT
