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

#include "accl/fpgadevice.hpp"
#include "accl/common.hpp"


namespace ACCL {
FPGADevice::FPGADevice(xrt::ip &cclo_ip, xrt::kernel &hostctrl_ip)
    : cclo(cclo_ip), hostctrl(hostctrl_ip) {}

void FPGADevice::start(const Options &options) {
  if (run) {
    throw std::runtime_error(
        "Error, collective is already running, wait for previous to complete!");
  }
  int function;
  if (options.scenario == operation::config) {
    function = static_cast<int>(options.cfg_function);
  } else {
    function = static_cast<int>(options.reduce_function);
  }
  run = hostctrl(static_cast<uint32_t>(options.scenario),
                 static_cast<uint32_t>(options.count),
                 static_cast<uint32_t>(options.comm),
                 static_cast<uint32_t>(options.root_src_dst),
                 static_cast<uint32_t>(function),
                 static_cast<uint32_t>(options.tag),
                 static_cast<uint32_t>(options.arithcfg_addr),
                 static_cast<uint32_t>(options.compression_flags),
                 static_cast<uint32_t>(options.stream_flags),
                 static_cast<uint64_t>(options.addr_0->physical_address()),
                 static_cast<uint64_t>(options.addr_1->physical_address()),
                 static_cast<uint64_t>(options.addr_2->physical_address()));
}

void FPGADevice::wait() {
  if (run) {
    run.wait();
    run = xrt::run();
  }
}

CCLO::timeoutStatus FPGADevice::wait(std::chrono::milliseconds timeout) {
  if (run) {
    auto status = run.wait(timeout);
    if (status == ert_cmd_state::ERT_CMD_STATE_TIMEOUT) {
      return CCLO::timeout;
    }
    run = xrt::run();
  }

  return CCLO::no_timeout;
}

void FPGADevice::call(const Options &options) {
  start(options);
  wait();
}

val_t FPGADevice::read(addr_t offset) { return cclo.read_register(offset); }

void FPGADevice::write(addr_t offset, val_t val) {
  return cclo.write_register(offset, val);
}
} // namespace ACCL
