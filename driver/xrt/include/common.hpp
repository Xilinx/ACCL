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
#include "arithconfig.hpp"
#include "cclo.hpp"
#include "constants.hpp"
#include <iostream>
#include <sstream>

/** @file common.hpp */

#define ACCL_SEND_LOG_FILE(i) "accl_send" + i + ".log"

namespace ACCL {

#ifdef ACCL_DEBUG
inline void debug(std::string message) { std::cerr << message << std::endl; }

inline std::string debug_hex(addr_t value) {
  std::stringstream stream;
  stream << std::hex << value;
  return stream.str();
}

void reset_log();

void accl_send_log(const std::string &label, const std::string &message);
#else
inline void debug(std::string message) {
  // NOP
}

inline std::string debug_hex(addr_t value) { return ""; }

inline void accl_send_log(const std::string &label,
                          const std::string &message) {
  // NOP
}

inline void reset_log() {
  // NOP
}
#endif

/**
 * Writes the arithmetic configuration to the CCLO on the FPGA at address
 * location.
 *
 * @param cclo     CCLO object to write arithmetic configuration to.
 * @param arithcfg Arithmetic configuration to write.
 * @param addr     Address on the FPGA to write arithmetic configuration to.
 */
void write_arithconfig(CCLO &cclo, ArithConfig &arithcfg, addr_t *addr);

} // namespace ACCL
