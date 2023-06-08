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
#include <string>

/** @file common.hpp */

#define ACCL_FPGA_ALIGNMENT 4096

namespace ACCL {

/**
 * Print debug message
 *
 * @param message Message to print.
 */
inline void debug(std::string message) {
#ifdef ACCL_DEBUG
  std::cerr << message << std::endl;
#endif
}

/**
 * Format value as hexadecimal number in string.
 *
 * @param value        Value to format.
 * @return std::string Value formatted as hexadecimal number.
 */
inline std::string debug_hex(addr_t value) {
#ifdef ACCL_DEBUG
  std::stringstream stream;
  stream << std::hex << value;
  return stream.str();
#else
  return "";
#endif
}

/**
 * Allocate aligned buffer.
 *
 * @param size       Size of aligned buffer.
 * @param alignment  Amount of bits to align to.
 * @return void*     The aligned buffer.
 */
void *allocate_aligned_buffer(size_t size,
                              size_t alignment = ACCL_FPGA_ALIGNMENT);

/**
 * Reset the log file.
 *
 */
#ifdef ACCL_DEBUG
void reset_log();
#else
inline void reset_log() {
  // NOP
}
#endif

/**
 * Append message to log file.
 *
 * @param label   Label of message.
 * @param message Message to write to log file.
 */
#ifdef ACCL_DEBUG
void accl_log(int rank, const std::string &message);
#else
inline void accl_log(int rank, const std::string &message) {
  // NOP
}
#endif

/**
 * Append message to log file.
 *
 * @param label   Label of message.
 * @param message Message to write to log file.
 */
#ifdef ACCL_DEBUG
void accl_send_log(const std::string &label, const std::string &message);
#else
inline void accl_send_log(const std::string &label,
                          const std::string &message) {
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

/**
 * Encode an IP address from string to integer.
 *
 * @param ip       IP string to encode.
 */
uint32_t ip_encode(std::string ip);

/**
 * Decode an IP address from integer to string.
 *
 * @param ip       Encoded IP integer to decode.
 */
std::string ip_decode(uint32_t ip);

} // namespace ACCL
