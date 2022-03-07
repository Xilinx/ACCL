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
#include "buffer.hpp"
#include "constants.hpp"
#include <vector>

/** @file cclo.hpp */

namespace ACCL {

/**
 * Abstract class that defines operations to be performed on the CCLO on the
 * FPGA.
 *
 */
class CCLO {
public:
  /**
   * CCLO options for call and start.
   *
   * TODO: Describe options.
   *
   */
  struct Options {
    operation scenario;
    unsigned int count;
    unsigned int comm;
    unsigned int root_src_dst;
    cfgFunc cfg_function;
    reduceFunction reduce_function;
    unsigned int tag;
    addr_t arithcfg_addr;
    dataType compress_dtype;
    compressionFlags compression_flags;
    streamFlags stream_flags;
    BaseBuffer *addr_0;
    BaseBuffer *addr_1;
    BaseBuffer *addr_2;
    std::vector<CCLO *> waitfor;

    /**
     * Construct a new CCLO Options object with default parameters.
     *
     */
    Options()
        : scenario(operation::nop), count(1), comm(0), root_src_dst(0),
          cfg_function(cfgFunc::reset_periph),
          reduce_function(reduceFunction::SUM), tag(TAG_ANY),
          arithcfg_addr(0x0), compress_dtype(dataType::none),
          compression_flags(compressionFlags::NO_COMPRESSION),
          stream_flags(streamFlags::NO_STREAM), addr_0(nullptr),
          addr_1(nullptr), addr_2(nullptr), waitfor({}) {}
  };

  /**
   * Construct a new CCLO object.
   *
   */
  CCLO() {}

  /**
   * Call a CCLO operation based on the options.
   *
   * @param options  Specifies CCLO operation and configuration.
   */
  virtual void call(const Options &options) = 0;

  /**
   * Call a CCLO operation based on the options, and wait for operation to
   * complete.
   *
   * @param options  Specifies CCLO operation and configuration.
   */
  virtual void start(const Options &options) = 0;

  /**
   * Read data from FPGA at specified address.
   *
   * @param offset  Offset to read data from on the FPGA.
   * @return val_t  Data located at specified address on FPGA.
   */
  virtual val_t read(addr_t offset) = 0;

  /**
   * Write data to specified address on FPGA.
   *
   * @param offset  Offset to write data to on the FPGA.
   * @param val     Value to write at specified address on FPGA.
   */
  virtual void write(addr_t offset, val_t val) = 0;

  /**
   * Wait for CCLO operation to complete.
   *
   */
  virtual void wait() = 0;

  virtual addr_t get_base_addr() = 0;
};
} // namespace ACCL
