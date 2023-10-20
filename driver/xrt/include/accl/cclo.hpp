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
#include "arithconfig.hpp"
#include "buffer.hpp"
#include "constants.hpp"
#include <chrono>
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
   */
  struct Options {
    operation scenario;                 /**< Operation to perform on CCLO. */
    unsigned int count;                 /**< Amount of elements to perform
                                             operation on. */
    unsigned int comm;                  /**< Address of communicator to use. */
    unsigned int root_src_dst;          /**< Rank to use as root for the
                                             operation. */
    cfgFunc cfg_function;               /**< Configuration function to use for
                                             operation. */
    reduceFunction reduce_function;     /**< Reduce function to use for
                                             operation. */
    unsigned int tag;                   /**< Tag to use for send or receive. */
    addr_t arithcfg_addr;               /**< Address of the arithmetic
                                             configuration to use. */
    dataType compress_dtype;            /**< Compress buffers to this
                                             datatype. */
    compressionFlags compression_flags; /**< Compression configuration. */
    streamFlags stream_flags;           /**< Stream configuration. */
    hostFlags host_flags;               /**< Indicate which buffers are host-only. */
    BaseBuffer *addr_0;                 /**< ACCL buffer of operand 0. */
    BaseBuffer *addr_1;                 /**< ACCL buffer of operand 1. */
    BaseBuffer *addr_2;                 /**< ACCL buffer of result. */
    dataType data_type_io_0;            /**< Data type of ACCL input or output from stream. */
    dataType data_type_io_1;            /**< Data type of ACCL input or output from stream. */
    dataType data_type_io_2;            /**< Data type of ACCL input or output from stream. */
    std::vector<ACCLRequest *> waitfor; /**< Wait for these operations to
                                              complete; currently unsupported. */

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
          stream_flags(streamFlags::NO_STREAM), host_flags(hostFlags::NO_HOST),
          addr_0(nullptr), addr_1(nullptr), addr_2(nullptr),
          data_type_io_0(dataType::none), data_type_io_1(dataType::none),
          data_type_io_2(dataType::none), waitfor({}) {}
  };

  enum deviceType{
    sim_device,
    xrt_device,
    coyote_device
  };

  /**
   * Construct a new CCLO object.
   *
   */
  CCLO() {}

  virtual ~CCLO() {}

  /**
   * Call a CCLO operation based on the options.
   *
   * @param options       Specifies CCLO operation and configuration.
   * @return ACCLRequest* Request associated to the call. Internal use only on
   *                      sync calls;
   */
  virtual ACCLRequest *call(const Options &options) = 0;

  /**
   * Call a CCLO operation based on the options, and wait for operation to
   * complete.
   *
   * @param options       Specifies CCLO operation and configuration.
   * @return ACCLRequest* Request associated to the call.
   */
  virtual ACCLRequest *start(const Options &options) = 0;

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
   * Wait for CCLO operation associated with the request to complete.
   *
   * @param request Request to wait for
   */
  virtual void wait(ACCLRequest *request) = 0;

  /**
   * Wait for CCLO operation associated to the given request to complete. 
   * Note that the timeout is currently ignored when using the simulator.
   *
   * @param request   Request to wait for
   * @param timeout   Time out wait after this many milliseconds have passed.
   *
   * @return timeoutStatus Status on whether the wait was timed out or not.
   */
  virtual timeoutStatus wait(ACCLRequest *request, 
                             std::chrono::milliseconds timeout) = 0;

  /**
   * Tests if the given request completed
   * 
   * @param request   Request to wait for
   * @return true     If the request was completed
   * @return false    If the request did not complete
   */
  virtual bool test(ACCLRequest *request) = 0;

  /**
   * Free given request
   * 
   * @param request Request to be freed
   */
  virtual void free_request(ACCLRequest *request) = 0;
  
  /**
   * Gets the return code of a given request
   * 
   * @param request Request to be look up
   */
  virtual val_t get_retcode(ACCLRequest *request) = 0;

  /**
   * Gets the duration of a given request in nanoseconds
   * 
   * @param request Request to be look up
   * @return uint64_t The duration in nanoseconds.
   */
  virtual uint64_t get_duration(ACCLRequest *request) = 0;

  /**
   * Get the base address of the CCLO, this currently returns 0x0 on hardware.
   *
   * @return addr_t The base address of the CCLO.
   */
  virtual addr_t get_base_addr() = 0;

  /**
   * Get the device type.
   *
   * @return deviceType Whether this is a sim device, xrt device or coyote device.
   */
  virtual deviceType get_device_type() = 0;

  /**
   * printDebug.
   *
   */
  virtual void printDebug() = 0;
};
} // namespace ACCL
