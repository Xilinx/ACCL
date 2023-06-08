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
#include "buffer.hpp"
#include "common.hpp"
#include <math.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

// Use posix_memalign if C++17 is not available
#if (__cplusplus >= 201703L)
#include <cstdlib>
#else
#include <stdlib.h>
#endif

#define ALIGNMENT 4096

/** @file fpgabufferp2p.hpp */

namespace ACCL {
/**
 * A P2P buffer that is allocated on the FPGA and can be accessed from the host
 * without explicit copying.
 *
 * @tparam dtype Datatype of the buffer.
 */
template <typename dtype> class FPGABufferP2P : public Buffer<dtype> {
public:
  /**
   * Construct a new FPGABufferP2P object from an existing P2P BO buffer.
   *
   * @param bo     An existing P2P BO buffer.
   * @param length Amount of elements in the P2P buffer.
   * @param type   ACCL datatype of the P2P buffer.
   */
  FPGABufferP2P(xrt::bo &bo, addr_t length, dataType type)
      : Buffer<dtype>(bo.map<dtype *>(), length, type, bo.address()), _bo(bo) {
    set_buffer();
  }

  /**
   * Construct a new FPGABufferP2P object without any existing buffer.
   *
   * @param length  Amount of elements to allocate for.
   * @param type    ACCL datatype of the P2P buffer.
   * @param device  Device to allocate the P2P buffer on.
   * @param mem_grp Memory bank on device to allocate the P2P buffer on.
   */
  FPGABufferP2P(addr_t length, dataType type, xrt::device &device,
                xrt::memory_group mem_grp)
      : Buffer<dtype>(nullptr, length, type, 0x0),
        _bo(device, length * sizeof(dtype), xrt::bo::flags::p2p, mem_grp) {
    set_buffer();
  }

  /**
   * Copy construct of a P2P buffer for internal use only.
   *
   */
  FPGABufferP2P(xrt::bo bo_, addr_t length, dataType type)
      : Buffer<dtype>(nullptr, length, type, 0x0), _bo(bo_) {
    set_buffer();
  }

  /**
   * Destroy the FPGABufferP2P object
   *
   */
  virtual ~FPGABufferP2P() {}

  /**
   * Return the underlying P2P BO buffer.
   *
   * @return xrt::bo* The underlying P2P BO buffer.
   */
  xrt::bo *bo() { return &_bo; }

  /**
   * Check if the buffer is simulated, always false.
   *
   */
  bool is_simulated() const override { return false; }

  /**
   * Check if the buffer is host-only, always false
   *
   */
  bool is_host_only() const override { return false; }

  /**
   * Sync the data from the device back to the host, which is not required with
   * a P2P buffer, so this function does nothing.
   *
   */
  void sync_from_device() override {
    // Not applicable for p2p buffer
  }

  /**
   * Sync the data from the host to the device, which is not required with a P2P
   * buffer, so this function does nothing.
   *
   */
  void sync_to_device() override {
    // Not applicable for p2p buffer
  }

  void free_buffer() override { return; }

  std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override {
    size_t start_bytes = start * sizeof(dtype);
    size_t end_bytes = end * sizeof(dtype);

    return std::unique_ptr<BaseBuffer>(
        new FPGABufferP2P(xrt::bo(_bo, end_bytes - start_bytes, start_bytes),
                          end - start, this->_type));
  }

private:
  xrt::bo _bo;

  // Set the buffer after initialization since bo needs to be initialized first,
  // but base constructor is called beforehand.
  void set_buffer() { this->update_buffer(_bo.map<dtype *>(), _bo.address()); }
};
} // namespace ACCL
