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
template <typename dtype> class FPGABufferP2P : public Buffer<dtype> {
public:
  FPGABufferP2P(xrt::bo &bo, addr_t length, dataType type)
      : Buffer<dtype>(bo.map<dtype *>(), length, type, bo.address()), _bo(bo) {
    set_buffer();
  }

  FPGABufferP2P(addr_t length, dataType type, xrt::device &device,
                xrt::memory_group mem_grp)
      : Buffer<dtype>(nullptr, length, type, 0x0),
        _bo(device, length * sizeof(dtype), xrt::bo::flags::p2p, mem_grp) {
    set_buffer();
  }

  // copy constructor
  FPGABufferP2P(xrt::bo bo_, addr_t length, dataType type)
      : Buffer<dtype>(nullptr, length, type, 0x0), _bo(bo_) {
    set_buffer();
  }

  virtual ~FPGABufferP2P() {}

  xrt::bo *bo() override { return &_bo; }

  bool is_simulated() const override { return false; }

  void sync_from_device() override {
    // Not applicable for p2p buffer
  }

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
