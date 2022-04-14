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

/** @file fpgabuffer.hpp */

namespace ACCL {
template <typename dtype> class FPGABuffer : public Buffer<dtype> {
public:
  FPGABuffer(dtype *buffer, addr_t length, dataType type, xrt::device &device,
             xrt::memory_group mem_grp)
      : Buffer<dtype>(nullptr, length, type, 0x0),
        bo(device, get_aligned_buffer(buffer, length * sizeof(dtype)),
           length * sizeof(dtype), mem_grp) {
    set_buffer();
  }

  FPGABuffer(addr_t length, dataType type, xrt::device &device,
             xrt::memory_group mem_grp)
      : Buffer<dtype>(nullptr, length, type, 0x0),
        bo(device, length * sizeof(dtype), mem_grp), is_aligned(true) {
    set_buffer();
    // Initialize memory to zero
    memset(this->_buffer, 0, this->_size);
  }

  // copy constructor
  FPGABuffer(xrt::bo bo_, addr_t length, dataType type, bool is_aligned_,
             dtype *unaligned_buffer_)
      : Buffer<dtype>(nullptr, length, type, 0x0), bo(bo_),
        is_aligned(is_aligned_), aligned_buffer(bo.map<dtype *>()),
        unaligned_buffer(unaligned_buffer_) {
    set_buffer();
  }

  ~FPGABuffer() override {
    if (!is_aligned) {
#if (__cplusplus >= 201703L)
      std::free(aligned_buffer);
#else
      free(aligned_buffer);
#endif
    }
  }

  void sync_from_device() override {
    bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
    if (!is_aligned) {
      memcpy(unaligned_buffer, aligned_buffer, this->size());
    }
  }

  void sync_to_device() override {
    if (!is_aligned) {
      memcpy(aligned_buffer, unaligned_buffer, this->size());
    }
    bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void free_buffer() override { return; }

  std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override {
    size_t start_bytes = start * sizeof(dtype);
    size_t end_bytes = end * sizeof(dtype);

    dtype *offset_unaligned_buffer = nullptr;
    if (!is_aligned) {
      offset_unaligned_buffer = &unaligned_buffer[start];
    }

    return std::unique_ptr<BaseBuffer>(new FPGABuffer(
        xrt::bo(bo, end_bytes - start_bytes, start_bytes), end - start,
        this->_type, this->is_aligned, offset_unaligned_buffer));
  }

private:
  xrt::bo bo;
  bool is_aligned;
  dtype *aligned_buffer;
  dtype *unaligned_buffer;

  dtype *get_aligned_buffer(dtype *host_buffer, size_t length) {
    if ((reinterpret_cast<uintptr_t>(host_buffer) % ALIGNMENT) != 0) {
      std::cerr
          << "[ACCL] Warning: you're creating a buffer from an unaligned host "
             "pointer. This will cause extra copying when syncing the buffers."
          << std::endl;
      size_t aligned_size =
          ((size_t)ceil(length * sizeof(dtype) / (double)ALIGNMENT)) *
          ALIGNMENT;
      is_aligned = false;
#if  (__cplusplus >= 201703L)
      aligned_buffer =
          static_cast<dtype *>(std::aligned_alloc(ALIGNMENT, aligned_size));
#else
      dtype *aligned_buffer;
      posix_memalign(static_cast<void **>(&aligned_buffer), ALIGNMENT, aligned_size));
#endif
      unaligned_buffer = host_buffer;
      return aligned_buffer;
    }

    is_aligned = true;
    return host_buffer;
  }

  // Set the buffer after initialization since bo needs to be initialized first,
  // but base constructor is called beforehand.
  void set_buffer() { this->update_buffer(bo.map<dtype *>(), bo.address()); }
};
} // namespace ACCL

#endif // ACCL_HARDWARE_SUPPORT
