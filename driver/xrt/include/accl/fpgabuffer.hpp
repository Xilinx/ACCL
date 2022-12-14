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
#include <cstdlib>
#include <cstring>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

/** @file fpgabuffer.hpp */

namespace ACCL {
/**
 * A buffer that is allocated on the FPGA with an accompanying host pointer.
 *
 * The host pointer will be aligned to 4096 bytes. If a non-aligned host pointer
 * is provided, ACCL will keep it's own aligned host buffer, and copy between
 * the unaligned and aligned host buffers when required. It is recommended to
 * provide an aligned host pointer to avoid unnecessary memory copies.
 *
 * @tparam dtype Datatype of the buffer.
 */
template <typename dtype> class FPGABuffer : public Buffer<dtype> {
public:
  /**
   * Construct a new FPGABuffer object from an existing host pointer.
   *
   * If a non-aligned host pointer is provided, ACCL will keep it's own aligned
   * host buffer, and copy between the unaligned and aligned host buffers when
   * required. It is recommended to provide an aligned host pointer to avoid
   * unnecessary memory copies.
   *
   * @param buffer  The host pointer containing the data.
   * @param length  Amount of elements in the host buffer.
   * @param type    ACCL datatype of buffer.
   * @param device  Device to allocate the buffer on.
   * @param mem_grp Memory bank on the device to allocate the buffer on.
   */
  FPGABuffer(dtype *buffer, addr_t length, dataType type, xrt::device &device,
             xrt::memory_group mem_grp)
      : Buffer<dtype>(nullptr, length, type, 0x0),
        _bo(device, get_aligned_buffer(buffer, length * sizeof(dtype)),
            length * sizeof(dtype), mem_grp) {
    set_buffer();
  }

  /**
   * Construct a new FPGABuffer object from an existing BO buffer.
   *
   * No new buffer is allocated when using this constructor, instead ACCL will
   * use the existing BO buffer.
   *
   * @param bo     Existing BO buffer to use.
   * @param length Amount of elements to allocate for.
   * @param type   ACCL datatype of buffer.
   */
  FPGABuffer(xrt::bo &bo, addr_t length, dataType type)
      : Buffer<dtype>(bo.map<dtype *>(), length, type, bo.address()), _bo(bo) {
    set_buffer();
  }

  /**
   * Construct a new FPGABuffer object without an existing host pointer.
   *
   * This constructor will allocate a buffer on both the host and the FPGA.
   *
   * @param length  Amount of elements to allocate the buffers for.
   * @param type    ACCL datatype of the buffer.
   * @param device  Device to allocate the FPGA buffer on.
   * @param mem_grp Memory bank of the device to allocate the FPGA buffer on.
   */
  FPGABuffer(addr_t length, dataType type, xrt::device &device,
             xrt::memory_group mem_grp)
      : Buffer<dtype>(nullptr, length, type, 0x0), is_aligned(true),
        _bo(device, length * sizeof(dtype), mem_grp) {
    set_buffer();
    // Initialize memory to zero
    std::memset(this->_buffer, 0, this->_size);
  }

  /**
   * Copy construct of an FPGA buffer for internal use only.
   *
   */
  FPGABuffer(xrt::bo bo_, addr_t length, dataType type, bool is_aligned_,
             dtype *unaligned_buffer_)
      : Buffer<dtype>(nullptr, length, type, 0x0), is_aligned(is_aligned_),
        _bo(bo_), aligned_buffer(_bo.map<dtype *>()),
        unaligned_buffer(unaligned_buffer_) {
    set_buffer();
  }

  /**
   * Destroy the FPGABuffer object
   *
   */
  virtual ~FPGABuffer() {
    // Only free the aligned buffer if it exists and we own it (might not be
    // the case if this is a slice).
    if (!is_aligned && own_unaligned) {
      std::free(aligned_buffer);
    }
  }

  /**
   * Return the underlying BO buffer.
   *
   * @return xrt::bo* The underlying BO buffer.
   */
  xrt::bo *bo() override { return &_bo; }

  /**
   * Check if the buffer is simulated, always false.
   *
   */
  bool is_simulated() const override { return false; }

  /**
   * Sync the data from the device back to the host. Will copy the data from
   * the aligned buffer to the unaligned buffer if an unaligned buffer was used
   * during construction of the FPGABuffer.
   *
   */
  void sync_from_device() override {
    _bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
    if (!is_aligned) {
      std::memcpy(unaligned_buffer, aligned_buffer, this->size());
    }
  }

  /**
   * Sync the data from the host to the device. Will copy the data from the
   * unaligned buffer to the aligned buffer first if an unaligned buffer was
   * used during construction of the FPGABuffer.
   *
   */
  void sync_to_device() override {
    if (!is_aligned) {
      std::memcpy(aligned_buffer, unaligned_buffer, this->size());
    }
    _bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
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
        xrt::bo(_bo, end_bytes - start_bytes, start_bytes), end - start,
        this->_type, this->is_aligned, offset_unaligned_buffer));
  }

private:
  bool is_aligned = true;
  bool own_unaligned{};
  xrt::bo _bo;
  dtype *aligned_buffer;
  dtype *unaligned_buffer;

  dtype *get_aligned_buffer(dtype *host_buffer, size_t length) {
    // Check if the existing pointer already is aligned
    if ((reinterpret_cast<uintptr_t>(host_buffer) % ACCL_FPGA_ALIGNMENT) != 0) {
      std::cerr
          << "[ACCL] Warning: you're creating a buffer from an unaligned host "
             "pointer. This will cause extra copying when syncing the buffers."
          << std::endl;
      /* C++11 requires that we request a size that is a multiple of the
         alignment, or std::aligned_alloc will return a nullptr. So we change
         the size to the minimal required size that meets this requirement.
         (https://en.cppreference.com/w/c/memory/aligned_alloc#Notes) */

      is_aligned = false;
      aligned_buffer =
          static_cast<dtype *>(allocate_aligned_buffer(length * sizeof(dtype)));
      unaligned_buffer = host_buffer;
      own_unaligned = true;
      return aligned_buffer;
    }

    is_aligned = true;
    return host_buffer;
  }

  // Set the buffer after initialization since bo needs to be initialized first,
  // but base constructor is called beforehand.
  void set_buffer() { this->update_buffer(_bo.map<dtype *>(), _bo.address()); }
};
} // namespace ACCL
