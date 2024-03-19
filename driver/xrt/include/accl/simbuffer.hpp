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
#include "zmq_client.h"
#include <cmath>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

/** @file simbuffer.hpp */

#define ACCL_SIM_DEFAULT_BANK 0

#ifndef ACCL_SIM_NUM_BANKS
#define ACCL_SIM_NUM_BANKS 1
#endif

#ifndef ACCL_SIM_MEM_SIZE_KB
#define ACCL_SIM_MEM_SIZE_KB (256*1024)
#endif

namespace ACCL {
/** Stores the next free address on the simulated device.
    Multiple card memory banks and one host memory bank */
extern addr_t next_free_card_address[ACCL_SIM_NUM_BANKS];
extern addr_t next_free_host_address;

/**
 * A buffer that is allocated on a external CCLO emulator or simulator with an
 * accompanying host pointer.
 *
 * Allocated memory on the external CCLO is currently not reused in simulation
 * mode, so when allocating a lot of buffers on the CCLO the host might run out
 * of memory.
 *
 * It is possible to pass a simulated BO buffer from the Vitis emulator,
 * in this case a new internal BO buffer will also be allocated for copying data
 * between the simulated BO buffer and the simulated ACCL buffer.
 *
 * @tparam dtype Datatype of the buffer.
 */
template <typename dtype> class SimBuffer : public Buffer<dtype> {
private:
  zmq_intf_context *zmq_ctx;
  bool own_buffer{};        // Initialize to false
  xrt::bo _bo;              // Only set if constructed using bo.
  xrt::bo internal_copy_bo; // Used to sync bo over zmq
  xrt::device _device{};    // Used to create copy buffers
  xrt::bo::flags flags;     // Flags identifying buffer type
  xrt::memory_group memgrp; //bank of buffer has device-side image
  bool bo_valid{};

  /**
   * Get the next free address available on the CCLO.
   *
   * @param size    Size of the buffer to allocate.
   * @return addr_t Next free address on the CCLO.
   */
  addr_t get_next_free_card_address(size_t size, xrt::memory_group memgrp = ACCL_SIM_DEFAULT_BANK) {
    if(memgrp > ACCL_SIM_NUM_BANKS){
      throw std::invalid_argument("Requested address in invalid memory bank");
    }
    addr_t address = next_free_card_address[memgrp];
    // allocate on 4K boundaries
    // not sure how realistic this is, but it does help
    // work around some addressing limitations in RTLsim
    next_free_card_address[memgrp] += ((addr_t)std::ceil(size / 4096.0)) * 4096;

    return address + memgrp*ACCL_SIM_MEM_SIZE_KB*1024;
  }

  /**
   * Get the next free host address available.
   *
   * @param size    Size of the buffer to allocate.
   * @return addr_t Next free host address.
   */
  addr_t get_next_free_host_address(size_t size) {
    addr_t address = next_free_host_address;
    // allocate on 4K boundaries
    // not sure how realistic this is, but it does help
    // work around some addressing limitations in RTLsim
    next_free_host_address += ((addr_t)std::ceil(size / 4096.0)) * 4096;

    return address;
  }

  /**
   * Create a new host buffer.
   *
   * @param length  Amount of elements in the buffer.
   * @return dtype* The new host buffer.
   */
  dtype *create_internal_buffer(size_t length) {
    own_buffer = true;
    return new dtype[length];
  }

  /**
   * Allocate the buffer on the simulated CCLO.
   *
   */
  void allocate_buffer() {
    zmq_client_memalloc(this->zmq_ctx, (uint64_t)this->_address,
                        (unsigned int)this->_size);
  }

public:
  /**
   * Construct a new simulated buffer from an existing host buffer.
   *
   * @param buffer  Host buffer to use.
   * @param length  Amount of elements in the existing host buffer.
   * @param type    ACCL datatype of buffer.
   * @param context The zmq server of the CCLO to use.
   * @param flags   The type of buffer to create.
   * @param mmegrp  The bank in which to allocate the buffer.
   */
  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq_intf_context *const context,
            xrt::bo::flags flags = xrt::bo::flags::normal,
            xrt::memory_group memgrp = ACCL_SIM_DEFAULT_BANK)
      : SimBuffer(buffer, length, type, context,
                  flags == xrt::bo::flags::host_only ?
                  this->get_next_free_host_address(length * sizeof(dtype)) :
                  this->get_next_free_card_address(length * sizeof(dtype), memgrp),
                  flags, memgrp) {}

  /**
   * Construct a new simulated buffer from a simulated BO buffer.
   *
   * This will create a new internal BO buffer as well to copy data between the
   * simulated BO buffer and the simulated ACCL buffer.
   *
   * @param bo      Existing BO buffer to use.
   * @param length  Amount of elements in the existing BO buffer.
   * @param type    ACCL datatype of buffer.
   * @param context The zmq server of the CCLO to use.
   */
  SimBuffer(xrt::bo &bo, xrt::device &device, size_t length, dataType type,
            zmq_intf_context *const context)
      : SimBuffer(bo.map<dtype *>(), length, type, context,
                  bo.get_flags() == xrt::bo::flags::host_only ?
                  this->get_next_free_host_address(length * sizeof(dtype)) :
                  this->get_next_free_card_address(length * sizeof(dtype), ACCL_SIM_DEFAULT_BANK),
                  bo, device, true) {}

  /**
   * Construct a new simulated buffer without an existing host pointer.
   *
   * @param length  Amount of elements to allocate for.
   * @param type    ACCL datatype of buffer.
   * @param context The zmq server of the CCLO to use.
   * @param flags   The type of buffer to create.
   * @param mmegrp  The bank in which to allocate the buffer.
   */
  SimBuffer(size_t length, dataType type, zmq_intf_context *const context,
            xrt::bo::flags flags = xrt::bo::flags::normal,
            xrt::memory_group memgrp = ACCL_SIM_DEFAULT_BANK)
      : SimBuffer(create_internal_buffer(length), length, type, context, flags, memgrp) {}

  /**
   * Construct a new simulated buffer from an existing host pointer at a
   * specific physical address. You should generally let ACCL itself decide
   * which physical address to use.
   *
   * @param buffer  Host buffer to use.
   * @param length  Amount of elements in host pointer.
   * @param type    ACCL datatype of buffer.
   * @param context The zmq server of the CCLO to use.
   * @param address The physical address of the device buffer.
   * @param flags   The type of buffer to create.
   * @param mmegrp  The bank in which to allocate the buffer.
   */
  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq_intf_context *const context, const addr_t address,
            xrt::bo::flags flags = xrt::bo::flags::normal,
            xrt::memory_group memgrp = ACCL_SIM_DEFAULT_BANK)
      : Buffer<dtype>(buffer, length, type, address), zmq_ctx(context),
        _bo(xrt::bo()), flags(flags), memgrp(memgrp) {
    allocate_buffer();
  }

  /**
   * Copy construct of a simulated buffer for internal use only.
   *
   */
  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq_intf_context *const context, const addr_t address,
            xrt::bo &bo, xrt::device &device, bool bo_valid_,
            bool is_slice = false)
      : Buffer<dtype>(buffer, length, type, address), zmq_ctx(context),
        _bo(bo), _device(device), bo_valid(bo_valid_) {
    if (bo_valid) {
      internal_copy_bo = xrt::bo(_device, this->_size, this->flags, this->memgrp);
    }

    allocate_buffer();
  }

  /**
   * Destroy the simulated buffer. Will not free anything on the CCLO.
   *
   */
  virtual ~SimBuffer() {
    if (own_buffer) {
      delete[] this->_buffer;
    }
  }

  /**
   * Return the underlying BO buffer. Will only return a BO buffer if the
   * simulated buffer was constructed from an existing BO buffer. Otherwise it
   * will return a nullptr.
   *
   * @return xrt::bo* The underlying BO buffer if it exists, otherwise a
   * nullptr.
   */
  xrt::bo *bo() {
    if (bo_valid) {
      return &_bo;
    }

    return nullptr;
  }

  /**
   * Check if the buffer is simulated, always true.
   *
   */
  bool is_simulated() const override { return true; }

  /**
   * Check if the buffer is host-only, always false in sim.
   *
   */
  bool is_host_only() const override { return false; }

  /**
   * Sync the user BO buffer to the simulated buffer.
   *
   */
  void sync_bo_to_device() override {
    if (bo_valid) {
      // Use the internal copy BO buffer to sync to the device, since we don't
      // want to overwrite the host pointer of the user BO buffer.
      internal_copy_bo.copy(_bo, this->_size);
      internal_copy_bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
      zmq_client_memwrite(this->zmq_ctx, (uint64_t)this->_address,
                          (unsigned int)this->_size,
                          internal_copy_bo.map<uint8_t *>());
    }
  }

  /**
   * Sync the user BO buffer from the simulated buffer.
   *
   */
  void sync_bo_from_device() override {
    if (bo_valid) {
      zmq_client_memread(this->zmq_ctx, (uint64_t)this->_address,
                         (unsigned int)this->_size,
                         internal_copy_bo.map<uint8_t *>());
      // Use the internal copy BO buffer to sync to the device, since we don't
      // want to overwrite the host pointer of the user BO buffer.
      internal_copy_bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
      _bo.copy(internal_copy_bo, this->_size);
    }
  }

  /**
   * Sync the data from the device back to the host. Will copy the data from
   * the BO buffer to the simulated buffer first if a BO buffer was provided
   * during construction of the simulated buffer.
   *
   */
  void sync_from_device() override {
    // First sync the BO buffer to the simulated buffer to make sure it's up to
    // date.
    sync_bo_to_device();
    zmq_client_memread(this->zmq_ctx, (uint64_t)this->_address,
                       (unsigned int)this->_size,
                       static_cast<uint8_t *>(this->_byte_array));
  }

  /**
   * Sync the data from the host to the device. Will copy the data from
   * the host buffer to the BO buffer as well if a BO buffer was provided
   * during construction of the simulated buffer.
   *
   */
  void sync_to_device() override {
    zmq_client_memwrite(this->zmq_ctx, (uint64_t)this->_address,
                        (unsigned int)this->_size,
                        static_cast<uint8_t *>(this->_byte_array));
    if (bo_valid) {
      _bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  }

  void free_buffer() override { return; }

  std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override {
    xrt::bo bo_slice;

    if (bo_valid) {
      size_t start_bytes = start * sizeof(dtype);
      size_t end_bytes = end * sizeof(dtype);
      bo_slice = xrt::bo(_bo, end_bytes - start_bytes, start_bytes);
    } else {
      bo_slice = _bo;
    }

    return std::unique_ptr<BaseBuffer>(new SimBuffer(
        &this->_buffer[start], end - start, this->_type, this->zmq_ctx,
        this->_address + start, bo_slice, _device, bo_valid, true));
  }
};
} // namespace ACCL
