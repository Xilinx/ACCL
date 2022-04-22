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

#define DEFAULT_SIMBUFFER_MEMGRP 0

namespace ACCL {
extern addr_t next_free_address;

template <typename dtype> class SimBuffer : public Buffer<dtype> {
private:
  zmq_intf_context *zmq_ctx;
  bool own_buffer{};        // Initialize to false
  xrt::bo _bo;              // Only set if constructed using bo.
  xrt::bo internal_copy_bo; // Used to sync bo over zmq
  xrt::device _device{};    // Used to create copy buffers
  bool bo_valid{};

  addr_t get_next_free_address(size_t size) {
    addr_t address = next_free_address;
    // allocate on 4K boundaries
    // not sure how realistic this is, but it does help
    // work around some addressing limitations in RTLsim
    next_free_address += ((addr_t)std::ceil(size / 4096.0)) * 4096;

    return address;
  }

  dtype *create_internal_buffer(size_t length) {
    own_buffer = true;
    return new dtype[length];
  }

public:
  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq_intf_context *const context, const addr_t physical_address,
            xrt::bo &bo, xrt::device &device, bool bo_valid_)
      : Buffer<dtype>(buffer, length, type, physical_address), zmq_ctx(context),
        _bo(bo), _device(device), bo_valid(bo_valid_) {
    if (bo_valid) {
      internal_copy_bo = xrt::bo(_device, this->_size,
                                 (xrt::memory_group)DEFAULT_SIMBUFFER_MEMGRP);
    }
  }

  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq_intf_context *const context, const addr_t physical_address)
      : Buffer<dtype>(buffer, length, type, physical_address), zmq_ctx(context),
        _bo(xrt::bo()) {}

  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq_intf_context *const context)
      : SimBuffer(buffer, length, type, context,
                  this->get_next_free_address(length * sizeof(dtype))) {}

  SimBuffer(xrt::bo &bo, xrt::device &device, size_t length, dataType type,
            zmq_intf_context *const context)
      : SimBuffer(bo.map<dtype *>(), length, type, context,
                  this->get_next_free_address(length * sizeof(dtype)), bo,
                  device, true) {}

  SimBuffer(size_t length, dataType type, zmq_intf_context *const context)
      : SimBuffer(create_internal_buffer(length), length, type, context) {}

  virtual ~SimBuffer() {
    if (own_buffer) {
      delete[] this->_buffer;
    }
  }

  xrt::bo *bo() override {
    if (bo_valid) {
      return &_bo;
    }

    return nullptr;
  }

  bool is_simulated() const override { return true; }

  void sync_bo_to_device() override {
    if (bo_valid) {
      internal_copy_bo.copy(_bo, this->_size);
      internal_copy_bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
      zmq_client_memwrite(this->zmq_ctx, (uint64_t)this->_physical_address,
                          (unsigned int)this->_size,
                          internal_copy_bo.map<uint8_t *>());
    }
  }

  void sync_bo_from_device() override {
    if (bo_valid) {
      zmq_client_memread(this->zmq_ctx, (uint64_t)this->_physical_address,
                         (unsigned int)this->_size,
                         internal_copy_bo.map<uint8_t *>());
      internal_copy_bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
      _bo.copy(internal_copy_bo, this->_size);
    }
  }

  void sync_from_device() override {
    sync_bo_to_device();
    zmq_client_memread(this->zmq_ctx, (uint64_t)this->_physical_address,
                       (unsigned int)this->_size,
                       static_cast<uint8_t *>(this->_byte_array));
  }

  void sync_to_device() override {
    zmq_client_memwrite(this->zmq_ctx, (uint64_t)this->_physical_address,
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
        this->_physical_address + start, bo_slice, _device, bo_valid));
  }
};
} // namespace ACCL
