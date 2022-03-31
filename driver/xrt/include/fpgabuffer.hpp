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
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

namespace ACCL {
template <typename dtype> class FPGABuffer : public Buffer<dtype> {

public:
  FPGABuffer(dtype *buffer, size_t length, dataType type, xrt::bo bo_)
      : bo(bo), Buffer<dtype>(buffer, length, type, bo.address()) {}
  FPGABuffer(dtype *buffer, size_t length, dataType type, xrt::device device,
             xrt::memory_group mem_grp)
      : bo(device, buffer, length * sizeof(dtype), mem_grp), Buffer<dtype>(
                                                                 buffer, length,
                                                                 type,
                                                                 bo.address()) {
  }

  void sync_from_device() override {
    bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  void sync_to_device() override {
    bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void free_buffer() override { return; }

  std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override {
    return std::unique_ptr<BaseBuffer>(
        new FPGABuffer(&this->buffer[start], end - start, this->_type,
                       xrt::bo(bo, end - start, start)));
  }

private:
  xrt::bo bo;
};
} // namespace ACCL

#endif
