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
#include "constants.hpp"
#include <memory>

namespace ACCL {
class DummyBuffer : public BaseBuffer {
public:
  DummyBuffer(addr_t physical_address = 0x0)
      : BaseBuffer(nullptr, 0, dataType::none, physical_address) {}

  virtual ~DummyBuffer() {}

  void sync_from_device() override {}
  void sync_to_device() override {}
  void free_buffer() override {}

  std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override {
    return std::unique_ptr<BaseBuffer>(new DummyBuffer());
  }
};

DummyBuffer dummy_buffer = DummyBuffer();
} // namespace ACCL
