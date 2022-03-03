#pragma once
#include "buffer.hpp"
#include "constants.hpp"
#include <memory>

namespace ACCL {
class DummyBuffer : public BaseBuffer {
public:
  DummyBuffer(addr_t physical_address = 0x0)
      : BaseBuffer(nullptr, 0, dataType::none, physical_address) {}

  void sync_from_device() override {}
  void sync_to_device() override {}
  void free_buffer() override {}

  std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override {
    return std::unique_ptr<BaseBuffer>(new DummyBuffer());
  }
};

DummyBuffer dummy_buffer = DummyBuffer();
} // namespace ACCL
