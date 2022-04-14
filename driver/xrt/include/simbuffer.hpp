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
#include <cmath>
#include "zmq_client.h"

/** @file simbuffer.hpp */

namespace ACCL {
extern addr_t next_free_address;

template <typename dtype> class SimBuffer : public Buffer<dtype> {
private:
    zmq_intf_context *zmq_ctx;

    addr_t get_next_free_address(size_t size) {
        addr_t address = next_free_address;
        // allocate on 4K boundaries
        // not sure how realistic this is, but it does help
        // work around some addressing limitations in RTLsim
        next_free_address += ((addr_t)std::ceil(size / 4096.0)) * 4096;

        return address;
    }

public:
    SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq_intf_context *const context, const addr_t physical_address)
        : Buffer<dtype>(buffer, length, type, physical_address), zmq_ctx(context) {}

    SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq_intf_context *const context)
        : SimBuffer(buffer, length, type, context,
                    this->get_next_free_address(length * sizeof(dtype))) {}
    void sync_from_device() override {
        zmq_client_memread( this->zmq_ctx, 
                            (uint64_t)this->_physical_address, 
                            (unsigned int)this->_size, 
                            static_cast<uint8_t *>(this->_byte_array));
    }

    void sync_to_device() override {
        zmq_client_memwrite( this->zmq_ctx, 
                            (uint64_t)this->_physical_address, 
                            (unsigned int)this->_size, 
                            static_cast<uint8_t *>(this->_byte_array));
    }

    void free_buffer() override { return; }

    std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override {
        return std::unique_ptr<BaseBuffer>(
            new SimBuffer(&this->buffer[start], end - start, this->_type,
                        this->zmq_ctx, this->_physical_address + start));
    }
};
} // namespace ACCL
