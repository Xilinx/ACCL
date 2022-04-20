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
#include "constants.hpp"
#include <memory>

/** @file buffer.hpp */

namespace ACCL {
/**
 * Abstract base class of device buffer that doesn't specify datatype of host
 * buffer.
 *
 */
class BaseBuffer {
public:
  /**
   * Construct a new Base Buffer object.
   *
   * @param byte_array        Pointer to the host buffer.
   * @param size              Size of the host and device buffer in bytes.
   * @param type              Datatype of the device buffer.
   * @param physical_address  The location of the device buffer.
   */
  BaseBuffer(void *byte_array, size_t size, dataType type,
             addr_t physical_address)
      : _byte_array(byte_array), _size(size), _type(type),
        _physical_address(physical_address) {}


  virtual ~BaseBuffer() {}

  /**
   * Sync the host buffer to the device buffer.
   *
   */
  virtual void sync_from_device() = 0;

  /**
   * Sync the device buffer to the host buffer.
   *
   */
  virtual void sync_to_device() = 0;

  /**
   * Free the device buffer.
   *
   */
  virtual void free_buffer() = 0;

  /**
   * Get the size of the buffer in bytes.
   *
   * @return size_t  The size of the buffer.
   */
  size_t size() const { return _size; }

  /**
   * Get the datatype of the device buffer.
   *
   * @return dataType  The datatype of the device buffer.
   */
  dataType type() const { return _type; }

  void *byte_array() const { return _byte_array; }

  /**
   * Get the location of the device buffer.
   *
   * @return addr_t  The location of the device buffer.
   */
  addr_t physical_address() const { return _physical_address; }

  /**
   * Get a slice of the buffer from start to end.
   *
   * @param start           Start of the slice.
   * @param end             End of the slice.
   * @return BaseBuffer     Slice of the buffer from start to end.
   */
  virtual std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) = 0;

protected:
  void *_byte_array;
  const size_t _size;
  const dataType _type;
  addr_t _physical_address;
};

/**
 * Abstract buffer class that specifies the datatype of the host buffer.
 *
 * @tparam dtype  The datatype of the host buffer.
 */
template <typename dtype> class Buffer : public BaseBuffer {
public:
  /**
   * Construct a new Buffer object.
   *
   * @param buffer            The host buffer.
   * @param length            The length of the host buffer.
   * @param type              The datatype of the device buffer.
   * @param physical_address  The location of the device buffer.
   */
  Buffer(dtype *buffer, size_t length, dataType type, addr_t physical_address)
      : BaseBuffer(static_cast<void *>(buffer), length * sizeof(dtype), type,
                   physical_address),
        _buffer(buffer), _length(length){};

  virtual ~Buffer() {}

  /**
   * Get the length of the host buffer.
   *
   * @return size_t  The length of the host buffer.
   */
  size_t length() const { return _length; }

  dtype *buffer() const { return _buffer; }

  dtype operator[](size_t i) { return this->_buffer[i]; }

  dtype &operator[](size_t i) const { return this->_buffer[i]; }

protected:
  dtype *_buffer;
  const size_t _length;

  void update_buffer(dtype *buffer, addr_t physical_address) {
    _buffer = buffer;
    _byte_array = static_cast<void *>(buffer);
    _physical_address = physical_address;
  }
};
} // namespace ACCL
