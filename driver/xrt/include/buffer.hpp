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
      : _byte_array(byte_array), _type(type),
        _physical_address(physical_address), _size(size) {}

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
  void *const _byte_array;
  const size_t _size;
  const dataType _type;
  const addr_t _physical_address;
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
      : BaseBuffer((void *)buffer, length * sizeof(dtype), type,
                   physical_address),
        buffer(buffer), _length(length){};

  /**
   * Get the length of the host buffer.
   *
   * @return size_t  The length of the host buffer.
   */
  size_t length() const { return _length; }

  dtype operator[](size_t i) { return this->buffer[i]; }

  dtype &operator[](size_t i) const { return this->buffer[i]; }

protected:
  dtype *const buffer;
  const size_t _length;
};
} // namespace ACCL
