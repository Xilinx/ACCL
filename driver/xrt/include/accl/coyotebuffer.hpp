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
#include "cProcess.hpp"
#include "cDefs.hpp"
#include "coyotedevice.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>

/** @file coyotebuffer.hpp */

namespace ACCL {
  /**
   * A coyote buffer that is allocated and mapped to FPGA TLB
   *
   * The host pointer will be aligned to 2M bytes. 
   *
   * @tparam dtype Datatype of the buffer.
   */
template <typename dtype> class CoyoteBuffer : public Buffer<dtype> {
  public:

    /**
     * Construct a new CoyoteBuffer object without an existing host pointer.
     *
     * This constructor will allocate a buffer on both the host and the FPGA.
     *
     * @param length  Amount of elements to allocate the buffers for.
     * @param type    ACCL datatype of the buffer.
     * @param device  Device to allocate the FPGA buffer on.
     */
    CoyoteBuffer(addr_t length, dataType type, CCLO *device)
        : Buffer<dtype>(nullptr, length, type, 0x0)
    {
      CoyoteDevice *dev = dynamic_cast<CoyoteDevice *>(device);
      this->device = dev;
      // 2M pages
      size_t page_size = 1ULL << 21;
      size_t buffer_size = length * sizeof(dtype);
      uint32_t n_pages = (buffer_size + page_size - 1) / page_size;
      std::cerr << "CoyoteBuffer contructor called! page_size:"<<page_size<<", buffer_size:"<<buffer_size<<",n_pages:"<<n_pages<< std::endl;

      this->aligned_buffer = (dtype *)this->device->coyote_proc.getMem({fpga::CoyoteAlloc::HUGE_2M, n_pages});

      this->update_buffer(this->aligned_buffer, (addr_t)this->aligned_buffer); 

      std::cerr << "Allocation successful! Allocated buffer: "<<std::setbase(16)<<(uint64_t)this->aligned_buffer << std::setbase(10) <<", Size: " << this->_size << std::endl;

      host_flag = true;
    }

    /**
     * Destroy the CoyoteBuffer object
     *
     */
    virtual ~CoyoteBuffer()
    {

      this->device->coyote_proc.freeMem(this->aligned_buffer);
    }

    /**
     * Check if the buffer is simulated, always false.
     *
     */
    bool is_simulated() const override
    {
      return false;
    }

    /**
    * Check if buffer is host-only.
    *
    * @return true   The buffer is host-only.
    * @return false  The buffer is not host-only.
    */
    bool is_host_only() const override
    {
      std::cerr << "check is_host_only: " << std::setbase(16) << (uint64_t)this->aligned_buffer << ", host_flag: " << std::setbase(10) << this->host_flag << std::endl;
      return this->host_flag;
    }

    /**
     * Sync the data from the device back to the host. 
     *
     */
    void sync_from_device() override
    {
      std::cerr << "calling sync: " << std::setbase(16) << (uint64_t)this->aligned_buffer << ", size: " << std::setbase(10) << this->size() << std::endl;

      this->device->coyote_proc.invoke({fpga::CoyoteOper::SYNC, this->aligned_buffer, (uint32_t)this->_size, false, false, 0, true});
    
      this->host_flag = true;
    }

    /**
     * Sync the data from the host to the device. 
     *
     */
    void sync_to_device() override
    {
      std::cerr << "calling offload: " << std::setbase(16) << (uint64_t)this->aligned_buffer << ", size: " << std::setbase(10) << this->size() << std::endl;

      this->device->coyote_proc.invoke({fpga::CoyoteOper::OFFLOAD, this->aligned_buffer, (uint32_t)this->_size, false, false, 0, false});
    
      this->host_flag = false;
    }


    void free_buffer() override
    {
      this->device->coyote_proc.freeMem(this->aligned_buffer);
      return;
    }

    std::unique_ptr<BaseBuffer> slice(size_t start, size_t end) override
    {
      size_t start_bytes = start * sizeof(dtype);
      size_t end_bytes = end * sizeof(dtype);

      dtype *offset_unaligned_buffer = nullptr;

      debug("CoyoteBuffer::slice not yet implemented!!!");
      // TODO: implement
      return std::unique_ptr<BaseBuffer>(nullptr);
    }


  private:
    dtype *aligned_buffer;
    CoyoteDevice *device;
    bool host_flag; 
  };
} // namespace ACCL
