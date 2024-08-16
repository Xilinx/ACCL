/*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
#  Modifications Copyright (c) 2024, Advanced Micro Devices, Inc.
#  All rights reserved.
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
#include <string.h>     // streerror
#include <sys/mman.h>
//#define COYOTE_HSA_SUPPORT 0
#ifdef COYOTE_HSA_SUPPORT
  #include <hip/hip_runtime.h>
  #include <hsa/hsa_ext_amd.h>
  #include <hsa.h>
  #include <hsa/hsa_ext_finalize.h>
  #include <hsakmt/hsakmt.h>
#endif

/** @file coyotebuffer.hpp */

namespace ACCL {


#ifdef COYOTE_HSA_SUPPORT

#define HIPVERIFY(stmt)				\
  do                                          \
      {                                       \
          hipError_t result = (stmt);           \
          if (result != HIP_SUCCESS) {       \
              const char *_err_name;          \
              fprintf(stderr, "HIP error: %s\n",hipGetErrorName(result));   \
          }                                   \
          assert(HIP_SUCCESS == result);     \
      } while (0)

typedef struct {
  int dev;
  hipDeviceProp_t prop;
  int dev_pci_domain_id;
  int dev_pci_bus_id;
  int dev_pci_device_id;
  hipCtx_t dev_ctx;
  hipDeviceptr_t dev_addr;
} gpu_handle;


#ifndef GLOBALS_HPP
#define GLOBALS_HPP
inline bool my_ready = false;
#endif

  /**
   * @brief Callback for HSA routine. It determines if the given agent is of type HSA_DEVICE_TYPE_GPU
   * and sets the value of data to the agent handle if it is.
   */
static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
  if (data == NULL) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
    hsa_device_type_t device_type;
    hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (stat != HSA_STATUS_SUCCESS) {
        return stat;
    }
    if (device_type == HSA_DEVICE_TYPE_GPU && my_ready == false) {
        *((hsa_agent_t *)data) = agent;
        my_ready = true;
        char name[64] = { 0 };
        stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
        debug("GPU found: " + std::string(name));
    }
    return HSA_STATUS_SUCCESS;
}


struct get_region_info_params {
    hsa_region_t * region;
    size_t desired_allocation_size;

};

/**
 * 
 * @brief Callback for HSA routine. It determines if a memory region can be used for a given memory allocation size.
 * 
 */
static hsa_status_t get_region_info(hsa_region_t region, void* data) {
    struct get_region_info_params * params = (struct get_region_info_params *) data;
    size_t max_size = 0;
    hsa_region_get_info(region, HSA_REGION_INFO_ALLOC_MAX_SIZE , &max_size);
    if(max_size < params->desired_allocation_size) {
        return HSA_STATUS_ERROR;
    }
    *params->region = region;
    return HSA_STATUS_SUCCESS;
}

#endif

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
     * This constructor will allocate a buffer on both the FPGA and the HOST or the GPU.
     *
     * @param length  Amount of elements to allocate the buffers for.
     * @param type    ACCL datatype of the buffer.
     * @param device  Device to allocate the FPGA buffer on.
     * @param dmabuf  If set to true, it allocates the buffer on the GPU for peer-to-peer DMA. Otherwise, the buffer is allocated on the host.
     */
    #ifdef COYOTE_HSA_SUPPORT
      CoyoteBuffer(addr_t length, dataType type, CCLO *device, bool dmabuf = false)
        : Buffer<dtype>(nullptr, length, type, 0x0)
      {
      CoyoteDevice *dev = dynamic_cast<CoyoteDevice *>(device);
      this->device = dev;
      // 2M pages
      size_t page_size = 1ULL << 21;
      this->buffer_size = length * sizeof(dtype);
      this->n_pages = (buffer_size + page_size - 1) / page_size;
        if(dmabuf) {
          this->dmabuf_support = true;
          hsa_agent_t gpu_device;
          hsa_status_t err;
          my_ready = false;

          //select GPU device
          err = hsa_iterate_agents(find_gpu, &gpu_device);
          if(err != HSA_STATUS_SUCCESS) {
            throw std::runtime_error("No GPU found!");
          }

          hsa_region_t region_to_use = {0}; 
          struct get_region_info_params info_params = {
              .region = &region_to_use,
              .desired_allocation_size = this->buffer_size
          };
          hsa_agent_iterate_regions(gpu_device, get_region_info, &info_params);
          err = (region_to_use.handle == 0) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
          if(err != HSA_STATUS_SUCCESS) {
            throw std::runtime_error("Insufficient memory on the GPU!");
          }

          err = hsa_memory_allocate(region_to_use, this->buffer_size, (void **) &(this->aligned_buffer));
          if(err != HSA_STATUS_SUCCESS) {
            throw std::runtime_error("Allocation failed on the GPU!");
          }
          this->_buffer = this->aligned_buffer;
          
          this->dma_buf_fd = 0;
          size_t offset = 0;

          //export DMABuf via HSA
          err = hsa_amd_portable_export_dmabuf(this->aligned_buffer, this->buffer_size, &this->dma_buf_fd, &offset);

          if(err != HSA_STATUS_SUCCESS) {
              hsa_amd_portable_close_dmabuf(this->dma_buf_fd);
              throw std::runtime_error("HSA export failed!");
          }
          debug("hsa_amd_portable_export_dmabuf done!");
          
          //import DMABuf, attach FPGA, and setup TLB
          auto dev = static_cast<::ACCL::CoyoteDevice *>(device);
          dev->attach_dma_buf(this->dma_buf_fd, (unsigned long) this->aligned_buffer, offset);
        } else {
          //standard memory allocation on the HOST and TLB setup
          this->aligned_buffer = (dtype *)this->device->coyote_proc->getMem({fpga::CoyoteAlloc::HUGE_2M, n_pages});
        }
      this->update_buffer(this->aligned_buffer, (addr_t)this->aligned_buffer); 

      //std::cerr << "Allocation successful! Allocated buffer: "<<std::setbase(16)<<(uint64_t)this->aligned_buffer << std::setbase(10) <<", Size: " << this->_size << std::endl;

      host_flag = true;
      

      // if Coyote device has multiple qProc, map the allocated buffer with all qProc
      // if(this->device->num_qp > 0 && this->device->coyote_qProc_vec.size()!=0){
      //   for (unsigned int i=0; i<this->device->coyote_qProc_vec.size(); i++)
      //   {
      //     std::cerr << "Mapping coyote buffer to qProc cPid:"<<this->device->coyote_qProc_vec[i]->getCpid()<<", buffer_size:"<<buffer_size<<std::endl;
      //     this->device->coyote_qProc_vec[i]->userMap(this->aligned_buffer, buffer_size);
      //   }
      // }
      }
    #else      
      CoyoteBuffer(addr_t length, dataType type, CCLO *device)
        : Buffer<dtype>(nullptr, length, type, 0x0)
    {
      CoyoteDevice *dev = dynamic_cast<CoyoteDevice *>(device);
      this->device = dev;
      // 2M pages
      size_t page_size = 1ULL << 21;
      this->buffer_size = length * sizeof(dtype);
      this->n_pages = (buffer_size + page_size - 1) / page_size;
      std::cerr << "CoyoteBuffer contructor called! page_size:"<<page_size<<", buffer_size:"<<buffer_size<<",n_pages:"<<n_pages<< std::endl;

      this->aligned_buffer = (dtype *)this->device->coyote_proc->getMem({fpga::CoyoteAlloc::HUGE_2M, n_pages});

      this->update_buffer(this->aligned_buffer, (addr_t)this->aligned_buffer); 

      std::cerr << "Allocation successful! Allocated buffer: "<<std::setbase(16)<<(uint64_t)this->aligned_buffer << std::setbase(10) <<", Size: " << this->_size << std::endl;

      host_flag = true;
      

      // if Coyote device has multiple qProc, map the allocated buffer with all qProc
      // if(this->device->num_qp > 0 && this->device->coyote_qProc_vec.size()!=0){
      //   for (unsigned int i=0; i<this->device->coyote_qProc_vec.size(); i++)
      //   {
      //     std::cerr << "Mapping coyote buffer to qProc cPid:"<<this->device->coyote_qProc_vec[i]->getCpid()<<", buffer_size:"<<buffer_size<<std::endl;
      //     this->device->coyote_qProc_vec[i]->userMap(this->aligned_buffer, buffer_size);
      //   }
      // }
      }
    #endif
    /**
     * Destroy the CoyoteBuffer object
     *
     */
    ~CoyoteBuffer()
    {
      if(this->aligned_buffer) {
        free_buffer();
      }

    }

    /**
     * Check if the buffer is simulated, always false.
     *
     */
    bool is_simulated() const override
    {
      return false;
    }

    void * buffer() const 
    {
      return this->aligned_buffer;
    }

    /**
    * Check if buffer is host-only.
    *
    * @return true   The buffer is host-only.
    * @return false  The buffer is not host-only.
    */
    bool is_host_only() const override
    {
      // std::cerr << "check is_host_only: " << std::setbase(16) << (uint64_t)this->aligned_buffer << ", host_flag: " << std::setbase(10) << this->host_flag << std::endl;
      return this->host_flag;
    }

    /**
     * Sync the data from the device back to the host. 
     *
     */
    void sync_from_device() override
    {
      std::cerr << "calling sync: " << std::setbase(16) << (uint64_t)this->aligned_buffer << ", size: " << std::setbase(10) << this->size() << std::endl;

      this->device->coyote_proc->invoke({fpga::CoyoteOper::SYNC, this->aligned_buffer, (uint32_t)this->_size, true, true, 0, false});
    
      this->host_flag = true;
    }

    /**
     * Sync the data from the host to the device. 
     *
     */
    void sync_to_device() override
    {
      std::cerr << "calling offload: " << std::setbase(16) << (uint64_t)this->aligned_buffer << ", size: " << std::setbase(10) << this->size() << std::endl;

      this->device->coyote_proc->invoke({fpga::CoyoteOper::OFFLOAD, this->aligned_buffer, (uint32_t)this->_size, true, true, 0, false});
    
      this->host_flag = false;
    }


    void free_buffer() override
    {
      // if Coyote device has multiple qProc, unmap the allocated buffer with all qProc
      // if(this->device->num_qp > 0 && this->device->coyote_qProc_vec.size()!=0){
      //   for (unsigned int i=0; i<this->device->coyote_qProc_vec.size(); i++)
      //   {
      //     std::cerr << "Unmapping user buffer from qProc cPid:"<< std::setbase(10)<<this->device->coyote_qProc_vec[i]->getCpid()<<", buffer_size:"<<buffer_size<<","<<std::setbase(16) << (uint64_t)this->aligned_buffer<<std::endl;
      //     this->device->coyote_qProc_vec[i]->userUnmap(this->aligned_buffer);
      //   }
      // }

      //std::cerr << "Free user buffer from cProc cPid:"<< std::setbase(10)<<this->device->coyote_proc->getCpid()<<", buffer_size:"<<buffer_size<<","<<std::setbase(16) << (uint64_t)this->aligned_buffer<<std::endl;
      #ifdef COYOTE_HSA_SUPPORT
      if(this->dmabuf_support) {
        //detach FPGA from DMABuf
        this->device->coyote_proc->detachDMABuf(this->dma_buf_fd);
        //close DMABuf exporter
        hsa_status_t err = hsa_amd_portable_close_dmabuf(this->dma_buf_fd);
        char * err_str;
		    hsa_status_string(err, (const char **) &err_str);
        debug("hsa_amd_portable_close_dmabuf exit status = " + std::string(err_str));
        //deallocate buffer
        hsa_memory_free(this->aligned_buffer);
        this->aligned_buffer = nullptr;
      } else {
        this->device->coyote_proc->freeMem(this->aligned_buffer);
        this->aligned_buffer = nullptr;
      }
      #else
        this->device->coyote_proc->freeMem(this->aligned_buffer);
        this->aligned_buffer = nullptr;
      #endif
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

    size_t size() const { return this->buffer_size; }

    size_t page() const { return this->n_pages; }


  private:
    dtype *aligned_buffer;
    CoyoteDevice *device;
    bool host_flag; 
    size_t buffer_size;
    uint32_t n_pages;
    int dma_buf_fd;
    bool dmabuf_support = false;
  };
} // namespace ACCL
