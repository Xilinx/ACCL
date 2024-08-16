/*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
#  Modifications Copyright (c) 2024, Advanced Micro Devices, Inc.
#  All rights reserved.
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
struct get_region_info_params {
    hsa_region_t * region;
    size_t desired_allocation_size;
    hsa_agent_t* agent;
    bool* taken;

};

struct GpuInfo {
    hsa_agent_t gpu_device; // L'attributo per il dispositivo GPU
    int requestedGPU; // 
    get_region_info_params* information;
};

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

bool my_ready = false;
int counter_gpu = 0;

static void print_info_region(hsa_region_t* region){


    hsa_region_segment_t segment;
    hsa_region_get_info(*region, HSA_REGION_INFO_SEGMENT, &segment); 
    uint32_t flags;
    hsa_region_get_info(*region, HSA_REGION_INFO_GLOBAL_FLAGS,&flags); 
    uint32_t info_size;
    hsa_region_get_info(*region, HSA_REGION_INFO_SIZE,&info_size ); 
    size_t max_size = 0;
    hsa_region_get_info(*region, HSA_REGION_INFO_ALLOC_MAX_SIZE , &max_size);
    uint32_t max_pvt_wg;
    hsa_region_get_info(*region, HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE  , &max_pvt_wg); 
    bool check;
    hsa_region_get_info(*region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, &check);
    size_t runtime_granule;
    hsa_region_get_info(*region, HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE, &runtime_granule); 
    size_t runtime_alignment;
    hsa_region_get_info(*region, HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT , &runtime_alignment);

    std::cout<<"HSA_REGION_INFO_SEGMENT: "<<segment<<std::endl;
    std::cout<<"HSA_REGION_INFO_GLOBAL_FLAGS Flags: "<<flags<<std::endl;
    std::cout<<"HSA_REGION_INFO_SIZE: "<<info_size<<std::endl;
    std::cout<<"HSA_REGION_INFO_ALLOC_MAX_SIZE : "<<max_size<<std::endl;
    std::cout<<"HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE : "<<max_pvt_wg<<std::endl;
    std::cout<<"HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED : "<<check<<std::endl;
    std::cout<<"HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE: "<<runtime_granule<<std::endl;
    std::cout<<"HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT: "<<runtime_alignment<<std::endl;
}

/**
 * 
 * @brief Callback for HSA routine. It determines if a memory region can be used for a given memory allocation size.
 * 
 */
static hsa_status_t get_region_info(hsa_region_t region, void* data) {
    struct get_region_info_params * params = (struct get_region_info_params *) data;
    
    size_t max_size = 0;
    int value = hsa_region_get_info(region, HSA_REGION_INFO_ALLOC_MAX_SIZE , &max_size); 
    uint32_t info_size;
    value = hsa_region_get_info(region, HSA_REGION_INFO_SIZE,&info_size );
    char name[64];
    int stat = hsa_agent_get_info(*params->agent, HSA_AGENT_INFO_NAME, name);
    if(!*params->taken && max_size > params->desired_allocation_size && info_size > params->desired_allocation_size )
      {
          //std::cout << "Belonging to the agent: " << name << std::endl;
          //print_info_region(&region);

    // if(max_size < params->desired_allocation_size) {
    //     return HSA_STATUS_ERROR;
    // }

    //TODO: check on memory size. Currently HSA_REGION_INFO_ALLOC_MAX_SIZE > HSA_REGION_INFO_SIZE for both GPUs
        *params->region = region;
        *params->taken = true;
        //std::cout<<"Returning a region"<<std::endl;

      }
  
    return HSA_STATUS_SUCCESS;
}

  /**
   * @brief Callback for HSA routine. It determines if the given agent is of type HSA_DEVICE_TYPE_GPU
   * and sets the value of data to the agent handle if it is.
   */
static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
  GpuInfo* info = reinterpret_cast<GpuInfo*>(data);
  if (data == NULL) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
    hsa_device_type_t device_type;
    char name[64];
    hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

    uint32_t NumaID; 
    stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &NumaID);

    hsa_agent_get_info(agent,HSA_AGENT_INFO_NAME,name);
    //std::cout<<"printing an agent"<<std::endl;
    //std::cout<< name << std::endl<< "with NUMA ID: "<< NumaID <<  std::endl;
    if (stat != HSA_STATUS_SUCCESS) {
        return stat;
    }

    //if (device_type == HSA_DEVICE_TYPE_GPU && my_ready == false && NumaID == info->NumaID) {
    //if (my_ready == false && NumaID == info->NumaID) {
    if(device_type == HSA_DEVICE_TYPE_GPU && my_ready == false && counter_gpu == info->requestedGPU){  
        std::cout<<"assigning the device with NumaID: "<<NumaID<<std::endl;
        *((hsa_agent_t *)data) = agent;
        my_ready = true;
        char name[64] = { 0 };
        stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
        //debug("GPU found: " + std::string(name));
        //std::cout<<"Inside FindGPU I am looking for the region"<<std::endl;
        *info->information->agent = agent;
        int code = hsa_agent_iterate_regions(agent, get_region_info, info->information); 
        if ( code != HSA_STATUS_SUCCESS)
        {
            std::cout << "Error here" << std::endl;
            throw std::runtime_error("Something happened iterating regions!");
        }
        get_region_info_params infos = *info->information;
        //std::cout<<"founded region: "<< infos.region<<std::endl;
        int err = (infos.region->handle == 0) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
        //std::cout<<"Value of handle: "<<infos.region<<std::endl;
        if(err != HSA_STATUS_SUCCESS) {
          throw std::runtime_error("Insufficient memory on the GPU!");
        }
        

    }else{

      if(device_type == HSA_DEVICE_TYPE_GPU)
      {
        counter_gpu++;
      }
    }
    return HSA_STATUS_SUCCESS;

}
static hsa_status_t find_gpu_noId(hsa_agent_t agent, void *data) {
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
        //debug("GPU found: " + std::string(name));
    }
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
      CoyoteBuffer(addr_t length, dataType type, CCLO *device, bool dmabuf = false, int device_id = -1)
        : Buffer<dtype>(nullptr, length, type, 0x0)
      {
      //std::cout<<"I am creating the CoyoteBuffer"<<std::endl;
      //std::cout<<"Device ID: "<<device_id<<std::endl;
      //std::cout<<"DMABuf: "<<dmabuf<<std::endl;

      CoyoteDevice *dev = dynamic_cast<CoyoteDevice *>(device);
      this->device = dev;
      // 2M pages
      size_t page_size = 1ULL << 21;
      this->buffer_size = length * sizeof(dtype);
      this->n_pages = (buffer_size + page_size - 1) / page_size;
        if(dmabuf) {
          this->dmabuf_support = true;
          hsa_status_t err;
          my_ready = false;
          counter_gpu = 0;
          hsa_agent_t gpu_device;
          bool check = false;
          hsa_region_t region_to_use = {0};
          hsa_region_segment_t segment_to_use;
          struct get_region_info_params info_params = {
              .region = &region_to_use,
              .desired_allocation_size = this->buffer_size,
              .agent = &gpu_device,
              .taken = &check

          };
          //std::cout<<"Printing before the choice"<<std::endl;
          //print_info_region(info_params.region);

          //select GPU device
          if(device_id != -1){
              GpuInfo g;
              g.requestedGPU = device_id; 
              g.information = &info_params;
              err = hsa_iterate_agents(find_gpu, &g);
              gpu_device = g.gpu_device; // this should guarantee code compatibility after changes for GPU selection
              if(err != HSA_STATUS_SUCCESS) {
                throw std::runtime_error("No GPU found! You have specified a NumaID but the GPU was not there. Please provide a correct NumaID");
              }
          }else{
              err = hsa_iterate_agents(find_gpu_noId,&gpu_device);
              if(err != HSA_STATUS_SUCCESS) {
                 throw std::runtime_error("No GPU found! You have not specified any NumaID. Please provide a correct NumaID");
              }
                //std::cout<<"Region INFO -to be checked"<<std::endl;
                hsa_agent_iterate_regions(gpu_device, get_region_info, &info_params); 
                //std::cout<<"founded region: "<< info_params.region<<std::endl;
                err = (region_to_use.handle == 0) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
                //std::cout<<"Value of handle: "<<region_to_use.handle<<std::endl;
                if(err != HSA_STATUS_SUCCESS) {
                  throw std::runtime_error("Insufficient memory on the GPU!");
                }
          }

          //std::cout<<"Now I am printing the chosen region"<<std::endl;
          //print_info_region(info_params.region);
          char name[64] = { 0 };
          int stat = hsa_agent_get_info(*info_params.agent, HSA_AGENT_INFO_NAME, name);
          if(stat != HSA_STATUS_SUCCESS) {
            throw std::runtime_error("Name Retrival failed!");
          }
          uint32_t id; 
          stat = hsa_agent_get_info(*info_params.agent, HSA_AGENT_INFO_NODE, &id);
          if(stat != HSA_STATUS_SUCCESS) {
            throw std::runtime_error("ID Retrival failed!");
          }
          //std::cout << "in the chosen region, the agent is: " <<  name << " and its id is: " << id << std::endl;

          //std::cout<<"Printing Memory Allocate Params"<<std::endl;
          //std::cout<<"Buffer Size: " << this->buffer_size<<std::endl;
          //std::cout<<"Aligned Buffer: " << (void **) &(this->aligned_buffer) << std::endl;
          //std::cout<<"=========================================================================================="<<std::endl;
          err = hsa_memory_allocate(*info_params.region, this->buffer_size, (void **) &(this->aligned_buffer)); 
          //std::cout<<"=========================================================================================="<<std::endl;

          if(err != HSA_STATUS_SUCCESS) {
            throw std::runtime_error("Allocation failed on the GPU!");
          }
          if(this->aligned_buffer==NULL)
            std::cout<<" Pointer null"<< std::endl;
          else
            std::cout<<"Pointer not null"<< std::endl;

          this->_buffer = this->aligned_buffer;
          this->dma_buf_fd = 0;
          size_t offset = 0;

          //export DMABuf via HSA

          //std::cout<<"Printing Input Params"<<std::endl;
          //std::cout<<this->aligned_buffer<<std::endl;
          //std::cout<<this->buffer_size<<std::endl;
          //std::cout<<&this->dma_buf_fd<<std::endl;
          //std::cout<<&offset<<std::endl;
          // this function fails before reaching the amd driver
          err = hsa_amd_portable_export_dmabuf(this->aligned_buffer, this->buffer_size, &this->dma_buf_fd, &offset);
          
          if(err != HSA_STATUS_SUCCESS) {
              if(err == HSA_STATUS_ERROR_INVALID_AGENT)
                std::cout << "HSA_STATUS_ERROR_INVALID_AGENT" << std::endl;
              std::cout << "Value of err: " << err << std::endl; 
              hsa_amd_portable_close_dmabuf(this->dma_buf_fd);
              throw std::runtime_error("HSA export failed!");
          }
          //debug("hsa_amd_portable_export_dmabuf done!");
          
          //import DMABuf, attach FPGA, and setup TLB
          auto dev = static_cast<::ACCL::CoyoteDevice *>(device);
          dev->attach_dma_buf(this->dma_buf_fd, (unsigned long) this->aligned_buffer, offset);
          //printf("Attach Completed. Going to Update\n");
        } else {
          //standard memory allocation on the HOST and TLB setup
          this->aligned_buffer = (dtype *)this->device->coyote_proc->getMem({fpga::CoyoteAlloc::HUGE_2M, n_pages});
        }
      this->update_buffer(this->aligned_buffer, (addr_t)this->aligned_buffer); 
      //printf("Attach Completed. Update Completed\n");

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
      printf("Defined HSA Support. Value of dmabuf_support = %d \n",this->dmabuf_support);
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
        debug("buffer free");        

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
