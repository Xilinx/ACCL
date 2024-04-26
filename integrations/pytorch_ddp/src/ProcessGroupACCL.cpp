/*****************************************************************************
  Copyright (C) 2023 Advanced Micro Devices, Inc

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*****************************************************************************/

#include "ProcessGroupACCL.hpp"

#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef ACCL_PROCESS_GROUP_HIP_ENABLED
#include "hip/hip_runtime.h"
#endif

#ifdef ACCL_PROCESS_GROUP_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>

#include <accl.hpp>
#include <accl_network_utils.hpp>
#include "coyote_init.hpp"

namespace cyt = coyote_init;
namespace py = pybind11;

namespace c10d {

#define CEIL_DIV(x, y) ((x) / (y) + ((x) % (y) != 0))

#define ACCL_ERROR(status)                                                     \
  ("ACCL error in: " + std::string(__FILE__) + ":" +                           \
   std::to_string(__LINE__) + ", with error code: " + std::to_string(status))

#if defined(ACCL_PROCESS_GROUP_HIP_ENABLED) &&                                 \
    defined(ACCL_PROCESS_GROUP_CUDA_ENABLED)
#error Cannot compile Process Group with both HIP and CUDA support
#endif // ACCL_PROCESS_GROUP_HIP_ENABLED && ACCL_PROCESS_GROUP_CUDA_ENABLED

namespace {

/* Alternative for std::format from C++20 in C++17.
   Source: https://stackoverflow.com/a/26221725 */
template <typename... Args>
std::string string_format(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

template <typename val_t>
std::string format_array(val_t *data, std::size_t size, std::size_t breakval = 3) {
  std::ostringstream buffer;
  buffer << "[";
  if (size <= breakval * 2 + 1) {
    for (std::size_t i = 0; i < size; ++i) {
      buffer << data[i];
      if (i + 1 != size) {
        buffer << ", ";
      }
    }
  } else {
    for (std::size_t i = 0; i < breakval; ++i) {
      buffer << data[i] << ", ";
    }
    buffer << "..., ";
    for (std::size_t i = size - breakval; i < size; ++i) {
      buffer << data[i];
      if (i + 1 != size) {
        buffer << ", ";
      }
    }
  }
  buffer << "]";

  return buffer.str();
}

// Op mapping
std::map<ReduceOp, ACCL::reduceFunction> acclOp = {
    {ReduceOp::SUM, ACCL::reduceFunction::SUM},
};
// Type mapping
std::map<at::ScalarType, ACCL::dataType> acclDatatype = {
    {at::kByte, ACCL::dataType::int32},
    {at::kChar, ACCL::dataType::int32},
    {at::kDouble, ACCL::dataType::float64},
    {at::kFloat, ACCL::dataType::float32},
    {at::kInt, ACCL::dataType::int32},
    {at::kLong, ACCL::dataType::int64},
    {at::kShort, ACCL::dataType::int32},
};

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor &tensor) {
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }
}

void checkSingleTensor(const std::vector<at::Tensor> &tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(false,
                "ACCL process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(const at::Tensor &t_in,
                          const std::vector<at::Tensor> &tensors) {
  for (const auto &tensor : tensors) {
    if ((tensor.numel() != t_in.numel()) ||
        (tensor.scalar_type() != t_in.scalar_type())) {
      TORCH_CHECK(false, "Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensor);
  }
}

// We have to use strings for the python bindings since it is not possible to
// convert between torch.dtype objects and c10::ScalarType values.
ACCL::dataType convert_datatype_from_torch(std::string torch_type) {
  if (torch_type == "torch.float16" || torch_type == "torch.half") {
    return ACCL::dataType::float16;
  } else if (torch_type == "torch.float32" || torch_type == "torch.float") {
    return ACCL::dataType::float32;
  } else if (torch_type == "torch.float64" || torch_type == "torch.double") {
    return ACCL::dataType::float64;
  } else if (torch_type == "torch.int32" || torch_type == "torch.int") {
    return ACCL::dataType::int32;
  } else if (torch_type == "torch.int64" || torch_type == "torch.long") {
    return ACCL::dataType::int64;
  } else {
    return ACCL::dataType::none;
  }
}

ACCL::dataType convert_datatype_from_torch(c10::ScalarType torch_type) {
  switch (torch_type) {
  case at::kHalf:
    return ACCL::dataType::float16;
  case at::kFloat:
    return ACCL::dataType::float32;
  case at::kDouble:
    return ACCL::dataType::float64;
  case at::kInt:
    return ACCL::dataType::int32;
  case at::kLong:
    return ACCL::dataType::int64;
  default:
    return ACCL::dataType::none;
  }
}

const char *convert_datatype_to_torch(ACCL::dataType torch_type) {
  switch (torch_type) {
  case ACCL::dataType::float16:
    return "torch.float16";
  case ACCL::dataType::float32:
    return "torch.float32";
  case ACCL::dataType::float64:
    return "torch.float64";
  case ACCL::dataType::int32:
    return "torch.int32";
  case ACCL::dataType::int64:
    return "torch.int64";
  default:
    return "unknown";
  }
}

std::map<ACCL::dataType, ACCL::dataType> convert_compression_from_dict(
    const std::map<std::string, std::string> &dictionary) {
  std::map<ACCL::dataType, ACCL::dataType> map;
  for (const auto &item : dictionary) {
    auto uncompressed = convert_datatype_from_torch(item.first);
    auto compressed = convert_datatype_from_torch(item.second);
    map.emplace(uncompressed, compressed);
  }

  return map;
}

std::map<std::string, std::string> convert_compression_to_dict(
    const std::map<ACCL::dataType, ACCL::dataType> &map) {
  std::map<std::string, std::string> dictionary;
  for (const auto &item : map) {
    auto uncompressed = convert_datatype_to_torch(item.first);
    auto compressed = convert_datatype_to_torch(item.second);
    dictionary.emplace(uncompressed, compressed);
  }

  return dictionary;
}

// Create an ACCL Buffer with correct type
std::unique_ptr<ACCL::BaseBuffer> create_buffer(ACCL::ACCL &accl, size_t length,
                                                c10::ScalarType type) {
  switch (type) {
  case at::kInt:
    return accl.create_buffer<int32_t>(length, acclDatatype.at(type));
  case at::kLong:
    return accl.create_buffer<int64_t>(length, acclDatatype.at(type));
  case at::kFloat:
    return accl.create_buffer<float>(length, acclDatatype.at(type));
  case at::kDouble:
    return accl.create_buffer<double>(length, acclDatatype.at(type));
  default:
    TORCH_CHECK(false, "Tensor has unsupported datatype");
    break;
  }
}

// Create an ACCL Buffer with correct type
std::unique_ptr<ACCL::BaseBuffer> create_coyotebuffer(ACCL::ACCL &accl, size_t length,
                                                c10::ScalarType type) {
  switch (type) {
  case at::kInt:
    return accl.create_coyotebuffer<int32_t>(length, acclDatatype.at(type));
  case at::kLong:
    return accl.create_coyotebuffer<int64_t>(length, acclDatatype.at(type));
  case at::kFloat:
    return accl.create_coyotebuffer<float>(length, acclDatatype.at(type));
  case at::kDouble:
    return accl.create_coyotebuffer<double>(length, acclDatatype.at(type));
  default:
    TORCH_CHECK(false, "Tensor has unsupported datatype");
    break;
  }
}

// Create an ACCL Buffer with correct type
std::unique_ptr<ACCL::BaseBuffer> wrap_buffer(ACCL::ACCL &accl, xrt::bo &bo,
                                              size_t length,
                                              c10::ScalarType type) {
  size_t size;
  if (type == at::kInt || type == at::kFloat) {
    size = length * 4;
  } else {
    size = length * 8;
  }
  xrt::bo slice = xrt::bo(bo, size, static_cast<size_t>(0));
  switch (type) {
  case at::kInt:
    return accl.create_buffer<int32_t>(slice, length, acclDatatype.at(type));
  case at::kLong:
    return accl.create_buffer<int64_t>(slice, length, acclDatatype.at(type));
  case at::kFloat:
    return accl.create_buffer<float>(slice, length, acclDatatype.at(type));
  case at::kDouble:
    return accl.create_buffer<double>(slice, length, acclDatatype.at(type));
  default:
    TORCH_CHECK(false, "Tensor has unsupported datatype");
    break;
  }
}

// Create an ACCL P2P Buffer with correct type
std::unique_ptr<ACCL::BaseBuffer>
create_buffer_p2p(ACCL::ACCL &accl, size_t length, c10::ScalarType type) {
  switch (type) {
  case at::kInt:
    return accl.create_buffer_p2p<int32_t>(length, acclDatatype.at(type));
  case at::kLong:
    return accl.create_buffer_p2p<int64_t>(length, acclDatatype.at(type));
  case at::kFloat:
    return accl.create_buffer_p2p<float>(length, acclDatatype.at(type));
  case at::kDouble:
    return accl.create_buffer_p2p<double>(length, acclDatatype.at(type));
  default:
    TORCH_CHECK(false, "Tensor has unsupported datatype");
    break;
  }
}

std::unique_ptr<ACCL::BaseBuffer> create_buffer_p2p(ACCL::ACCL &accl,
                                                    const at::Tensor &tensor) {
  return create_buffer_p2p(accl, tensor.numel(), tensor.scalar_type());
}

// Create an ACCL Buffer with correct type from Tensor
std::unique_ptr<ACCL::BaseBuffer> create_buffer(ACCL::ACCL &accl,
                                                const at::Tensor &tensor) {
  std::unique_ptr<ACCL::BaseBuffer> buffer;
  switch (tensor.scalar_type()) {
  case at::kInt:
    buffer = accl.create_buffer(static_cast<int32_t *>(tensor.data_ptr()),
                                tensor.numel(),
                                acclDatatype.at(tensor.scalar_type()));

    ACCL::debug("Creating int32 buffer at 0x" +
                ACCL::debug_hex(buffer->address()) + " of " +
                std::to_string(buffer->size()) + "B.");
    break;
  case at::kLong:
    buffer = accl.create_buffer(static_cast<int64_t *>(tensor.data_ptr()),
                                tensor.numel(),
                                acclDatatype.at(tensor.scalar_type()));

    ACCL::debug("Creating int64 buffer at 0x" +
                ACCL::debug_hex(buffer->address()) + " of " +
                std::to_string(buffer->size()) + "B.");
    break;
  case at::kFloat:
    buffer = accl.create_buffer(static_cast<float *>(tensor.data_ptr()),
                                tensor.numel(),
                                acclDatatype.at(tensor.scalar_type()));

    ACCL::debug("Creating float32 buffer at 0x" +
                ACCL::debug_hex(buffer->address()) + " of " +
                std::to_string(buffer->size()) + "B.");

    break;
  case at::kDouble:
    buffer = accl.create_buffer(static_cast<double *>(tensor.data_ptr()),
                                tensor.numel(),
                                acclDatatype.at(tensor.scalar_type()));

    ACCL::debug("Creating float64 buffer at 0x" +
                ACCL::debug_hex(buffer->address()) + " of " +
                std::to_string(buffer->size()) + "B.");
    break;
  default:
    TORCH_CHECK(false, "Tensor has unsupported datatype");
    break;
  }

  return buffer;
}

// Check if process is compiled with HIP support
inline bool hip_enabled() {
#ifdef ACCL_PROCESS_GROUP_HIP_ENABLED
  return true;
#else
  return false;
#endif
}

// Check if process is compiled with CUDA support
inline bool cuda_enabled() {
#ifdef ACCL_PROCESS_GROUP_CUDA_ENABLED
  return true;
#else
  return false;
#endif
}

// Check if tensor is a GPU tensor, the ProcessGroup is compiled with GPU
// support, ACCL is not running in simulation mode, and the ProcessGroup was
// initialized with p2p_enabled
bool p2p_applicable(ACCL::ACCL &accl, const at::Tensor &tensor,
                    bool p2p_enabled) {
  auto type = tensor.device().type();
  if (type != c10::DeviceType::CPU && p2p_enabled && !accl.is_simulated()) {
    if (type == c10::DeviceType::HIP) {
      return hip_enabled();
    } else if (type == c10::DeviceType::CUDA) {
      // HIP tensors will identify themselves as CUDA tensor depending on the
      // initialization, so we have to see CUDA tensors as HIP tensors if
      // ProcessGroup is compiled with HIP support
#ifdef ACCL_PROCESS_GROUP_HIP_ENABLED
      return hip_enabled();
#else
      return cuda_enabled();
#endif
    }
  }
  return false;
}

// Copy a GPU tensor to a P2P FPGA buffer
void copy_to_p2p_buffer(ACCL::BaseBuffer &buffer, const at::Tensor &tensor) {
  if (tensor.device().type() == c10::DeviceType::HIP) {
    ACCL::debug("Syncing HIP GPU buffer to FPGA");
#ifdef ACCL_PROCESS_GROUP_HIP_ENABLED
    hipMemcpy(buffer.byte_array(), tensor.data_ptr(), tensor.nbytes(),
              hipMemcpyDeviceToHost);
#else
    TORCH_CHECK(false, "ACCL ProcessGroup is build without HIP support");
#endif
  } else if (tensor.device().type() == c10::DeviceType::CUDA) {
#ifdef ACCL_PROCESS_GROUP_HIP_ENABLED
    ACCL::debug("Syncing HIP GPU buffer to FPGA");
    hipMemcpy(buffer.byte_array(), tensor.data_ptr(), tensor.nbytes(),
              hipMemcpyDeviceToHost);
#else
    ACCL::debug("Syncing CUDA GPU buffer to FPGA");
#ifdef ACCL_PROCESS_GROUP_CUDA_ENABLED
    cudaMemcpy(buffer.byte_array(), tensor.data_ptr(), tensor.nbytes(),
               cudaMemcpyDeviceToHost);
#else
    TORCH_CHECK(false, "ACCL ProcessGroup is build without CUDA support");
#endif // ACCL_PROCESS_GROUP_CUDA_ENABLED
#endif // ACCL_PROCESS_GROUP_HIP_ENABLED
  }
}

// Create a new FPGA P2P buffer and copy contents of GPU tensor
inline std::unique_ptr<ACCL::BaseBuffer>
create_and_copy_p2p_buffer(ACCL::ACCL &accl, const at::Tensor &tensor) {
  ACCL::debug("Creating p2p buffer of size " + std::to_string(tensor.nbytes()));
  std::unique_ptr<ACCL::BaseBuffer> buffer =
      create_buffer_p2p(accl, tensor.numel(), tensor.scalar_type());
  copy_to_p2p_buffer(*buffer, tensor);
  return buffer;
}

// Copy results from an FPGA P2P buffer back to the GPU tensor
void copy_back_p2p_buffer(ACCL::BaseBuffer &buffer, const at::Tensor &tensor) {
  if (tensor.device().type() == c10::DeviceType::HIP) {
    ACCL::debug("Syncing HIP GPU buffer from FPGA");
#ifdef ACCL_PROCESS_GROUP_HIP_ENABLED
    hipMemcpy(tensor.data_ptr(), buffer.byte_array(), tensor.nbytes(),
              hipMemcpyHostToDevice);
#else
    TORCH_CHECK(false, "ACCL ProcessGroup is build without HIP support");
#endif
  } else if (tensor.device().type() == c10::DeviceType::CUDA) {
#ifdef ACCL_PROCESS_GROUP_HIP_ENABLED
    ACCL::debug("Syncing HIP GPU buffer from FPGA");
    hipMemcpy(tensor.data_ptr(), buffer.byte_array(), tensor.nbytes(),
              hipMemcpyHostToDevice);
#else
    ACCL::debug("Syncing CUDA GPU buffer from FPGA");
#ifdef ACCL_PROCESS_GROUP_CUDA_ENABLED
    cudaMemcpy(tensor.data_ptr(), buffer.byte_array(), tensor.nbytes(),
               cudaMemcpyHostToDevice);
#else
    TORCH_CHECK(false, "ACCL ProcessGroup is build without CUDA support");
#endif // ACCL_PROCESS_GROUP_CUDA_ENABLED
#endif // ACCL_PROCESS_GROUP_HIP_ENABLED
  }
}

bool check_arp(vnx::Networklayer &network_layer,
               std::vector<ACCL::rank_t> &ranks, int rank, int size) {
  std::map<unsigned, bool> ranks_checked;
  for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
    ranks_checked[i] = false;
  }

  bool sanity_check = true;
  const std::map<int, std::pair<std::string, std::string>> arp =
      network_layer.read_arp_table(size);

  std::ostringstream ss_arp;
  ss_arp << "ARP table:";

  for (const std::pair<const int, std::pair<std::string, std::string>> &elem :
       arp) {
    const unsigned index = elem.first;
    const std::pair<std::string, std::string> &entry = elem.second;
    const std::string &mac = entry.first;
    const std::string &ip = entry.second;
    ss_arp << "\n(" << index << ") " << mac << ": " << ip;

    for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
      if (ranks[i].ip == ip) {
        if (ranks_checked[i]) {
          std::cerr << "Double entry for " << ip << " in arp table!"
                    << std::endl;
          sanity_check = false;
        } else {
          ranks_checked[i] = true;
        }
      }
    }
  }

  ACCL::debug(ss_arp.str());

  if (!sanity_check) {
    return false;
  }

  unsigned hosts = 0;
  for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
    if (ranks_checked[i]) {
      hosts += 1;
    }
  }
  if (hosts < static_cast<unsigned>(size) - 1) {
    std::cerr << "Found only " << hosts << " hosts out of " << size - 1 << "!"
              << std::endl;
    return false;
  }

  return true;
}

} // namespace

ACCL::dataType ProcessGroupACCL::get_compressed_type(c10::ScalarType datatype) {
  ACCL::dataType accl_datatype = convert_datatype_from_torch(datatype);

  if (compression.count(accl_datatype) > 0) {
    return compression[accl_datatype];
  } else {
    return ACCL::dataType::none;
  }
}

std::vector<at::Tensor> ProcessGroupACCL::WorkACCL::result() {
  return outputTensors_;
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupACCL::WorkACCL::getFuture() {
  return future_;
}

void ProcessGroupACCL::WorkACCL::finishWorkACCLError(std::exception_ptr eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupACCL::WorkACCL::finishWorkACCL() {
  future_->markCompleted(at::IValue(outputTensors_));
  finish();
}

// Static global states
std::mutex ProcessGroupACCL::pgGlobalMutex_;
// We only want to initialize once
std::once_flag ProcessGroupACCL::onceFlagInitACCL;

void ProcessGroupACCL::acclExit() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
}

void ProcessGroupACCL::initACCLOnce() {
  // Initialize ACCL environment
  std::call_once(onceFlagInitACCL, []() {
    if (std::atexit(ProcessGroupACCL::acclExit)) {
      TORCH_CHECK(false, "Fail to register the ACCL exit handler");
    }
  });
}

// Convert list of Python rank pointers to a vector of ranks
std::vector<ACCL::rank_t> convert_ranks(
    const std::vector<c10::intrusive_ptr<ProcessGroupACCL::Rank>> &ranks) {
  std::vector<ACCL::rank_t> accl_ranks = {};

  for (const auto &rank : ranks) {
    ACCL::rank_t accl_rank = {rank->ip, rank->port, rank->session_id,
                              rank->max_segment_size};
    accl_ranks.emplace_back(accl_rank);
  }

  return accl_ranks;
}

// Initialize ACCL
ProcessGroupACCL::ProcessGroupACCL(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    std::vector<c10::intrusive_ptr<ProcessGroupACCL::Rank>> &ranks,
    bool simulator, bool p2p_enabled,
    const std::map<ACCL::dataType, ACCL::dataType> &compression,
    const std::vector<int> &profiling_ranks, double profiling_timeout,
    const std::string &xclbin, accl_network_utils::acclDesign design,
    int device_index, int nbufs, uint64_t bufsize, bool rsfec)
    : ProcessGroup(rank, size), store_(store), stop_(false),
      device_index_(device_index), nbufs_(nbufs), bufsize_(bufsize),
      rsfec_(rsfec), simulator_(simulator), xclbin_(xclbin),
      bufsize(bufsize), p2p_enabled(p2p_enabled),
      coyote_enabled(
        design == accl_network_utils::acclDesign::CYT_RDMA
        || design == accl_network_utils::acclDesign::CYT_TCP),
      compression(compression), initialized(false) {

  if (std::find(profiling_ranks.begin(), profiling_ranks.end(), rank) !=
      profiling_ranks.end()) {
    std::this_thread::sleep_for(
        std::chrono::duration<double>(profiling_timeout));
  }

  ranks_ = convert_ranks(ranks);
  design_ = design;
  if (!simulator){
    if (coyote_enabled) {
      if (design_ == accl_network_utils::acclDesign::CYT_TCP) {
        cyt_device = new ACCL::CoyoteDevice();
      } else if (design_ == accl_network_utils::acclDesign::CYT_RDMA) {
        cyt_device = new ACCL::CoyoteDevice(size_);
        cyt::setup_cyt_rdma(ibvQpConn_vec, ranks_, rank_, *cyt_device);
      } else {
        throw std::runtime_error("Undefined ACCL design");
      }
    }
    // use xrt
    else{
      xrt_device = xrt::device(device_index);
    }
  }
}

std::vector<std::uint8_t> ProcessGroupACCL::get_local_qp(unsigned int rank) {
  std::vector<std::uint8_t> qp;
  char *data = (char *) &ibvQpConn_vec[rank]->getQpairStruct()->local;
  for (std::size_t i = 0; i < sizeof(fpga::ibvQ); ++i) {
    qp.push_back(data[i]);
  }

  return qp;
}

void ProcessGroupACCL::set_remote_qp(unsigned int rank, std::vector<std::uint8_t> &qp) {
  fpga::ibvQ remote_qp;
  char *data = (char *) &remote_qp;
  for (std::size_t i = 0; i < sizeof(fpga::ibvQ); ++i) {
    data[i] = qp[i];
  }

  ibvQpConn_vec[rank]->getQpairStruct()->remote = remote_qp;
}

void ProcessGroupACCL::initialize() {
  std::cout << "PG initialize called\n";
  if (initialized) {
    throw std::runtime_error("Already initialized process group");
  }

  if (coyote_enabled && !simulator_) {
    if (design_ == accl_network_utils::acclDesign::CYT_RDMA) {
      cyt::configure_cyt_rdma(ibvQpConn_vec, ranks_, rank_);
    } else {
      throw std::runtime_error("Coyote configure not implemented");
    }

    accl = std::make_unique<ACCL::ACCL>(cyt_device);

    // eager protocol for now
    int protoc = 0;
    // default from test.cpp
    int segsize = 4096 * 1024;
    
    if (protoc == 0){
      std::cout<<"Eager Protocol"<<std::endl;
      accl.get()->initialize(ranks_, rank_,
			     size_+2, bufsize, segsize, 4096*1024*2);
    } else {
      std::cout<<"Rendezvous Protocol"<<std::endl;
      accl.get()->initialize(ranks_, rank_, size_, 64, 64, segsize);
    }  
    
    ACCL::debug(std::string("[ACCL coyote] communicator: ") + accl->dump_communicator());
  } else {
    accl = accl_network_utils::initialize_accl(ranks_, rank_,
                                               simulator_, design_, xrt_device,
                                               xclbin_, nbufs_, bufsize, 0,
                                               rsfec_);
    accl->set_timeout(1e6);
    accl->set_rendezvous_threshold(16*1024);
                                      
    int devicemem = accl->devicemem();
    if (!simulator_) {
      // Initialize cache buffers
      buf0 = xrt::bo(xrt_device, bufsize, devicemem);
      buf1 = xrt::bo(xrt_device, bufsize, devicemem);
    }
  }

  accl->set_timeout(1e8);
  // Start the worker thread accepting ACCL calls
  workerThread_ = std::thread(&ProcessGroupACCL::runLoop, this);
  initialized = true;
}

ProcessGroupACCL::~ProcessGroupACCL() { destroy(); }

void ProcessGroupACCL::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();
  queueProduceCV_.notify_all();

  // Join the single worker thread
  workerThread_.join();
}

void ProcessGroupACCL::abort() {
  destroy();
  accl->deinit();
  exit(EXIT_FAILURE);
}

// Worker thread
void ProcessGroupACCL::runLoop() {
  std::unique_lock<std::mutex> lock(pgMutex_);

  while (!stop_) { // Set by destroy
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    auto workTuple = std::move(queue_.front());

    queue_.pop_front();

    auto &workEntry = std::get<0>(workTuple);
    auto &work = std::get<1>(workTuple);

    lock.unlock(); // Item popped from queue, unlock lock
    queueConsumeCV_.notify_one();

    try {
      workEntry->run(workEntry);
      work->finishWorkACCL();
    } catch (...) {
      work->finishWorkACCLError(std::current_exception());
    }

    lock.lock(); // Lock before popping next item
  }
}

c10::intrusive_ptr<Work> ProcessGroupACCL::enqueue(
    std::unique_ptr<WorkEntry> entry, const char *profilingTitle, OpType optype,
    const c10::optional<std::vector<at::Tensor>> &inputTensors) {
  if (!initialized) {
    throw std::runtime_error("Process group not yet initialized, "
                             "call accl_process_group.initialize()");
  }
  auto work = c10::make_intrusive<WorkACCL>(entry->dst, profilingTitle, optype,
                                            inputTensors);
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.push_back(std::make_tuple(std::move(entry), work));
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

void ProcessGroupACCL::run_broadcast(at::Tensor tensor_original,
                                     const BroadcastOptions &opts) {
  at::Tensor *tensor = &tensor_original;
  at::Tensor empty_tensor;
  std::unique_ptr<ACCL::BaseBuffer> data;

  // Reserve device
  c10::DeviceGuard guard(tensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Copy data from GPU to FPGA if necessary
  if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
    data = create_and_copy_p2p_buffer(*accl, tensor_original);
  } else {
    if (coyote_enabled) {
      // Copy tensor to CPU tensor first
      data = create_coyotebuffer(*accl, tensor->numel(), tensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(tensor_original.numel()));
      empty_tensor = torch::from_blob(
          data->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU));
      tensor = &empty_tensor;
      if (rank_ == opts.rootRank) {
        tensor->copy_(tensor_original);
      }
    } else if (tensor_original.device().type() != c10::DeviceType::CPU) {
      // Copy tensor to CPU tensor first
      data = create_buffer(*accl, tensor->numel(), tensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(tensor_original.numel()));
      empty_tensor = torch::from_blob(
          data->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU));
      tensor = &empty_tensor;
      if (rank_ == opts.rootRank) {
        tensor->copy_(tensor_original);
      }
    } else {
      data = create_buffer(*accl, *tensor);
    }
  }

  // Run broadcast
  ACCL::debug("Starting broadcast of " + std::to_string(tensor->numel()) +
              " items");

  if (!coyote_enabled && rank_ == opts.rootRank) {
    data->sync_to_device();
  }

  accl->bcast(*data, tensor->numel(), opts.rootRank, ACCL::GLOBAL_COMM, true,
              true, get_compressed_type(tensor->scalar_type()));
  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }

  if (!coyote_enabled && rank_ != opts.rootRank) {
    data->sync_from_device();
  }

  // Copy results back to GPU if necessary
  if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
    copy_back_p2p_buffer(*data, tensor_original);
  } else if (coyote_enabled || tensor_original.device().type() != c10::DeviceType::CPU) {
    ACCL::debug("Copying data back from CPU tensor of size " +
                std::to_string(tensor_original.numel()));
    if (rank_ != opts.rootRank) {
      tensor_original.copy_(*tensor);
    }
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::broadcast(std::vector<at::Tensor> &tensors,
                            const BroadcastOptions &opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        at::Tensor &tensor = (entry->src)[0];
        // Segment data if necessary
        if (tensor.nbytes() > bufsize) {
          size_t n = bufsize / tensor.itemsize();
          for (size_t i = 0; i < tensor.numel(); i += n) {
            size_t end = std::min(i + n, static_cast<size_t>(tensor.numel()));
            run_broadcast(tensor.slice(0, i, end), opts);
          }
        } else {
          run_broadcast(tensor, opts);
        }
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "accl::broadcast", OpType::BROADCAST,
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

void ProcessGroupACCL::run_allreduce(at::Tensor tensor_original,
                                     const AllreduceOptions &opts) {
  at::Tensor *tensor = &tensor_original;
  at::Tensor empty_tensor;
  std::unique_ptr<ACCL::BaseBuffer> data;
  std::unique_ptr<ACCL::BaseBuffer> result;

  // Reserve device
  c10::DeviceGuard guard(tensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Copy data from GPU to FPGA if necessary, and create a new result buffer,
  // since ACCL doesn't support in-place allreduce
  if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
    data = create_and_copy_p2p_buffer(*accl, tensor_original);
    result = create_buffer_p2p(*accl, tensor->numel(), tensor->scalar_type());
  } else {
    if (accl->is_simulated() || coyote_enabled) {
      data = create_buffer(*accl, tensor->numel(), tensor->scalar_type());
    } else {
      data = wrap_buffer(*accl, buf0, tensor->numel(), tensor->scalar_type());
    }
    ACCL::debug("Copying data to aligned CPU tensor of size " +
                std::to_string(tensor_original.numel()));
    empty_tensor = torch::from_blob(
        data->byte_array(), tensor_original.sizes(),
        tensor_original.options().device(c10::DeviceType::CPU));
    tensor = &empty_tensor;
    tensor->copy_(tensor_original);
    ACCL::debug("Creating extra result buffer of size " +
                std::to_string(tensor_original.numel()));
    if (accl->is_simulated() || coyote_enabled) {
      result = create_buffer(*accl, tensor->numel(), tensor->scalar_type());
    } else {
      result = wrap_buffer(*accl, buf1, tensor->numel(), tensor->scalar_type());
    }
  }

  // Run allreduce
  ACCL::debug("Starting allreduce of " + std::to_string(tensor->numel()) +
              " items");
  if (!coyote_enabled) {
    data->sync_to_device();
  }
  accl->allreduce(*data, *result, tensor->numel(), acclOp.at(opts.reduceOp),
                  ACCL::GLOBAL_COMM, true, true,
                  get_compressed_type(tensor->scalar_type()));
  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }

  if (!coyote_enabled) {
    result->sync_from_device();
  }

  // Copy result buffer back to original tensor
  if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
    copy_back_p2p_buffer(*result, tensor_original);
  } else {
    ACCL::debug("Copying result data back to original tensor of size " +
                std::to_string(tensor_original.numel()));
    tensor_original.copy_(torch::from_blob(
        result->byte_array(), tensor_original.sizes(),
        tensor_original.options().device(c10::DeviceType::CPU)));
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::allreduce(std::vector<at::Tensor> &tensors,
                            const AllreduceOptions &opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        auto tensor = (entry->src)[0];
        // Segment data if necessary
        if (tensor.nbytes() > bufsize) {
          size_t n = bufsize / tensor.itemsize();
          for (size_t i = 0; i < tensor.numel(); i += n) {
            size_t end = std::min(i + n, static_cast<size_t>(tensor.numel()));
            run_allreduce(tensor.slice(0, i, end), opts);
          }
        } else {
          run_allreduce(tensor, opts);
        }
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "accl::all_reduce", OpType::ALLREDUCE,
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::allreduce_coalesced(std::vector<at::Tensor> &tensors,
                                      const AllreduceCoalescedOptions &opts) {
  TORCH_CHECK(false,
              "allreduce_coalesced is currently not supported with ACCL");
}

void ProcessGroupACCL::run_reduce(at::Tensor tensor_original,
                                  const ReduceOptions &opts) {
  at::Tensor *tensor = &tensor_original;
  at::Tensor empty_tensor;
  std::unique_ptr<ACCL::BaseBuffer> data;
  std::unique_ptr<ACCL::BaseBuffer> result;

  // Reserve device
  c10::DeviceGuard guard(tensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Copy data from GPU to FPGA if necessary, and create a new result buffer,
  // since ACCL doesn't support in-place reduce
  if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
    data = create_and_copy_p2p_buffer(*accl, tensor_original);

    if (rank_ == opts.rootRank) {
      result = create_buffer_p2p(*accl, tensor->numel(), tensor->scalar_type());
    }
  } else {
    if (coyote_enabled) {
      // Copy tensor to CPU tensor first
      data = create_coyotebuffer(*accl, tensor->numel(), tensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(tensor_original.numel()));
      empty_tensor = torch::from_blob(
          data->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU));
      tensor = &empty_tensor;
      tensor->copy_(tensor_original);
    } else if (tensor_original.device().type() != c10::DeviceType::CPU) {
      data = create_buffer(*accl, tensor->numel(), tensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(tensor_original.numel()));
      empty_tensor = torch::from_blob(
          data->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU));
      tensor = &empty_tensor;
      tensor->copy_(tensor_original);
    } else {
      data = create_buffer(*accl, *tensor);
    }

    if (rank_ == opts.rootRank) {
      ACCL::debug("Creating extra result buffer of size " +
                  std::to_string(tensor_original.numel()));
      if (coyote_enabled) {
        result = create_coyotebuffer(*accl, tensor->numel(), tensor->scalar_type());
      } else {
        result = create_buffer(*accl, tensor->numel(), tensor->scalar_type());
      }
    }
  }

  // Run reduce
  ACCL::debug("Starting reduce of " + std::to_string(tensor->numel()) +
              " items");
  if (!coyote_enabled) {
    data->sync_to_device();
  }
  accl->reduce(*data, *result, tensor->numel(), opts.rootRank,
               acclOp.at(opts.reduceOp), ACCL::GLOBAL_COMM, true, true,
               get_compressed_type(tensor->scalar_type()));
  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }

  if (!coyote_enabled && rank_ == opts.rootRank) {
    result->sync_from_device();
  }

  // Copy result buffer back to original tensor
  if (rank_ == opts.rootRank) {
    if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
      copy_back_p2p_buffer(*result, tensor_original);
    } else {
      ACCL::debug("Copying back results to original tensor of size " +
                  std::to_string(tensor_original.numel()));
      tensor_original.copy_(torch::from_blob(
          result->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU)));
    }
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::reduce(std::vector<at::Tensor> &tensors,
                         const ReduceOptions &opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        auto tensor = (entry->src)[0];
        // Segment data if necessary
        if (tensor.nbytes() > bufsize) {
          size_t n = bufsize / tensor.itemsize();
          for (size_t i = 0; i < tensor.numel(); i += n) {
            size_t end = std::min(i + n, static_cast<size_t>(tensor.numel()));
            run_reduce(tensor.slice(0, i, end), opts);
          }
        } else {
          run_reduce(tensor, opts);
        }
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "accl::reduce", OpType::REDUCE,
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

void ProcessGroupACCL::run_allgather(
    at::Tensor srctensor_original,
    const std::vector<at::Tensor> &dsttensorvec) {
  at::Tensor *srctensor = &srctensor_original;
  at::Tensor empty_srctensor;
  std::unique_ptr<ACCL::BaseBuffer> srcdata;
  at::Tensor dsttensor;
  std::unique_ptr<ACCL::BaseBuffer> dstdata;

  // Reserve device
  c10::DeviceGuard guard(srctensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Copy data from GPU to FPGA if necessary
  if (p2p_applicable(*accl, srctensor_original, p2p_enabled)) {
    srcdata = create_and_copy_p2p_buffer(*accl, srctensor_original);
  } else {
    if (coyote_enabled) {
      // Copy tensor to CPU tensor first
      srcdata = create_coyotebuffer(*accl, srctensor->numel(), srctensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(srctensor_original.numel()));
      empty_srctensor = torch::from_blob(
          srcdata->byte_array(), srctensor_original.sizes(),
          srctensor_original.options().device(c10::DeviceType::CPU));
      srctensor = &empty_srctensor;
      srctensor->copy_(srctensor_original);
    } else if (srctensor_original.device().type() != c10::DeviceType::CPU) {
      srcdata =
          create_buffer(*accl, srctensor->numel(), srctensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(srctensor_original.numel()));
      empty_srctensor = torch::from_blob(
          srcdata->byte_array(), srctensor_original.sizes(),
          srctensor_original.options().device(c10::DeviceType::CPU));
      srctensor = &empty_srctensor;
      srctensor->copy_(srctensor_original);
    } else {
      srcdata = create_buffer(*accl, *srctensor);
    }
  }

  // Create new output tensor, since dsttensorvec is not continuous in memory
  if (p2p_applicable(*accl, dsttensorvec[0], p2p_enabled)) {
    dstdata = create_buffer_p2p(*accl,
                                srctensor->numel() * static_cast<size_t>(size_),
                                srctensor->scalar_type());
  } else if (coyote_enabled) {
    dstdata =
        create_coyotebuffer(*accl, srctensor->numel() * static_cast<size_t>(size_),
                      srctensor->scalar_type());
    std::vector<int64_t> sizes = {static_cast<int64_t>(srctensor->numel()) *
                                  size_};
    dsttensor = torch::from_blob(
        dstdata->byte_array(), sizes,
        srctensor_original.options().device(c10::DeviceType::CPU));
  } else {
    dstdata =
        create_buffer(*accl, srctensor->numel() * static_cast<size_t>(size_),
                      srctensor->scalar_type());
    std::vector<int64_t> sizes = {static_cast<int64_t>(srctensor->numel()) *
                                  size_};
    dsttensor = torch::from_blob(
        dstdata->byte_array(), sizes,
        srctensor_original.options().device(c10::DeviceType::CPU));
  }

  // Run allgather
  ACCL::debug("Starting allgather of " + std::to_string(srctensor->numel()) +
              " items");
  if (!coyote_enabled) {
    srcdata->sync_to_device();
  }
  accl->allgather(*srcdata, *dstdata, srctensor->numel(), ACCL::GLOBAL_COMM,
                  true, true, get_compressed_type(srctensor->scalar_type()));

  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }
  if (!coyote_enabled) {
    dstdata->sync_from_device();
  }

  // Copy results back to dsttensorvec
  for (const auto i : c10::irange(dsttensorvec.size())) {
    if (p2p_applicable(*accl, dsttensorvec[0], p2p_enabled)) {
      auto slice =
          dstdata->slice(i * srctensor->numel(), (i + 1) * srctensor->numel());
      copy_back_p2p_buffer(*slice, dsttensorvec[i]);
    } else {
      dsttensorvec[i].copy_(dsttensor.slice(0, i * srctensor->numel(),
                                            (i + 1) * srctensor->numel()));
    }
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                            std::vector<at::Tensor> &inputTensors,
                            const AllgatherOptions &opts) {
  checkSingleTensor(inputTensors);
  if (outputTensors.size() != 1) {
    TORCH_CHECK(false, "ACCL process group only supports a single "
                       "tensor op");
  }
  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    TORCH_CHECK(false, "All gather: number of output tensors should equal "
                       "to the world size");
  }

  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [this](std::unique_ptr<WorkEntry> &entry) {
        auto srctensor = (entry->src)[0];
        auto &dsttensors = entry->dst;
        // Segment data if necessary
        if (srctensor.nbytes() > bufsize) {
          size_t n = bufsize / srctensor.itemsize();
          for (size_t i = 0; i < srctensor.numel(); i += n) {
            size_t end =
                std::min(i + n, static_cast<size_t>(srctensor.numel()));
            std::vector<at::Tensor> dsttensorslices;
            dsttensorslices.reserve(dsttensors.size());
            for (auto &dsttensor : dsttensors) {
              dsttensorslices.emplace_back(dsttensor.slice(0, i, end));
            }
            run_allgather(srctensor.slice(0, i, end), dsttensorslices);
          }
        } else {
          run_allgather(srctensor, dsttensors);
        }
      };
  auto entry = std::make_unique<WorkEntry>(&inputTensors, &outputTensors[0],
                                           std::move(runFunc));
  return enqueue(std::move(entry), "accl::all_gather", OpType::ALLGATHER,
                 c10::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<Work> ProcessGroupACCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>> & /* unused */,
    std::vector<at::Tensor> & /* unused */,
    const AllgatherOptions & /* unused */) {
  TORCH_CHECK(false, "ProcessGroupACCL does not support allgather_coalesced");
}

void ProcessGroupACCL::run_gather(at::Tensor srctensor_original,
                                  const std::vector<at::Tensor> &dsttensorvec,
                                  const GatherOptions &opts) {
  at::Tensor *srctensor = &srctensor_original;
  at::Tensor empty_srctensor;
  std::unique_ptr<ACCL::BaseBuffer> srcdata;
  at::Tensor dsttensor;
  std::unique_ptr<ACCL::BaseBuffer> dstdata;

  // Reserve device
  c10::DeviceGuard guard(srctensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Copy data from GPU to FPGA if necessary
  if (p2p_applicable(*accl, srctensor_original, p2p_enabled)) {
    srcdata = create_and_copy_p2p_buffer(*accl, srctensor_original);
  } else {
    if (coyote_enabled) {
      srcdata =
          create_coyotebuffer(*accl, srctensor->numel(), srctensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(srctensor_original.numel()));
      empty_srctensor = torch::from_blob(
          srcdata->byte_array(), srctensor_original.sizes(),
          srctensor_original.options().device(c10::DeviceType::CPU));
      srctensor = &empty_srctensor;
      srctensor->copy_(srctensor_original);
    } else if (srctensor_original.device().type() != c10::DeviceType::CPU) {
      srcdata =
          create_buffer(*accl, srctensor->numel(), srctensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(srctensor_original.numel()));
      empty_srctensor = torch::from_blob(
          srcdata->byte_array(), srctensor_original.sizes(),
          srctensor_original.options().device(c10::DeviceType::CPU));
      srctensor = &empty_srctensor;
      srctensor->copy_(srctensor_original);
    } else {
      srcdata = create_buffer(*accl, *srctensor);
    }
  }

  // Create new output tensor, since dsttensorvec is not continuous in memory
  if (rank_ == opts.rootRank) {
    if (p2p_applicable(*accl, dsttensorvec[0], p2p_enabled)) {
      dstdata = create_buffer_p2p(
          *accl, srctensor->numel() * static_cast<size_t>(size_),
          srctensor->scalar_type());
    } else if (coyote_enabled) {
      dstdata =
          create_coyotebuffer(*accl, srctensor->numel() * static_cast<size_t>(size_),
                        srctensor->scalar_type());
      std::vector<int64_t> sizes = {static_cast<int64_t>(srctensor->numel()) *
                                    size_};
      dsttensor =
          torch::from_blob(dstdata->byte_array(), sizes,
                           srctensor->options().device(c10::DeviceType::CPU));
    } else {
      dstdata =
          create_buffer(*accl, srctensor->numel() * static_cast<size_t>(size_),
                        srctensor->scalar_type());
      std::vector<int64_t> sizes = {static_cast<int64_t>(srctensor->numel()) *
                                    size_};
      dsttensor =
          torch::from_blob(dstdata->byte_array(), sizes,
                           srctensor->options().device(c10::DeviceType::CPU));
    }
  }

  // Run gather
  ACCL::debug("Starting gather of " + std::to_string(srctensor->numel()) +
              " items");

  if (!coyote_enabled) {
    srcdata->sync_to_device();
  }

  accl->gather(*srcdata, *dstdata, srctensor->numel(), opts.rootRank,
               ACCL::GLOBAL_COMM, true, true,
               get_compressed_type(srctensor->scalar_type()));

  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }

  if (!coyote_enabled && rank_ == opts.rootRank) {
    dstdata->sync_from_device();
  }

  // Copy results back to dsttensorvec
  if (rank_ == opts.rootRank) {
    for (const auto i : c10::irange(dsttensorvec.size())) {
      if (p2p_applicable(*accl, dsttensorvec[0], p2p_enabled)) {
        auto slice = dstdata->slice(i * srctensor->numel(),
                                    (i + 1) * srctensor->numel());
        copy_back_p2p_buffer(*slice, dsttensorvec[i]);
      } else {
        dsttensorvec[i].copy_(dsttensor.slice(0, i * srctensor->numel(),
                                              (i + 1) * srctensor->numel()));
      }
    }
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::gather(std::vector<std::vector<at::Tensor>> &outputTensors,
                         std::vector<at::Tensor> &inputTensors,
                         const GatherOptions &opts) {
  checkSingleTensor(inputTensors);

  if (rank_ != opts.rootRank) {
    if (outputTensors.size() > 0) {
      TORCH_CHECK(false, "Gather: number of output tensors should be 0 "
                         "for non-root");
    }
  } else {
    if (outputTensors.size() != 1) {
      TORCH_CHECK(false, "Gather: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != outputTensors[0].size()) {
      TORCH_CHECK(false, "Gather: number of output tensors should equal "
                         "to the world size");
    }
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        auto srctensor = (entry->src)[0];
        auto &dsttensors = entry->dst;
        // Segment data if necessary
        if (srctensor.nbytes() > bufsize) {
          size_t n = bufsize / srctensor.itemsize();
          for (size_t i = 0; i < srctensor.numel(); i += n) {
            size_t end =
                std::min(i + n, static_cast<size_t>(srctensor.numel()));
            std::vector<at::Tensor> dsttensorslices;
            dsttensorslices.reserve(dsttensors.size());
            for (auto &dsttensor : dsttensors) {
              dsttensorslices.emplace_back(dsttensor.slice(0, i, end));
            }
            run_gather(srctensor.slice(0, i, end), dsttensorslices, opts);
          }
        } else {
          run_gather(srctensor, dsttensors, opts);
        }
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::make_unique<WorkEntry>(&inputTensors, &outputTensors[0],
                                             std::move(runFunc));
    return enqueue(std::move(entry), "accl::gather", OpType::GATHER,
                   c10::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    auto entry =
        std::make_unique<WorkEntry>(&inputTensors, nullptr, std::move(runFunc));
    return enqueue(std::move(entry), "accl::gather", OpType::GATHER,
                   c10::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

void ProcessGroupACCL::run_scatter(std::vector<at::Tensor> &srctensorvec,
                                   at::Tensor dsttensor_original,
                                   const ScatterOptions &opts) {
  std::unique_ptr<ACCL::BaseBuffer> srcdata;
  at::Tensor *dsttensor = &dsttensor_original;
  at::Tensor empty_dsttensor;
  std::unique_ptr<ACCL::BaseBuffer> dstdata;

  // Reserve device
  c10::DeviceGuard guard(dsttensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Create new input buffer, since srctensorvec is not continuous in memory
  if (rank_ == opts.rootRank) {
    at::Tensor srctensor;
    if (rank_ == opts.rootRank) {
      if (p2p_applicable(*accl, srctensorvec[0], p2p_enabled)) {
        srcdata = create_buffer_p2p(
            *accl, dsttensor->numel() * static_cast<size_t>(size_),
            dsttensor->scalar_type());
      } else if (coyote_enabled) {
        srcdata = create_coyotebuffer(*accl,
                                dsttensor->numel() * static_cast<size_t>(size_),
                                dsttensor->scalar_type());
        std::vector<int64_t> sizes = {static_cast<int64_t>(dsttensor->numel()) *
                                      size_};
        srctensor =
            torch::from_blob(srcdata->byte_array(), sizes,
                             dsttensor->options().device(c10::DeviceType::CPU));
      } else {
        srcdata = create_buffer(*accl,
                                dsttensor->numel() * static_cast<size_t>(size_),
                                dsttensor->scalar_type());
        std::vector<int64_t> sizes = {static_cast<int64_t>(dsttensor->numel()) *
                                      size_};
        srctensor =
            torch::from_blob(srcdata->byte_array(), sizes,
                             dsttensor->options().device(c10::DeviceType::CPU));
      }
    }

    // Copy data to input buffer
    for (const auto i : c10::irange(srctensorvec.size())) {
      if (p2p_applicable(*accl, srctensorvec[0], p2p_enabled)) {
        auto slice = srcdata->slice(i * dsttensor->numel(),
                                    (i + 1) * dsttensor->numel());
        copy_to_p2p_buffer(*slice, srctensorvec[i]);
      } else {
        auto slice = srctensor.slice(0, i * dsttensor->numel(),
                                     (i + 1) * dsttensor->numel());
        slice.copy_(srctensorvec[i]);
      }
    }
  }

  // Create output buffer
  if (p2p_applicable(*accl, dsttensor_original, p2p_enabled)) {
    dstdata = create_and_copy_p2p_buffer(*accl, dsttensor_original);
  } else {
    if (coyote_enabled) {
      dstdata =
          create_coyotebuffer(*accl, dsttensor->numel(), dsttensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(dsttensor_original.numel()));
      empty_dsttensor = torch::from_blob(
          dstdata->byte_array(), dsttensor_original.sizes(),
          dsttensor_original.options().device(c10::DeviceType::CPU));
      dsttensor = &empty_dsttensor;
      dsttensor->copy_(dsttensor_original);
    } else if (dsttensor_original.device().type() != c10::DeviceType::CPU) {
      dstdata =
          create_buffer(*accl, dsttensor->numel(), dsttensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(dsttensor_original.numel()));
      empty_dsttensor = torch::from_blob(
          dstdata->byte_array(), dsttensor_original.sizes(),
          dsttensor_original.options().device(c10::DeviceType::CPU));
      dsttensor = &empty_dsttensor;
      dsttensor->copy_(dsttensor_original);
    } else {
      dstdata = create_buffer(*accl, *dsttensor);
    }
  }

  // Run scatter
  ACCL::debug("Starting scatter of " + std::to_string(dsttensor->numel()) +
              " items");

  if (!coyote_enabled && rank_ == opts.rootRank) {
    srcdata->sync_to_device();
  }

  accl->scatter(*srcdata, *dstdata, dsttensor->numel(), opts.rootRank,
                ACCL::GLOBAL_COMM, true, true,
                get_compressed_type(dsttensor->scalar_type()));
  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }

  if (!coyote_enabled) {
    dstdata->sync_from_device();
  }

  // Copy result back to GPU if necessary
  if (p2p_applicable(*accl, dsttensor_original, p2p_enabled)) {
    copy_back_p2p_buffer(*dstdata, dsttensor_original);
  } else if (coyote_enabled || dsttensor_original.device().type() != c10::DeviceType::CPU) {
    ACCL::debug("Copying data back from CPU tensor of size " +
                std::to_string(dsttensor_original.numel()));
    dsttensor_original.copy_(*dsttensor);
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::scatter(std::vector<at::Tensor> &outputTensors,
                          std::vector<std::vector<at::Tensor>> &inputTensors,
                          const ScatterOptions &opts) {
  checkSingleTensor(outputTensors);

  if (rank_ != opts.rootRank) {
    if (inputTensors.size() > 0) {
      TORCH_CHECK(false, "Scatter: number of input tensors should be 0 "
                         "for non-root");
    }
  } else {
    if (inputTensors.size() != 1) {
      TORCH_CHECK(false, "Scatter: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != inputTensors[0].size()) {
      TORCH_CHECK(false, "Scatter: number of input tensors should equal "
                         "to the world size");
    }
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        auto &srctensors = entry->src;
        auto dsttensor = (entry->dst)[0];
        // Segment data if necessary
        if (dsttensor.nbytes() > bufsize) {
          ACCL::debug("dsttensor to large!");
          size_t n = bufsize / dsttensor.itemsize();
          for (size_t i = 0; i < dsttensor.numel(); i += n) {
            ACCL::debug("part " + std::to_string(i) + "!");
            size_t end =
                std::min(i + n, static_cast<size_t>(dsttensor.numel()));
            std::vector<at::Tensor> srctensorslices;
            srctensorslices.reserve(srctensors.size());
            for (auto &srctensor : srctensors) {
              srctensorslices.emplace_back(srctensor.slice(0, i, end));
            }
            run_scatter(srctensorslices, dsttensor.slice(0, i, end), opts);
          }
        } else {
          run_scatter(srctensors, dsttensor, opts);
        }
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::make_unique<WorkEntry>(&inputTensors[0], &outputTensors,
                                             std::move(runFunc));
    return enqueue(std::move(entry), "accl::scatter", OpType::SCATTER,
                   inputTensors.size() > 0
                       ? c10::optional<std::vector<at::Tensor>>(inputTensors[0])
                       : c10::nullopt);
  } else {
    auto entry = std::make_unique<WorkEntry>(nullptr, &outputTensors,
                                             std::move(runFunc));
    return enqueue(std::move(entry), "accl::scatter", OpType::SCATTER,
                   inputTensors.size() > 0
                       ? c10::optional<std::vector<at::Tensor>>(inputTensors[0])
                       : c10::nullopt);
  }
}

c10::intrusive_ptr<Work> ProcessGroupACCL::reduce_scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const ReduceScatterOptions &opts) {
  TORCH_CHECK(false, "ProcessGroupACCL does not support reduce_scatter");
}

void ProcessGroupACCL::run_alltoall(at::Tensor srctensor_original,
                                    at::Tensor dsttensor_original,
                                    const AllToAllOptions &opts) {
  at::Tensor *srctensor = &srctensor_original;
  at::Tensor *dsttensor = &dsttensor_original;
  at::Tensor empty_srctensor, empty_dsttensor;
  std::unique_ptr<ACCL::BaseBuffer> srcdata;
  std::unique_ptr<ACCL::BaseBuffer> dstdata;

  // Reserve device
  c10::DeviceGuard guard(srctensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Copy data from GPU to FPGA if necessary, and create a new result buffer,
  // since ACCL doesn't support in-place allreduce
  if (p2p_applicable(*accl, srctensor_original, p2p_enabled)) {
    srcdata = create_and_copy_p2p_buffer(*accl, srctensor_original);
  } else {
    if (accl->is_simulated() || coyote_enabled) {
      srcdata = create_buffer(*accl, srctensor->numel(), srctensor->scalar_type());
    } else {
      srcdata = wrap_buffer(*accl, buf0, srctensor->numel(), srctensor->scalar_type());
    }
    ACCL::debug("Copying data to aligned CPU tensor of size " +
                std::to_string(srctensor_original.numel()));
    empty_srctensor = torch::from_blob(
        srcdata->byte_array(), srctensor_original.sizes(),
        srctensor_original.options().device(c10::DeviceType::CPU));
    srctensor = &empty_srctensor;
    srctensor->copy_(srctensor_original);
    ACCL::debug("Creating extra result buffer of size " +
                std::to_string(srctensor_original.numel()));
  }

  // Create output buffer
  if (p2p_applicable(*accl, dsttensor_original, p2p_enabled)) {
    dstdata = create_and_copy_p2p_buffer(*accl, dsttensor_original);
  } else {
    if (accl->is_simulated() || coyote_enabled) {
      dstdata = create_buffer(*accl, dsttensor->numel(), dsttensor->scalar_type());
    } else {
      dstdata = wrap_buffer(*accl, buf0, dsttensor->numel(), dsttensor->scalar_type());
    }
  }

  // Run alltoall
  ACCL::debug("Starting alltoall of " + std::to_string(srctensor->numel()) +
              " items");
  if (!coyote_enabled) {
    srcdata->sync_to_device();
  }
  accl->alltoall(*srcdata, *dstdata, srctensor->numel(),
                  ACCL::GLOBAL_COMM, true, true,
                  get_compressed_type(srctensor->scalar_type()));
  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }
  if (!coyote_enabled) {
    dstdata->sync_from_device();
  }

  // Copy result buffer back to original tensor
  if (p2p_applicable(*accl, dsttensor_original, p2p_enabled)) {
    copy_back_p2p_buffer(*dstdata, dsttensor_original);
  } else {
    ACCL::debug("Copying result data back to original tensor of size " +
                std::to_string(dsttensor_original.numel()));
    dsttensor_original.copy_(torch::from_blob(
        dstdata->byte_array(), dsttensor_original.sizes(),
        dsttensor_original.options().device(c10::DeviceType::CPU)));
  }
}

c10::intrusive_ptr<Work> ProcessGroupACCL::alltoall_base(
    at::Tensor &outputTensor, at::Tensor &inputTensor,
    std::vector<int64_t> &outputSplitSizes,
    std::vector<int64_t> &inputSplitSizes, const AllToAllOptions &opts) {
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    // We can use alltoall
    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.type() == inputTensor.type(),
        "Tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "Tensor's dim 0 does not divide equally across group size");

    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [opts, this](std::unique_ptr<WorkEntry>& entry) {
          auto srctensor = (entry->src)[0];
          auto dsttensor = (entry->dst)[0];
          c10::DeviceGuard guard(srctensor.device());
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          // Segment data if necessary
          if (dsttensor.nbytes() > bufsize) {
            ACCL::debug("dsttensor to large!");
            size_t n = bufsize / dsttensor.itemsize();
            for (size_t i = 0; i < dsttensor.numel(); i += n) {
              ACCL::debug("part " + std::to_string(i) + "!");
              size_t end =
                  std::min(i + n, static_cast<size_t>(dsttensor.numel()));
              run_alltoall(srctensor.slice(0, i, end), dsttensor.slice(0, i, end), opts);
            }
          } else {
            run_alltoall(srctensor, dsttensor, opts);
          }
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "accl:all_to_all_base", OpType::ALLTOALL_BASE,
        c10::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    TORCH_CHECK(false, "ProcessGroupACCL does not support alltoallv required by alltoall_base");
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::alltoall(std::vector<at::Tensor> &outputTensors,
                           std::vector<at::Tensor> &inputTensors,
                           const AllToAllOptions &opts) {
  TORCH_CHECK(false, "ProcessGroupACCL does not support alltoall");
}

void ProcessGroupACCL::run_send(at::Tensor tensor_original, int dstRank,
                                int tag) {
  at::Tensor *tensor = &tensor_original;
  at::Tensor empty_tensor;
  std::unique_ptr<ACCL::BaseBuffer> data;

  // Reserve device
  c10::DeviceGuard guard(tensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Copy data from GPU to FPGA if necessary
  if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
    data = create_and_copy_p2p_buffer(*accl, tensor_original);
  } else {
    if (coyote_enabled) {
      // Copy tensor to CPU tensor first
      data = create_coyotebuffer(*accl, tensor->numel(), tensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(tensor_original.numel()));
      empty_tensor = torch::from_blob(
          data->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU));
      tensor = &empty_tensor;
      tensor->copy_(tensor_original);
    } else if (tensor_original.device().type() != c10::DeviceType::CPU) {
      // Copy tensor to CPU tensor first
      data = create_buffer(*accl, tensor->numel(), tensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(tensor_original.numel()));
      empty_tensor = torch::from_blob(
          data->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU));
      tensor = &empty_tensor;
      tensor->copy_(tensor_original);
    } else {
      data = create_buffer(*accl, *tensor);
    }
  }

  // Run send
  ACCL::debug("Starting send of " + std::to_string(tensor->numel()) +
              " items to " + std::to_string(dstRank));
  if (!coyote_enabled) {
    data->sync_to_device();
  }
  accl->send(*data, tensor->numel(), dstRank, tag, ACCL::GLOBAL_COMM, true,
             get_compressed_type(tensor->scalar_type()));

  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::send(std::vector<at::Tensor> &tensors, int dstRank, int tag) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [dstRank, tag, this](std::unique_ptr<WorkEntry> &entry) {
        at::Tensor &tensor = (entry->src)[0];
        // Segment data if necessary
        if (tensor.nbytes() > bufsize) {
          size_t n = bufsize / tensor.itemsize();
          for (size_t i = 0; i < tensor.numel(); i += n) {
            size_t end = std::min(i + n, static_cast<size_t>(tensor.numel()));
            run_send(tensor.slice(0, i, end), dstRank, tag);
          }
        } else {
          run_send(tensor, dstRank, tag);
        }
      };

  auto entry =
      std::make_unique<WorkEntry>(&tensors, nullptr, std::move(runFunc));
  return enqueue(std::move(entry), "accl::send", OpType::SEND,
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

void ProcessGroupACCL::run_recv(at::Tensor tensor_original, int srcRank,
                                int tag) {
  at::Tensor *tensor = &tensor_original;
  at::Tensor empty_tensor;
  std::unique_ptr<ACCL::BaseBuffer> data;

  // Reserve device
  c10::DeviceGuard guard(tensor->device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // Create FPGA buffer
  if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
    data = create_buffer_p2p(*accl, tensor_original);
  } else {
    if (coyote_enabled) {
      // Copy tensor to CPU tensor first
      data = create_coyotebuffer(*accl, tensor->numel(), tensor->scalar_type());
      ACCL::debug("Creating CPU tensor of size " +
                  std::to_string(tensor_original.numel()));
      empty_tensor = torch::from_blob(
          data->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU));
      tensor = &empty_tensor;
    } else if (tensor_original.device().type() != c10::DeviceType::CPU) {
      data = create_buffer(*accl, tensor->numel(), tensor->scalar_type());
      ACCL::debug("Copying data to CPU tensor of size " +
                  std::to_string(tensor_original.numel()));
      empty_tensor = torch::from_blob(
          data->byte_array(), tensor_original.sizes(),
          tensor_original.options().device(c10::DeviceType::CPU));
      tensor = &empty_tensor;
    } else {
      data = create_buffer(*accl, *tensor);
    }
  }

  // Run recieve
  ACCL::debug("Starting recieve of " + std::to_string(tensor->numel()) +
              " items from " + std::to_string(srcRank));
  accl->recv(*data, tensor->numel(), srcRank, tag, ACCL::GLOBAL_COMM, true,
             get_compressed_type(tensor->scalar_type()));

  int retcode = accl->get_retcode();
  if (retcode) {
    TORCH_CHECK(false, ACCL_ERROR(retcode));
  }

  if (!coyote_enabled) {
    data->sync_from_device();
  }

  // Copy data back to original tensor if necessary
  if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
    copy_back_p2p_buffer(*data, tensor_original);
  } else if (coyote_enabled || tensor_original.device().type() != c10::DeviceType::CPU) {
    ACCL::debug("Copying data back from CPU tensor of size " +
                std::to_string(tensor_original.numel()));
    tensor_original.copy_(*tensor);
  }
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::recv(std::vector<at::Tensor> &tensors, int srcRank, int tag) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [srcRank, tag, this](std::unique_ptr<WorkEntry> &entry) {
        const at::Tensor &tensor = (entry->dst)[0];
        // Segment data if necessary
        if (tensor.nbytes() > bufsize) {
          size_t n = bufsize / tensor.itemsize();
          for (size_t i = 0; i < tensor.numel(); i += n) {
            size_t end = std::min(i + n, static_cast<size_t>(tensor.numel()));
            run_recv(tensor.slice(0, i, end), srcRank, tag);
          }
        } else {
          run_recv(tensor, srcRank, tag);
        }
      };

  auto entry =
      std::make_unique<WorkEntry>(nullptr, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "accl::recv", OpType::RECV,
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::recvAnysource(std::vector<at::Tensor> &tensors, int tag) {
  TORCH_CHECK(false, "ProcessGroupACCL does not support recvAnysource");
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::barrier(const BarrierOptions &opts) {
  TORCH_CHECK(false, "ProcessGroupACCL does not support barrier");
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::_allgather_base(at::Tensor & /*unused */,
                                  at::Tensor & /*unused */,
                                  const AllgatherOptions & /*unused */) {
  TORCH_CHECK(false, "no support for _allgather_base in ACCL process group");
}

c10::intrusive_ptr<ProcessGroupACCL> createProcessGroupACCL(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    std::vector<c10::intrusive_ptr<ProcessGroupACCL::Rank>> &ranks,
    bool simulator, accl_network_utils::acclDesign design, bool p2p_enabled,
    std::map<ACCL::dataType, ACCL::dataType> compression,
    std::vector<int> profiling_ranks, double profiling_timeout,
    const std::string &xclbin, int device_index, int nbufs, uint64_t bufsize,
    bool rsfec) {
  return c10::make_intrusive<ProcessGroupACCL>(
      store, rank, size, ranks, simulator, p2p_enabled, compression,
      profiling_ranks, profiling_timeout, xclbin, design, device_index,
      nbufs, bufsize, rsfec);
}

// Define Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<ACCL::dataType>(m, "DataType")
      .value("float16", ACCL::dataType::float16, "16-bit floating-point number")
      .value("float32", ACCL::dataType::float32, "32-bit floating-point number")
      .value("float64", ACCL::dataType::float64, "64-bit floating-point number")
      .value("int32", ACCL::dataType::int32, "32-bit integer")
      .value("int64", ACCL::dataType::int64, "64-bit integer");

  py::enum_<accl_network_utils::acclDesign>(m, "ACCLDesign")
      .value("axis3x", accl_network_utils::acclDesign::AXIS3x,
             "Only applicable for hardware; uses UDP ACCL backend and loopback "
             "network kernel")
      .value("tcp", accl_network_utils::acclDesign::TCP,
             "TCP ACCL backend; uses EasyNet network kernel on hardware")
      .value("udp", accl_network_utils::acclDesign::UDP,
             "UDP ACCL backend; uses VNx network kernel on hardware")
      .value("cyt_tcp", accl_network_utils::acclDesign::CYT_TCP,
             "Only applicable for hardware; uses coyote ACCL backend with a "
             "TCP network kernel")
      .value("cyt_rdma", accl_network_utils::acclDesign::CYT_RDMA,
             "Only applicable for hardware; uses coyote ACCL backend with a "
             "RDMA network kernel");

  py::class_<ProcessGroupACCL::Rank,
             c10::intrusive_ptr<ProcessGroupACCL::Rank>>(m, "Rank")
      .def(py::init(&ProcessGroupACCL::Rank::create), py::arg("ip"),
           py::arg("port"), py::arg("session_id"), py::arg("max_segment_size"))
      .def_readonly("ip", &ProcessGroupACCL::Rank::ip)
      .def_readonly("port", &ProcessGroupACCL::Rank::port)
      .def_readonly("session_id", &ProcessGroupACCL::Rank::session_id)
      .def_readonly("max_segment_size",
                    &ProcessGroupACCL::Rank::max_segment_size)
      .def("__repr__", [](const ProcessGroupACCL::Rank &r) {
        return string_format(
            "accl_process_group.Rank(%s, %d, %d, %ju)",
            py::cast(r.ip).attr("__repr__")().cast<std::string>().c_str(),
            r.port, r.session_id, r.max_segment_size);
      });

  py::class_<ProcessGroupACCL, c10::intrusive_ptr<ProcessGroupACCL>,
             ProcessGroup>(m, "ProcessGroupACCL")
      .def(py::init(&createProcessGroupACCL), py::arg("store"), py::arg("rank"),
           py::arg("size"), py::arg("ranks"), py::arg("simulator"),
           py::arg("design"), py::kw_only(), py::arg("p2p_enabled") = false,
           py::arg("compression") = std::map<ACCL::dataType, ACCL::dataType>(),
           py::arg("profiling_ranks") = std::vector<int>(),
           py::arg("profiling_timeout") = 0.0,
           py::arg("xclbin") = std::string(),
           py::arg("device_index") = 0,  py::arg("nbufs") = 16,
           py::arg("bufsize") = 1024, py::arg("rsfec") = false)
      .def("get_local_qp", &ProcessGroupACCL::get_local_qp, py::arg("rank"))
      .def("set_remote_qp", &ProcessGroupACCL::set_remote_qp, py::arg("rank"),
           py::arg("qp"))
      .def("initialize", &ProcessGroupACCL::initialize)
      .def_property("compression", &ProcessGroupACCL::get_compression,
                    &ProcessGroupACCL::set_compression);
}

} // namespace c10d
