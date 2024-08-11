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
#include <signal.h>
#include <mpi.h>

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
using namespace ACCL;

namespace c10d {

// Toggles to run Collectives via OpenMPI instead(To sidestep any issues with them in ACCL)
// The sidestep-code is copied from the ProcessGroupMPI
#define SCATTER_SIDESTEP
#define GATHER_SIDESTEP
// #define ALLGATHER_SIDESTEP

#define BROADCAST_SIDESTEP false
// #define BROADCAST_SIDESTEP true

    
#define ALLREDUCE_SIDESTEP false
// #define ALLREDUCE_SIDESTEP true

// #define SIDESTEP_BCAST_WITH_ALLREDUCE
    
#define RDVZ_THRESHOLD 64

#define MICRO_BENCH_FINE 1

#define MICRO_BENCH_COARSE 1

#if MICRO_BENCH_FINE
#define START_FINE(name) \
  std::chrono::time_point<std::chrono::high_resolution_clock> start_##name  = std::chrono::high_resolution_clock::now();
#define STOP_FINE(name) \
  auto end_##name = std::chrono::high_resolution_clock::now();		\
  double durationUs_##name = (std::chrono::duration_cast<std::chrono::nanoseconds>(end_##name-start_##name).count() / 1000.0); \
  ACCL::debug(#name "_tensor durationUs:" + std::to_string(durationUs_##name));
#else
#define START_FINE(name)
#define STOP_FINE(name)
#endif

#if MICRO_BENCH_COARSE
#define START_COARSE(name) \
  std::chrono::time_point<std::chrono::high_resolution_clock> start_##name  = std::chrono::high_resolution_clock::now();
#define STOP_COARSE(name) \
  auto end_##name = std::chrono::high_resolution_clock::now();		\
  double durationUs_##name = (std::chrono::duration_cast<std::chrono::nanoseconds>(end_##name-start_##name).count() / 1000.0); \
  ACCL::debug(#name "_tensor durationUs:" + std::to_string(durationUs_##name));
#else
#define START_COARSE(name)
#define STOP_COARSE(name)
#endif

// Used in sidestepping
#define MPI_CHECK(cmd)                                                   \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      TORCH_CHECK(false, err);                                           \
    }                                                                    \
  } while (0)    

// Used in sidestepping    
// Op mapping
std::map<ReduceOp::RedOpType, MPI_Op> mpiOp = {
    {ReduceOp::MIN, MPI_MIN},
    {ReduceOp::MAX, MPI_MAX},
    {ReduceOp::SUM, MPI_SUM},
    {ReduceOp::PRODUCT, MPI_PROD},
};
// Used in sidestepping
// Type mapping
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};
    
#define CEIL_DIV(x, y) ((x) / (y) + ((x) % (y) != 0))

#define ACCL_ERROR(status)                                                     \
  ("ACCL error in: " + std::string(__FILE__) + ":" +                           \
   std::to_string(__LINE__) + ", with error code: " + std::to_string(status))

#if defined(ACCL_PROCESS_GROUP_HIP_ENABLED) &&                                 \
    defined(ACCL_PROCESS_GROUP_CUDA_ENABLED)
#error Cannot compile Process Group with both HIP and CUDA support
#endif // ACCL_PROCESS_GROUP_HIP_ENABLED && ACCL_PROCESS_GROUP_CUDA_ENABLED

// Activate Parameter printing:
#define DO_PARA_PRINT

#if defined(DO_PARA_PRINT)
  #define PARA_PRINT(x)							\
    ACCL::debug(#x "size: " + std::to_string(x.numel()) + " of type: " + string_of_accl_datatype(convert_datatype_from_torch(x.scalar_type())))
#else
  #define PARA_PRINT(x)
#endif


#define STANDARD_DECL \
  std::unique_ptr<ACCL::BaseBuffer> data;				\
  std::unique_ptr<ACCL::BaseBuffer> dstdata;				\

#define DO_COND ((do_on_root && opts_root_rank == rank_) || (do_on_others && opts_root_rank != rank_))

#define PRE_REQUEST(opname, tensor)					\
  START_FINE(type)    \
  in_buf->change_type(convert_datatype_from_torch(tensor.scalar_type())); \
  out_buf->change_type(convert_datatype_from_torch(tensor.scalar_type()));   \
  STOP_FINE(type)						\
  ACCL::debug("[" #opname "] Entering barrier");			\
  START_FINE(barrier)    \
  accl->barrier();							\
  STOP_FINE(barrier)							\
  ACCL::debug("Performing " #opname " of " + std::to_string(tensor.numel()) + " items"); \
  START_FINE(lib)							

#define POST_REQUEST(opname, n_bytes)				\
STOP_FINE(lib)						

#define TIMER_WRAP()
    
// Better logging
// accl_log(mpi_rank, format_log("bcast", options, durationUs, 0));

  
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


std::string format_log(std::string collective, int world_size, int rank, double time, int n_bytes)
{
    std::string log_str = collective + "," + std::to_string(world_size) + "," + std::to_string(rank) + "," + std::to_string(time) + "," + std::to_string(n_bytes);
    return log_str;
}    

#define ACCL_PG_LOG_FILE(i)                                                       \
  (std::string("accl_log/accl_pg_") + i + std::string(".log"))    
    
void accl_pg_log(int rank, const std::string &message) {
  std::string str_rank = std::to_string(rank);
  std::string filename = ACCL_PG_LOG_FILE(str_rank);
  std::ofstream outfile;
  outfile.open(filename, std::ios::out | std::ios_base::app);
  outfile << message << std::endl;
  outfile.close();
}
    
    
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

const char *string_of_accl_datatype(ACCL::dataType accl_type) {
  switch (accl_type) {
  case ACCL::dataType::float16:
    return "ACCL::dataType::float16";
  case ACCL::dataType::float32:
    return "ACCL::dataType::float32";
  case ACCL::dataType::float64:
    return "ACCL::dataType::float64";
  case ACCL::dataType::int32:
    return "ACCL::dataType::int32";
  case ACCL::dataType::int64:
    return "ACCL::dataType::int64";
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

// just for the sa_handler
std::unique_ptr<::ACCL::ACCL>* global_accl;  

void accl_sa_handler(int)
{
	static bool once = true;
	if(once) {
		global_accl->reset();
		// std::cout << "Error! Signal received. Finalizing MPI..." << std::endl;
		// MPI_Finalize();
		// std::cout << "Done. Terminating..." << std::endl;
		once = false;
	}
	exit(EXIT_FAILURE);
}

// TODO delete when not needed anymore
void ProcessGroupACCL::init_input_tensor(at::Tensor &tensor, std::unique_ptr<ACCL::BaseBuffer> &data, bool do_on_root, bool do_on_others, int opts_root_rank) {
  if DO_COND {
	ACCL::debug("Copying data to CPU tensor of size " + std::to_string(tensor.numel()));
	// at::Tensor wrapper_tensor = torch::from_blob(data->byte_array(), tensor.sizes(), tensor.options().device(c10::DeviceType::CPU)); 
	// wrapper_tensor.copy_(tensor);
	std::memcpy(data->byte_array(), tensor.data_ptr(), tensor.numel() * tensor.element_size());

	//TODO check if necessary in coyote
	if (!coyote_enabled) {
	    data->sync_to_device();
	}
	
    }
    // don't sync if no rank initializes, we will fill content and sync later
}

  void ProcessGroupACCL::init_input_data_vec(std::vector<at::Tensor> &tensor_vec, std::unique_ptr<ACCL::BaseBuffer> &data, const at::TensorOptions &options, bool do_on_root, bool do_on_others, int opts_root_rank) {
  if DO_COND {
    int64_t tens_size = static_cast<size_t>(tensor_vec[0].numel());
    int64_t total_size = tens_size * static_cast<size_t>(size_);
    std::vector<int64_t> sizes = tensor_vec[0].sizes().vec();
    // Prepend another dimension for vector length
    sizes.insert(sizes.begin(), tensor_vec.size());
      
    ACCL::debug("Copying data to CPU tensor of size " + std::to_string(total_size));
    at::Tensor wrapper_tensor = torch::from_blob(data->byte_array(), sizes, options);

    for (const auto i : c10::irange(tensor_vec.size())) {
      if (p2p_applicable(*accl, tensor_vec[0], p2p_enabled)) {
	auto slice = data->slice(i * tens_size, (i + 1) * tens_size);
	copy_to_p2p_buffer(*slice, tensor_vec[i]);
      } else {
	auto slice = wrapper_tensor[i];
	slice.copy_(tensor_vec[i]);
      }
    }
    if (!coyote_enabled) {
      data->sync_to_device();
    }
  // } else {
    // data = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }
}  
  
  // like init_output_tensor but without needlessly setting the tensor
  // TODO: remove once all collectives reuse the buffer
void ProcessGroupACCL::init_output_data(at::Tensor &tensor_original, std::unique_ptr<ACCL::BaseBuffer> &dstdata, int out_tensor_size, c10::ScalarType type, bool do_on_root, bool do_on_others, int opts_root_rank) {
  if DO_COND {
      if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
	dstdata = create_buffer_p2p(*accl, out_tensor_size, type);
    } else {
	if (coyote_enabled) {
	  dstdata = create_coyotebuffer(*accl, out_tensor_size, type);
	} else {
	  dstdata = create_buffer(*accl, out_tensor_size, type);
	}
    }
  } else {
    dstdata = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }
}

    void ProcessGroupACCL::init_output_tensor(const at::Tensor &tensor_original, at::Tensor &dsttensor, std::unique_ptr<ACCL::BaseBuffer> &dstdata, int num_tensors, c10::ScalarType type, bool do_on_root, bool do_on_others, int opts_root_rank) {
    if DO_COND {
	int64_t num_tensors_s = static_cast<size_t>(num_tensors);
	std::vector<int64_t> sizes = tensor_original.sizes().vec();
	int64_t total_size = static_cast<size_t>(tensor_original.numel());
	if  (num_tensors != 0) {
	    // Prepend another dimension for vector length
	    sizes.insert(sizes.begin(), num_tensors_s);
	    total_size = total_size * num_tensors_s;
	}
	
	if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
	  dstdata = create_buffer_p2p(*accl, total_size, type);
	} else if (coyote_enabled) {
	    // std::vector<int64_t> sizes = {static_cast<int64_t>(out_tensor_size)};
	    dsttensor = torch::from_blob(dstdata->byte_array(), sizes, tensor_original.options().device(c10::DeviceType::CPU));
	    // This should not be necessary:
	    // dsttensor.copy_(tensor_original);
	} else {
	    dsttensor = torch::from_blob(dstdata->byte_array(), sizes, tensor_original.options().device(c10::DeviceType::CPU));
	    // This should not be necessary:
	    // dsttensor.copy_(tensor_original);
	}
      } else {
      dsttensor = at::Tensor(nullptr);
    }
}
  
void ProcessGroupACCL::copy_back_tensor(at::Tensor tensor_original, std::unique_ptr<ACCL::BaseBuffer> &data, bool do_on_root, bool do_on_others, int opts_root_rank){
  if DO_COND {
      if (!coyote_enabled) {
	data->sync_from_device();
      }
      if (p2p_applicable(*accl, tensor_original, p2p_enabled)) {
	copy_back_p2p_buffer(*data, tensor_original);
      } else {
	ACCL::debug("Copying data back from CPU tensor of size " +
		    std::to_string(tensor_original.numel()));
	std::memcpy(tensor_original.data_ptr(), data->byte_array(), tensor_original.numel() * tensor_original.element_size());
	// tensor_original.copy_(torch::from_blob(data->byte_array(), tensor_original.sizes(), tensor_original.options().device(c10::DeviceType::CPU)));
	ACCL::debug("Finished Copying ");
      }
  }
}

  void ProcessGroupACCL::copy_back_tensorvec(const std::vector<at::Tensor> &dsttensorvec, std::unique_ptr<ACCL::BaseBuffer> &data, at::Tensor &dsttensor, int numel, bool do_on_root, bool do_on_others, int opts_root_rank){
  if DO_COND {
    if (!coyote_enabled) {
      data->sync_from_device();
    }
    for (const auto i : c10::irange(dsttensorvec.size())) {
	// TODO uncomment and correct
      // if (p2p_applicable(*accl, dsttensorvec[0], p2p_enabled)) {
	// auto slice =
	  // data->slice(i * numel, (i + 1) * numel);
	// copy_back_p2p_buffer(*slice, dsttensorvec[i]);
      // } else {
	dsttensorvec[i].copy_(dsttensor[i]);
      // }
    }
  }
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

  ACCL::debug("Process Group constructor called");
  
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = accl_sa_handler;
  sigfillset(&sa.sa_mask);
  sigaction(SIGINT,&sa,NULL);
  sigaction(SIGSEGV, &sa, NULL);
  
  if (std::find(profiling_ranks.begin(), profiling_ranks.end(), rank) !=
      profiling_ranks.end()) {
    std::this_thread::sleep_for(
        std::chrono::duration<double>(profiling_timeout));
  }
  
  ACCL::debug("Converting ranks");
  ranks_ = convert_ranks(ranks);
  design_ = design;
  MPI_Barrier(MPI_COMM_WORLD);
  if (!simulator){
    if (coyote_enabled) {
      if (design_ == accl_network_utils::acclDesign::CYT_TCP) {
        cyt_device = new ACCL::CoyoteDevice();
	accl_network_utils::configure_cyt_tcp(ranks_, rank_, cyt_device);
      } else if (design_ == accl_network_utils::acclDesign::CYT_RDMA) {
	ACCL::debug("Creating CoyoteDevice");
        cyt_device = new ACCL::CoyoteDevice(size_);
	accl_network_utils::configure_cyt_rdma(ranks_, rank_, cyt_device);
      } else {
        throw std::runtime_error("Undefined ACCL design");
      }
      // create the two buffers, which are gonna be reused during calls
      // We use float32, but they are gonna be filled arbitrarily

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

    accl = std::make_unique<ACCL::ACCL>(cyt_device);
    global_accl = &accl;

    // Rendezvous protocol for now
    int protoc = 1;
    // default from test.cpp
    int segsize = 4096 * 1024;

    
    if (protoc == 0){
      std::cout<<"Eager Protocol"<<std::endl;
      accl.get()->initialize(ranks_, rank_, size_+2, bufsize, segsize, 4096*1024*2);
    } else {
      std::cout<<"Rendezvous Protocol"<<std::endl;
      accl.get()->initialize(ranks_, rank_, 16, 1024, RDVZ_THRESHOLD);
    }  
    
    ACCL::debug(std::string("[ACCL coyote] communicator: ") + accl->dump_communicator());



  } else {
    // ACCL::debug(std::string("Error XRT initialization deprecated"));
    accl = accl_network_utils::initialize_accl(ranks_, rank_,
                                               simulator_, design_, xrt_device,
                                               xclbin_, nbufs_, bufsize, 0,
                                               rsfec_);
    ACCL::debug(std::string("Setting timeout and Threshold"));
    accl->set_timeout(1e6);
    // accl->set_rendezvous_threshold(16*1024);
                                      
    int devicemem = accl->devicemem();

    // Not sure if this is needed:
    // Initialize cache buffers
    // if (!simulator_){
	// buf0 = xrt::bo(xrt_device, bufsize, devicemem);
	// buf1 = xrt::bo(xrt_device, bufsize, devicemem);
    // }
    
	    
  }

  in_buf = accl->create_buffer_host<float>(bufsize/sizeof(float), ACCL::dataType::float32);
  out_buf = accl->create_buffer_host<float>(bufsize/sizeof(float), ACCL::dataType::float32);
  
  accl->set_timeout(1e8);
  // Start the worker thread accepting ACCL calls
  workerThread_ = std::thread(&ProcessGroupACCL::runLoop, this);
  initialized = true;
  ACCL::debug(std::string("Finished Initialization"));
}

ProcessGroupACCL::~ProcessGroupACCL() { destroy(); }

void ProcessGroupACCL::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

  //TODO free other buffer types
  if (!simulator_) {
      // if(coyote_enabled){
	  in_buf->free_buffer();
	  out_buf->free_buffer();
      // }
  }

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

void ProcessGroupACCL::run_broadcast(at::Tensor in_tensor,
                                     const BroadcastOptions &opts) {

  std::chrono::time_point<std::chrono::high_resolution_clock> start_inner  = std::chrono::high_resolution_clock::now();

  // This case split is necessary, because otherwise data will be set to a nullptr
  #ifdef SIDESTEP_BCAST_WITH_ALLREDUCE
  START_FINE(init)
  
  if (opts.rootRank == rank_){
      init_input_tensor(in_tensor, in_buf, true, false, opts.rootRank);
  }
  else{
      auto zero_tensor = torch::zeros({in_tensor.numel()}, in_tensor.scalar_type());
      init_input_tensor(zero_tensor, in_buf, false, true, opts.rootRank);
  }

  STOP_FINE(init)

  START_FINE(lock)      
  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  STOP_FINE(lock)      

  PRE_REQUEST(Broadcast,in_tensor)  


  // It seems to have issues with non-even numbers, so we round to 256
  int rounded_count = (in_tensor.numel() + 1023) & ~1023;

  
  accl->allreduce(*in_buf, *out_buf, rounded_count, ACCL::reduceFunction::SUM);      

  POST_REQUEST("allreduce", in_tensor.nbytes())

  START_FINE(copy)      
  copy_back_tensor(in_tensor, out_buf, true, true);
  STOP_FINE(copy)
  
  #else

  START_FINE(init)
      
  if (opts.rootRank == rank_){
      init_input_tensor(in_tensor, in_buf, true, false, opts.rootRank);
  }
  // else{
      // init_output_data(in_tensor, in_buf, in_tensor.numel(), in_tensor.scalar_type(), false, true, opts.rootRank);
  // }

  STOP_FINE(init)
  
  // Reserve device

      
  START_FINE(lock)
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  STOP_FINE(lock)
  
  PRE_REQUEST(Broadcast,in_tensor)

  int rounded_count = (in_tensor.numel() + 1023) & ~1023;

  accl->bcast(*in_buf, rounded_count, opts.rootRank);

  POST_REQUEST("bcast", in_tensor.nbytes())

  // in_buf->sync_from_device();
  // for(int i = 0; i<in_tensor.numel(); i++){
      // ACCL::debug(std::to_string(((double *) in_buf->byte_array())[i]));
  // }
  START_FINE(copy)
  copy_back_tensor(in_tensor, in_buf, true, true, opts.rootRank);
  STOP_FINE(copy)
  #endif
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::broadcast(std::vector<at::Tensor> &tensors,
                            const BroadcastOptions &opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
	ACCL::debug("Starting Broadcast");
	// if (((entry->src)[0]).numel() <= RDVZ_THRESHOLD || BROADCAST_SIDESTEP){
	if (BROADCAST_SIDESTEP){
	    
	auto data = (entry->src)[0];
	ACCL::debug("[Broadcast] -- Sidestepped using OpenMPI -- size " + std::to_string(data.numel()));
	c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            MPI_COMM_WORLD));
	} else {
	START_COARSE(total)    
	at::Tensor &tensor = (entry->src)[0];
        // Segment data if necessary
        if (tensor.nbytes() > bufsize / 2) {
	  size_t non_zero_dim_count = tensor.numel() / tensor.size(0);
          size_t n = bufsize / 2 / tensor.itemsize() / non_zero_dim_count;
	  ACCL::debug("[Broadcast] Segmenting tensor of size " + std::to_string(tensor.nbytes()) + " into " + std::to_string(n * non_zero_dim_count) + "-sized elements ");
          for (size_t i = 0; i < tensor.size(0); i += n) {
	    ACCL::debug("part " + std::to_string(i) + "!");
            size_t end = std::min(i + n, static_cast<size_t>(tensor.size(0)));
            run_broadcast(tensor.slice(0, i, end), opts);
          }
        } else {
	  ACCL::debug("[Broadcast] Broadcasting entire tensor of size " + std::to_string(tensor.nbytes()) + " without segmentation.");
          run_broadcast(tensor, opts);
        }
	STOP_COARSE(total)    
	}
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "accl::broadcast", OpType::BROADCAST,
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

void ProcessGroupACCL::run_allreduce(at::Tensor in_tensor,
                                     const AllreduceOptions &opts) {

  START_FINE(init)
  
  init_input_tensor(in_tensor, in_buf, true, true);

  STOP_FINE(init)


  START_FINE(lock)      
  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  STOP_FINE(lock)      

  PRE_REQUEST(Allreduce,in_tensor)  


  // It seems to have issues with non-even numbers, so we round to 256
  int rounded_count = (in_tensor.numel() + 1023) & ~1023;

  
  accl->allreduce(*in_buf, *out_buf, rounded_count, acclOp.at(opts.reduceOp));      

  POST_REQUEST("allreduce", in_tensor.nbytes())

  START_FINE(copy)      
  copy_back_tensor(in_tensor, out_buf, true, true);
  STOP_FINE(copy)
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::allreduce(std::vector<at::Tensor> &tensors,
                            const AllreduceOptions &opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
	ACCL::debug("Starting Allreduce");
	// sidestep eager allreduce
	// if (((entry->src)[0]).numel() <= RDVZ_THRESHOLD || ALLREDUCE_SIDESTEP){
	  if (ALLREDUCE_SIDESTEP){
	    auto data = (entry->src)[0];
	    ACCL::debug("[Allreduce] -- Sidestepped using OpenMPI -- size " + std::to_string(data.numel()));
	    c10::DeviceGuard guard(data.device());
	    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
	    MPI_CHECK(MPI_Allreduce(
			  MPI_IN_PLACE,
			  data.data_ptr(),
			  data.numel(),
			  mpiDatatype.at(data.scalar_type()),
			  mpiOp.at(opts.reduceOp),
			  MPI_COMM_WORLD));
	} else {
	    START_COARSE(total)    
	    auto tensor = (entry->src)[0];
	    // Segment data if necessary
	    if (tensor.nbytes() > bufsize/2) {
		size_t non_zero_dim_count = tensor.numel() / tensor.size(0);
		size_t n = bufsize / 2 / tensor.itemsize() / non_zero_dim_count;
		ACCL::debug("[Allreduce] Segmenting tensor of size " + std::to_string(tensor.nbytes()) + " into " + std::to_string(n * non_zero_dim_count) + "-sized elements ");
		for (size_t i = 0; i < tensor.size(0); i += n) {
		    ACCL::debug("part " + std::to_string(i) + "!");
		    size_t end = std::min(i + n, static_cast<size_t>(tensor.size(0)));
		    run_allreduce(tensor.slice(0, i, end), opts);
		}
	    } else {
		run_allreduce(tensor, opts);
	    }
	    STOP_COARSE(total)    
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

void ProcessGroupACCL::run_reduce(at::Tensor in_tensor,
                                  const ReduceOptions &opts) {

  init_input_tensor(in_tensor, in_buf, true, true);    

  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  PRE_REQUEST(Reduce,in_tensor)  

  accl->reduce(*in_buf, *out_buf, in_tensor.numel(), opts.rootRank, acclOp.at(opts.reduceOp));

  POST_REQUEST("reduce", in_tensor.nbytes())

  copy_back_tensor(in_tensor, out_buf, true, false, opts.rootRank);
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
    at::Tensor in_tensor,
    const std::vector<at::Tensor> &dsttensorvec) {

    
  at::Tensor dsttensor;

  init_input_tensor(in_tensor, in_buf, true, true);    
  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  init_output_tensor(in_tensor, dsttensor, out_buf, size_, in_tensor.scalar_type(), true, true);
  
  PRE_REQUEST(Allgather,in_tensor)

  accl->allgather(*in_buf, *out_buf, in_tensor.numel());

  POST_REQUEST("allgather", in_tensor.nbytes())

  copy_back_tensorvec(dsttensorvec, out_buf, dsttensor, in_tensor.numel(), true, true);
    
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
        #ifdef ALLGATHER_SIDESTEP
	ACCL::debug("[AllGather] -- Sidestepped using OpenMPI --");
	auto data = (entry->src)[0];
        std::vector<at::Tensor> outputDataVec = entry->dst;
        auto flatOutputTensor = newLikeFlat(outputDataVec);

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            flatOutputTensor.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            MPI_COMM_WORLD));

        for (const auto i : c10::irange(outputDataVec.size())) {
          outputDataVec[i].copy_(flatOutputTensor[i]);
        }
        #else
        auto srctensor = (entry->src)[0];
        auto &dsttensors = entry->dst;
        // Segment data if necessary
        if (srctensor.nbytes() > bufsize) {
	  size_t non_zero_dim_count = srctensor.numel() / srctensor.size(0);
          size_t n = bufsize / srctensor.itemsize() / non_zero_dim_count;
          for (size_t i = 0; i < srctensor.size(0); i += n) {
            size_t end =
                std::min(i + n, static_cast<size_t>(srctensor.size(0)));
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
      #endif
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

void ProcessGroupACCL::run_gather(at::Tensor in_tensor,
                                  const std::vector<at::Tensor> &dsttensorvec,
                                  const GatherOptions &opts) {
  at::Tensor dsttensor;

  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  
  init_input_tensor(in_tensor, in_buf, true, true);

  init_output_tensor(in_tensor, dsttensor, out_buf, size_, in_tensor.scalar_type(), true, false, opts.rootRank);

  PRE_REQUEST(Gather, in_tensor)

  accl->gather(*in_buf, *out_buf, in_tensor.numel(), opts.rootRank);

  POST_REQUEST("gather", in_tensor.nbytes())

  copy_back_tensorvec(dsttensorvec, out_buf, dsttensor, in_tensor.numel(), true, false, opts.rootRank);
    
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
	#ifdef GATHER_SIDESTEP
	ACCL::debug("[Gather] -- Sidestepped using OpenMPI --");
	  auto data = (entry->src)[0];
        void* recvbuf = nullptr;
        at::Tensor flatOutputTensor;

        std::vector<at::Tensor> dstdata = entry->dst;
        if (rank_ == opts.rootRank) {
          flatOutputTensor = newLikeFlat(dstdata);
          recvbuf = flatOutputTensor.data_ptr();
        }

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            MPI_COMM_WORLD));

        if (rank_ == opts.rootRank) {
          const std::vector<at::Tensor>& outputDataVec = entry->dst;
          // copy the flattened output tensors to the outputs
          for (const auto i : c10::irange(outputDataVec.size())) {
            outputDataVec.at(i).copy_(flatOutputTensor[i]);
          }
        }
	#else
        auto srctensor = (entry->src)[0];
        auto &dsttensors = entry->dst;
        // Segment data if necessary
        if (srctensor.nbytes() > bufsize) {
	  size_t non_zero_dim_count = srctensor.numel() / srctensor.size(0);
          size_t n = bufsize / srctensor.itemsize() / non_zero_dim_count;
	  ACCL::debug("[Gather] Segmenting tensor of size " + std::to_string(srctensor.nbytes()) + " into " + std::to_string(n * non_zero_dim_count) + "-sized elements ");
          for (size_t i = 0; i < srctensor.size(0); i += n) {
            size_t end =
                std::min(i + n, static_cast<size_t>(srctensor.size(0)));
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
      #endif
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

void ProcessGroupACCL::run_scatter(std::vector<at::Tensor> &in_tensor_vec,
                                   at::Tensor out_tensor,
                                   const ScatterOptions &opts) {
  at::Tensor dsttensor;

  // Reserve device
  c10::DeviceGuard guard(out_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  init_input_data_vec(in_tensor_vec, in_buf, out_tensor.options().device(c10::DeviceType::CPU), true, false, opts.rootRank);
  
  init_output_tensor(out_tensor, dsttensor, out_buf, 0, out_tensor.scalar_type(), true, true, opts.rootRank);

  PRE_REQUEST(Scatter, dsttensor)
  
  // Run scatter
  accl->scatter(*in_buf, *out_buf, out_tensor.numel(), opts.rootRank);

  POST_REQUEST("scatter", out_tensor.nbytes())

  copy_back_tensor(out_tensor, out_buf, true, true, opts.rootRank);
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
        #ifdef SCATTER_SIDESTEP
	ACCL::debug("[Scatter] -- Sidestepped using OpenMPI --");
	auto data = (entry->dst)[0];
        void* sendbuf = nullptr;
        at::Tensor flatInputTensor;

        if (rank_ == opts.rootRank) {
          std::vector<at::Tensor>& inputDataVec = entry->src;
          flatInputTensor = newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

          // copy the input tensors to the flatten large send buffer
          for (const auto i : c10::irange(inputDataVec.size())) {
            flatInputTensor[i].copy_(inputDataVec.at(i));
          }
        }

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Scatter(
            sendbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            MPI_COMM_WORLD));
        #else
        auto &srctensors = entry->src;
        auto dsttensor = (entry->dst)[0];
        // Segment data if necessary
        if (dsttensor.nbytes() > bufsize) {
          ACCL::debug("dsttensor to large!");
	  size_t non_zero_dim_count = dsttensor.numel() / dsttensor.size(0);
          size_t n = bufsize / dsttensor.itemsize() / non_zero_dim_count;
          for (size_t i = 0; i < dsttensor.size(0); i += n) {
            ACCL::debug("part " + std::to_string(i) + "!");
            size_t end =
                std::min(i + n, static_cast<size_t>(dsttensor.size(0)));
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
        #endif
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

void ProcessGroupACCL::run_alltoall(at::Tensor in_tensor,
                                    at::Tensor out_tensor,
                                    const AllToAllOptions &opts) {

  START_FINE(init)
  init_input_tensor(in_tensor, in_buf, true, true);
  STOP_FINE(init)

  // Reserve device
  START_FINE(lock)      
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  STOP_FINE(lock)

  // init_output_data(out_tensor, dstdata, out_tensor.numel(), out_tensor.scalar_type(), true, true);

  PRE_REQUEST(AlltoAll, in_tensor)

  accl->alltoall(*in_buf, *out_buf, in_tensor.numel()/size_);

  POST_REQUEST("alltoall", in_tensor.nbytes()/size_)

  START_FINE(copy)      
  copy_back_tensor(out_tensor, out_buf, true, true);    
  STOP_FINE(copy)
}

    
void ProcessGroupACCL::run_alltoall_vec(std::vector<at::Tensor> &in_tensor_vec,
                                    std::vector<at::Tensor> &out_tensor_vec,
                                    const AllToAllOptions &opts) {
  at::Tensor dsttensor;

  // Reserve device
  c10::DeviceGuard guard(in_tensor_vec[0].device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  init_input_data_vec(in_tensor_vec, in_buf, out_tensor_vec[0].options().device(c10::DeviceType::CPU), true, true);

  init_output_tensor(in_tensor_vec[0], dsttensor, out_buf, size_, in_tensor_vec[0].scalar_type(), true, true);
  
  PRE_REQUEST(AlltoAll, in_tensor_vec[0])

  accl->alltoall(*in_buf, *out_buf, in_tensor_vec[0].numel());

  POST_REQUEST("alltoall", in_tensor_vec[0].nbytes())

  copy_back_tensorvec(out_tensor_vec, out_buf, dsttensor, in_tensor_vec[0].numel(), true, true);
      
}

c10::intrusive_ptr<Work> ProcessGroupACCL::alltoall_base(
    at::Tensor &outputTensor, at::Tensor &inputTensor,
    std::vector<int64_t> &outputSplitSizes,
    std::vector<int64_t> &inputSplitSizes, const AllToAllOptions &opts) {
    ACCL::debug("Starting AlltoAll");
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    // We can use alltoall
    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.scalar_type() == inputTensor.scalar_type(),
        "Tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "Tensor's dim 0 does not divide equally across group size");

    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [opts, this](std::unique_ptr<WorkEntry>& entry) {
	  START_COARSE(total)    
          auto srctensor = (entry->src)[0];
          auto dsttensor = (entry->dst)[0];


          // c10::DeviceGuard guard(srctensor.device());
          // std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          // Segment data if necessary
          if (dsttensor.nbytes() > bufsize) {
            ACCL::debug("dsttensor to large!");

	    // Split individual entries
	    size_t non_zero_dim_count = dsttensor.numel() / dsttensor.size(0);
	    size_t n = bufsize / dsttensor.itemsize() / size_ / non_zero_dim_count;
	    size_t entry_size = dsttensor.numel() / size_ / non_zero_dim_count;
            for (size_t i = 0; i < entry_size; i += n) {
              ACCL::debug("part " + std::to_string(i) + "!");
              size_t end = std::min(i + n, static_cast<size_t>(entry_size));

	      std::vector<at::Tensor> srctensorslices;
	      srctensorslices.reserve(size_);
	      ACCL::debug("dsttensorslices:");
	      for (int j = 0; j < size_; j++) {
		  int bufpos = j * entry_size;
		  srctensorslices.emplace_back(srctensor.slice(0, i + bufpos, end + bufpos));
	      }
	      std::vector<at::Tensor> dsttensorslices;
	      dsttensorslices.reserve(size_);
	      ACCL::debug("dsttensorslices:");
	      for (int j = 0; j < size_; j++) {
		  int bufpos = j * entry_size;
		  dsttensorslices.emplace_back(dsttensor.slice(0, i + bufpos, end + bufpos));
	      }
              run_alltoall_vec(srctensorslices, dsttensorslices, opts);
            }
          } else {
	    ACCL::debug("Running without segmentation");
            run_alltoall(srctensor, dsttensor, opts);
          }
	STOP_COARSE(total)	  
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
  ACCL::debug("ProcessGroupACCL does not support alltoall");
  TORCH_CHECK(false, "ProcessGroupACCL does not support alltoall");
}

void ProcessGroupACCL::run_send(at::Tensor in_tensor, int dstRank,
                                int tag) {

  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  init_input_tensor(in_tensor, in_buf, true, true);

  PRE_REQUEST(Send,in_tensor)
  
  ACCL::ACCLRequest* req = accl->send(*in_buf, in_tensor.numel(), dstRank, tag);

  POST_REQUEST("send", in_tensor.nbytes())
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

void ProcessGroupACCL::run_recv(at::Tensor out_tensor, int srcRank,
                                int tag) {

  // Reserve device
  c10::DeviceGuard guard(out_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // init_output_data(out_tensor, dstdata, out_tensor.numel(), out_tensor.scalar_type(), true, true);

  PRE_REQUEST(Receive, out_tensor)  
  
  ACCL::ACCLRequest* req = accl->recv(*out_buf, out_tensor.numel(), srcRank, tag);

  POST_REQUEST("recv", out_tensor.nbytes())

  copy_back_tensor(out_tensor, out_buf, true, true);      
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
