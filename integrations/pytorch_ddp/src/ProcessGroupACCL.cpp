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

namespace py = pybind11;
using namespace ACCL;

namespace c10d {

// Toggles to run Collectives via OpenMPI instead(To sidestep any issues with them in ACCL)
// The sidestep-code is copied from the ProcessGroupMPI
// #define SCATTER_SIDESTEP
// #define GATHER_SIDESTEP
// #define ALLGATHER_SIDESTEP

#define BROADCAST_SIDESTEP false
// #define BROADCAST_SIDESTEP true

    
#define ALLREDUCE_SIDESTEP false
// #define ALLREDUCE_SIDESTEP true

// #define SIDESTEP_BCAST_WITH_ALLREDUCE
    
#define RDVZ_THRESHOLD 64

// This is the maximal message size. larger sizes get segmented
#define ACCL_MSG_SIZE 2097152

// counts are rounded up to this number for stability reasons
#define ROUND_NR 256

// This is intended for debugging, you can refer to the name of the collective using this
#define COLL_NAME UNNAMED

#define x_MAKE_STRING(s) MAKE_STRING(s)
#define MAKE_STRING(s) #s    

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

#if defined(ACCL_PROCESS_GROUP_HIP_ENABLED) &&                                 \
    defined(ACCL_PROCESS_GROUP_CUDA_ENABLED)
#error Cannot compile Process Group with both HIP and CUDA support
#endif // ACCL_PROCESS_GROUP_HIP_ENABLED && ACCL_PROCESS_GROUP_CUDA_ENABLED

#define DO_COND ((do_on_root && opts_root_rank == rank_) || (do_on_others && opts_root_rank != rank_))

#define PRE_REQUEST(opname, tensor)					\
  in_buf->change_type(convert_datatype_from_torch(tensor.scalar_type())); \
  out_buf->change_type(convert_datatype_from_torch(tensor.scalar_type()));   \
  ACCL::debug("Performing " #opname " of " + std::to_string(tensor.numel()) + " items")

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

const char *string_of_torch_datatype(c10::ScalarType torch_type) {
  switch (torch_type) {
  case at::kHalf:
      return "torch.float16";
  case at::kFloat:
    return "torch.float32";
  case at::kDouble:
    return "torch.float64";
  case at::kInt:
    return "torch.int32";
  case at::kLong:
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

void ProcessGroupACCL::init_input_tensor(at::Tensor &tensor, std::unique_ptr<ACCL::BaseBuffer> &data, bool do_on_root, bool do_on_others, int opts_root_rank) {
  if DO_COND {
	std::memcpy(data->byte_array(), tensor.data_ptr(), tensor.numel() * tensor.element_size());
	if (!coyote_enabled) {
	    data->sync_to_device();
	}
    }
}

void ProcessGroupACCL::init_input_data_vec(std::vector<at::Tensor> &tensor_vec, std::unique_ptr<ACCL::BaseBuffer> &data, const at::TensorOptions &options, bool do_on_root, bool do_on_others, int opts_root_rank) {
  if DO_COND {
    int64_t tens_size = static_cast<size_t>(tensor_vec[0].numel());
    int64_t total_size = tens_size * static_cast<size_t>(size_);
      
    for (const auto i : c10::irange(tensor_vec.size())) {
	std::memcpy(data->byte_array() + i * tens_size * tensor_vec[0].element_size(), tensor_vec[i].data_ptr(), tens_size * tensor_vec[0].element_size());
    }
    if (!coyote_enabled) {
      data->sync_to_device();
    }
  }
}  

void ProcessGroupACCL::copy_back_tensor(at::Tensor tensor_original, std::unique_ptr<ACCL::BaseBuffer> &data, bool do_on_root, bool do_on_others, int opts_root_rank){
  if DO_COND {
      if (!coyote_enabled) {
	data->sync_from_device();
      }
      std::memcpy(tensor_original.data_ptr(), data->byte_array(), tensor_original.numel() * tensor_original.element_size());
  }
}

void ProcessGroupACCL::copy_back_tensorvec(const std::vector<at::Tensor> &dsttensorvec, std::unique_ptr<ACCL::BaseBuffer> &data, at::Tensor &dsttensor, int numel, int offset, bool do_on_root, bool do_on_others, int opts_root_rank){
  if DO_COND {
    if (!coyote_enabled) {
      data->sync_from_device();
    }
    for (const auto i : c10::irange(dsttensorvec.size())) {
	std::memcpy(dsttensorvec[i].data_ptr(), data->byte_array() + i * offset * dsttensor.element_size(), numel * dsttensor.element_size());
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
    }
    else{
      xrt_device = xrt::device(device_index);
    }
  }
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
    int segsize = 4096 * 1024;

    
    accl.get()->initialize(ranks_, rank_, 16, 1024, RDVZ_THRESHOLD, 4096*1024);
    
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

  // TODO free other buffer types
  if (!simulator_) {
      in_buf->free_buffer();
      out_buf->free_buffer();
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
#undef COLL_NAME
#define COLL_NAME Broadcast
void ProcessGroupACCL::run_broadcast(at::Tensor in_tensor,
                                     const BroadcastOptions &opts) {

  std::chrono::time_point<std::chrono::high_resolution_clock> start_inner  = std::chrono::high_resolution_clock::now();

  // This is very experimental
  #ifdef SIDESTEP_BCAST_WITH_ALLREDUCE
  // It seems to have issues with non-even numbers, so we round to ACCL_MSG_SIZE
  int rounded_count = (in_tensor.numel() + ROUND_NR) & ~ROUND_NR;
  
  int imaginary_count = rounded_count;
  if (in_tensor.scalar_type() == at::kDouble || in_tensor.scalar_type() == at::kLong){
      imaginary_count = (in_tensor.numel()*2 + ROUND_NR) & ~ROUND_NR;
  }

  auto zero_tensor = torch::zeros({imaginary_count}, at::kInt);
  if (opts.rootRank == rank_){
      init_input_tensor(in_tensor, in_buf, true, false, opts.rootRank);
  }
  else{
      init_input_tensor(zero_tensor, in_buf, false, true, opts.rootRank);
  }
  init_input_tensor(zero_tensor, out_buf, true, false, opts.rootRank);

  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  PRE_REQUEST(Broadcast, in_tensor);
  
  auto req = accl->allreduce(*in_buf, *out_buf, imaginary_count, ACCL::reduceFunction::SUM);      

  copy_back_tensor(in_tensor, out_buf, true, true);
  
  #else

  int rounded_count = (in_tensor.numel() + ROUND_NR) & ~ROUND_NR;
  
  if (opts.rootRank == rank_){
      init_input_tensor(in_tensor, in_buf, true, false, opts.rootRank);
  }

  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  PRE_REQUEST(Broadcast, in_tensor);
      
  auto req = accl->bcast(*in_buf, rounded_count, opts.rootRank);

  copy_back_tensor(in_tensor, in_buf, false, true, opts.rootRank);
  #endif
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::broadcast(std::vector<at::Tensor> &tensors,
                            const BroadcastOptions &opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
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
	at::Tensor &tensor = (entry->src)[0];
        // Segment data if necessary
        if (tensor.nbytes() > ACCL_MSG_SIZE) {
	  size_t non_zero_dim_count = tensor.numel() / tensor.size(0);
          size_t n = ACCL_MSG_SIZE / tensor.itemsize() / non_zero_dim_count;
	  ACCL::debug("[Broadcast] Segmenting tensor of size " + std::to_string(tensor.nbytes()) + " into " + std::to_string(n * non_zero_dim_count) + "-sized elements ");
          for (size_t i = 0; i < tensor.size(0); i += n) {
	    ACCL::debug("part " + std::to_string(i) + "!");
            size_t end = std::min(n, static_cast<size_t>(tensor.size(0)) - i);
            run_broadcast(tensor.narrow(0, i, end), opts);
          }
        } else {
          run_broadcast(tensor, opts);
        }
	}
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "accl::broadcast", OpType::BROADCAST,
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

#undef COLL_NAME
#define COLL_NAME Allreduce

void ProcessGroupACCL::run_allreduce(at::Tensor in_tensor,
                                     const AllreduceOptions &opts) {

  init_input_tensor(in_tensor, in_buf, true, true);

  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  PRE_REQUEST(Allreduce,in_tensor); 
  int rounded_count = (in_tensor.numel() + ROUND_NR) & ~ROUND_NR;
  
  auto req = accl->allreduce(*in_buf, *out_buf, rounded_count, acclOp.at(opts.reduceOp));      

  copy_back_tensor(in_tensor, out_buf, true, true);
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::allreduce(std::vector<at::Tensor> &tensors,
                            const AllreduceOptions &opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
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
	    auto tensor = (entry->src)[0];
	    // Segment data if necessary
	    if (tensor.nbytes() > (ACCL_MSG_SIZE)) {
		size_t non_zero_dim_count = tensor.numel() / tensor.size(0);
		size_t n = ACCL_MSG_SIZE / (tensor.itemsize() * non_zero_dim_count);
		ACCL::debug("[Allreduce] Segmenting tensor of size " + std::to_string(tensor.nbytes()) + " into " + std::to_string(n * non_zero_dim_count) + "-sized elements ");
		for (size_t i = 0; i < tensor.size(0); i += n) {
		    // ACCL::debug("part " + std::to_string(i) + "!");
		    size_t end = std::min(n, static_cast<size_t>(tensor.size(0)) - i);
		    run_allreduce(tensor.narrow(0, i, end), opts);
		}
	    } else {
		run_allreduce(tensor, opts);
	    }
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
#undef COLL_NAME
#define COLL_NAME Reduce
void ProcessGroupACCL::run_reduce(at::Tensor in_tensor,
                                  const ReduceOptions &opts) {


  init_input_tensor(in_tensor, in_buf, true, true);

  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  PRE_REQUEST(Reduce,in_tensor);

  int rounded_count = (in_tensor.numel() + ROUND_NR) & ~ROUND_NR;

  auto req = accl->reduce(*in_buf, *out_buf, rounded_count, opts.rootRank, acclOp.at(opts.reduceOp));

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
            size_t end = std::min(n, static_cast<size_t>(tensor.numel()) - i);
            run_reduce(tensor.narrow(0, i, end), opts);
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

#undef COLL_NAME
#define COLL_NAME Allgather
void ProcessGroupACCL::run_allgather(
    at::Tensor in_tensor,
    const std::vector<at::Tensor> &dsttensorvec) {
    
  init_input_tensor(in_tensor, in_buf, true, true);    
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

      
  PRE_REQUEST(Allgather,in_tensor);

  int rounded_count = (in_tensor.numel() + 1023) & ~1023;
      
  auto req = accl->allgather(*in_buf, *out_buf, rounded_count);

  copy_back_tensorvec(dsttensorvec, out_buf, in_tensor, in_tensor.numel(), rounded_count, true, true);
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
	  ACCL::debug("Starting AllGather");
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
                std::min(n, static_cast<size_t>(srctensor.size(0) - i));
            std::vector<at::Tensor> dsttensorslices;
            dsttensorslices.reserve(dsttensors.size());
            for (auto &dsttensor : dsttensors) {
              dsttensorslices.emplace_back(dsttensor.narrow(0, i, end));
            }
            run_allgather(srctensor.narrow(0, i, end), dsttensorslices);
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

#undef COLL_NAME
#define COLL_NAME Gather
    
void ProcessGroupACCL::run_gather(at::Tensor in_tensor,
                                  const std::vector<at::Tensor> &dsttensorvec,
                                  const GatherOptions &opts) {
  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  init_input_tensor(in_tensor, in_buf, true, true);

  PRE_REQUEST(Gather, in_tensor);

  auto req = accl->gather(*in_buf, *out_buf, in_tensor.numel(), opts.rootRank);

  copy_back_tensorvec(dsttensorvec, out_buf, in_tensor, in_tensor.numel(), in_tensor.numel(), true, false, opts.rootRank);
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
                std::min(n, static_cast<size_t>(srctensor.size(0)) - i);
            std::vector<at::Tensor> dsttensorslices;
            dsttensorslices.reserve(dsttensors.size());
            for (auto &dsttensor : dsttensors) {
              dsttensorslices.emplace_back(dsttensor.narrow(0, i, end));
            }
            run_gather(srctensor.narrow(0, i, end), dsttensorslices, opts);
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

#undef COLL_NAME
#define COLL_NAME Scatter
void ProcessGroupACCL::run_scatter(std::vector<at::Tensor> &in_tensor_vec,
                                   at::Tensor out_tensor,
                                   const ScatterOptions &opts) {
  // Reserve device
  c10::DeviceGuard guard(out_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);


  init_input_data_vec(in_tensor_vec, in_buf, out_tensor.options().device(c10::DeviceType::CPU), true, false, opts.rootRank);
  
  PRE_REQUEST(Scatter, out_tensor);
  
  auto req = accl->scatter(*in_buf, *out_buf, out_tensor.numel(), opts.rootRank);

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
	  size_t non_zero_dim_count = dsttensor.numel() / dsttensor.size(0);
          size_t n = bufsize / 4 / dsttensor.itemsize() / non_zero_dim_count;
          for (size_t i = 0; i < dsttensor.size(0); i += n) {
            ACCL::debug("part " + std::to_string(i) + "!");
            size_t end =
                std::min(n, static_cast<size_t>(dsttensor.size(0)) - i);
            std::vector<at::Tensor> srctensorslices;
            srctensorslices.reserve(srctensors.size());
            for (auto &srctensor : srctensors) {
              srctensorslices.emplace_back(srctensor.narrow(0, i, end));
            }
            run_scatter(srctensorslices, dsttensor.narrow(0, i, end), opts);
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

#undef COLL_NAME
#define COLL_NAME AlltoAll

void ProcessGroupACCL::run_alltoall(at::Tensor in_tensor,
                                    at::Tensor out_tensor,
                                    const AllToAllOptions &opts) {

  init_input_tensor(in_tensor, in_buf, true, true);

  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  // init_output_data(out_tensor, dstdata, out_tensor.numel(), out_tensor.scalar_type(), true, true);

  PRE_REQUEST(AlltoAll, in_tensor);

  auto req = accl->alltoall(*in_buf, *out_buf, in_tensor.numel()/size_);

  copy_back_tensor(out_tensor, out_buf, true, true);    
}

    
void ProcessGroupACCL::run_alltoall_vec(std::vector<at::Tensor> &in_tensor_vec,
                                    std::vector<at::Tensor> &out_tensor_vec,
                                    const AllToAllOptions &opts) {
  int a2a_nbytes = in_tensor_vec[0].nbytes();
  
  c10::DeviceGuard guard(in_tensor_vec[0].device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  init_input_data_vec(in_tensor_vec, in_buf, out_tensor_vec[0].options().device(c10::DeviceType::CPU), true, true);

  PRE_REQUEST(AlltoAll, in_tensor_vec[0]);

  auto req = accl->alltoall(*in_buf, *out_buf, in_tensor_vec[0].numel());

  copy_back_tensorvec(out_tensor_vec, out_buf, in_tensor_vec[0], in_tensor_vec[0].numel(), in_tensor_vec[0].numel(), true, true);
      
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
          auto srctensor = (entry->src)[0];
          auto dsttensor = (entry->dst)[0];


          // c10::DeviceGuard guard(srctensor.device());
          // std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          // Segment data if necessary
          if (dsttensor.nbytes() > bufsize) {

	    // Split individual entries
	    size_t non_zero_dim_count = dsttensor.numel() / dsttensor.size(0);
	    size_t n = bufsize / dsttensor.itemsize() / size_ / non_zero_dim_count;
	    size_t entry_size = dsttensor.numel() / size_ / non_zero_dim_count;
            for (size_t i = 0; i < entry_size; i += n) {
              ACCL::debug("part " + std::to_string(i) + "!");
              size_t end = std::min(n, static_cast<size_t>(entry_size) - i);

	      std::vector<at::Tensor> srctensorslices;
	      srctensorslices.reserve(size_);
	      for (int j = 0; j < size_; j++) {
		  int bufpos = j * entry_size;
		  srctensorslices.emplace_back(srctensor.narrow(0, i + bufpos, end));
	      }
	      std::vector<at::Tensor> dsttensorslices;
	      dsttensorslices.reserve(size_);
	      for (int j = 0; j < size_; j++) {
		  int bufpos = j * entry_size;
		  dsttensorslices.emplace_back(dsttensor.narrow(0, i + bufpos, end));
	      }
              run_alltoall_vec(srctensorslices, dsttensorslices, opts);
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
  ACCL::debug("ProcessGroupACCL does not support alltoall");
  TORCH_CHECK(false, "ProcessGroupACCL does not support alltoall");
}

#undef COLL_NAME
#define COLL_NAME Send

void ProcessGroupACCL::run_send(at::Tensor in_tensor, int dstRank,
                                int tag) {

  // Reserve device
  c10::DeviceGuard guard(in_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  init_input_tensor(in_tensor, in_buf, true, true);

  PRE_REQUEST(Send,in_tensor);
  
  ACCL::ACCLRequest* req = accl->send(*in_buf, in_tensor.numel(), dstRank, tag);

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
            size_t end = std::min(n, static_cast<size_t>(tensor.numel()) - i);
            run_send(tensor.narrow(0, i, end), dstRank, tag);
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

#undef COLL_NAME
#define COLL_NAME Recv    
    
void ProcessGroupACCL::run_recv(at::Tensor out_tensor, int srcRank,
                                int tag) {

  c10::DeviceGuard guard(out_tensor.device());
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  PRE_REQUEST(Receive, out_tensor);
  
  ACCL::ACCLRequest* req = accl->recv(*out_buf, out_tensor.numel(), srcRank, tag);

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
            size_t end = std::min(n, static_cast<size_t>(tensor.numel()) - i);
            run_recv(tensor.narrow(0, i, end), srcRank, tag);
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

#undef COLL_NAME
#define COLL_NAME Unnamed
    
c10::intrusive_ptr<Work>
ProcessGroupACCL::recvAnysource(std::vector<at::Tensor> &tensors, int tag) {
  TORCH_CHECK(false, "ProcessGroupACCL does not support recvAnysource");
}

c10::intrusive_ptr<Work>
ProcessGroupACCL::barrier(const BarrierOptions &opts) {
  accl->barrier();
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
