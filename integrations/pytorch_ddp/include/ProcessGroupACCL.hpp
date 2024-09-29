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

// This process group is adapted from the ProcessGroupMPI.

#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <experimental/xrt_ip.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <accl.hpp>
#include <accl_network_utils.hpp>

namespace c10d {

constexpr const char *ACCL_BACKEND_NAME = "accl";

// WorkEntry is the state associated with a single CCLO run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(std::vector<at::Tensor> *srcPtr,
                     std::vector<at::Tensor> *dstPtr,
                     std::function<void(std::unique_ptr<WorkEntry> &)> run)
      : dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()), run(std::move(run)) {
    if (srcPtr) {
      src = *srcPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry &) = delete;
  // Not copy assignable
  WorkEntry &operator=(const WorkEntry &) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;

  // Copy of user provided outputs.
  const std::vector<at::Tensor> dst;

  // src rank returned, for recv only
  int *srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry> &)> run;
};

// ProcessGroupACCL implements ACCL bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All ACCL functions provided by this class is asynchronously scheduled on a
// Worker thread. That is, The process may be multi-threaded, but all ACCL calls
// are serialized.
//
// ProcessGroupACCL only supports a singe process group. In other words, no more
// than 1 process group can be created globally.
//
// Also note that ProcessGroupACCL only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.

class TORCH_API ProcessGroupACCL : public ProcessGroup {
public:

  class WorkACCL : public Work {
  public:
    explicit WorkACCL(std::vector<at::Tensor> outputTensors,
                      const char *profilingTitle = nullptr,
                      OpType optype = OpType::UNKNOWN,
                      const c10::optional<std::vector<at::Tensor>>
                          &inputTensors = c10::nullopt)
        : Work(rank_, optype, profilingTitle, inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(c10::make_intrusive<at::ivalue::Future>(
              c10::ListType::create(c10::TensorType::get()))) {}

    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    int sourceRank() const override { return 0; }

  protected:
    friend class ProcessGroupACCL;

  private:
    void finishWorkACCL();
    void finishWorkACCLError(std::exception_ptr eptr);

    std::vector<at::Tensor> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  struct TORCH_API Rank : torch::CustomClassHolder {
    explicit Rank(std::string ip, int port, int session_id,
                  uint64_t max_segment_size)
        : ip(ip), port(port), session_id(session_id),
          max_segment_size(max_segment_size) {}

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Rank> create(std::string ip, int port,
                                           int session_id,
                                           uint64_t max_segment_size) {
      return c10::make_intrusive<Rank>(ip, port, session_id, max_segment_size);
    }

    const std::string ip;
    const int port;
    const int session_id;
    const uint64_t max_segment_size;
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupACCL(
      const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
      std::vector<c10::intrusive_ptr<ProcessGroupACCL::Rank>> &ranks,
      bool simulator, bool p2p_enabled,
      const std::map<ACCL::dataType, ACCL::dataType> &compression_enabled,
      const std::vector<int> &profiling_ranks, double profiling_timeout,
      const std::string &xclbin, accl_network_utils::acclDesign design,
      int device_index = 0, int nbufs = 16, uint64_t bufsize = 1024,
      bool rsfec = false);

  void initialize();

  std::vector<std::uint8_t> get_local_qp(unsigned int rank);
  void set_remote_qp(unsigned int rank, std::vector<std::uint8_t> &qp);

  virtual ~ProcessGroupACCL();

  // Abort the ACCL program, needs to be called when exception is detected
  void abort();

  const std::string getBackendName() const override {
    return std::string(ACCL_BACKEND_NAME);
  }

  c10::intrusive_ptr<Work>
  broadcast(std::vector<at::Tensor> &data,
            const BroadcastOptions &opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work>
  allreduce(std::vector<at::Tensor> &tensors,
            const AllreduceOptions &opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work>
  allreduce_coalesced(std::vector<at::Tensor> &tensors,
                      const AllreduceCoalescedOptions &opts =
                          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work>
  reduce(std::vector<at::Tensor> &tensors,
         const ReduceOptions &opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work>
  allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
            std::vector<at::Tensor> &inputTensors,
            const AllgatherOptions &opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work>
  _allgather_base(at::Tensor &outputbuffer, at::Tensor &inputbuffer,
                  const AllgatherOptions &opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>> &outputTensorLists,
      std::vector<at::Tensor> &inputTensors,
      const AllgatherOptions &opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work>
  gather(std::vector<std::vector<at::Tensor>> &outputTensors,
         std::vector<at::Tensor> &inputTensors,
         const GatherOptions &opts = GatherOptions()) override;

  c10::intrusive_ptr<Work>
  scatter(std::vector<at::Tensor> &outputTensors,
          std::vector<std::vector<at::Tensor>> &inputTensors,
          const ScatterOptions &opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor> &outputTensors,
      std::vector<std::vector<at::Tensor>> &inputTensors,
      const ReduceScatterOptions &opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work>
  alltoall_base(at::Tensor &outputTensor, at::Tensor &inputTensor,
                std::vector<int64_t> &outputSplitSizes,
                std::vector<int64_t> &inputSplitSizes,
                const AllToAllOptions &opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work>
  alltoall(std::vector<at::Tensor> &outputTensors,
           std::vector<at::Tensor> &inputTensors,
           const AllToAllOptions &opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors,
                                              int dstRank, int tag) override;

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors,
                                              int srcRank, int tag) override;

  c10::intrusive_ptr<Work>
  recvAnysource(std::vector<at::Tensor> &tensor, int tag) override;

  c10::intrusive_ptr<Work>
  barrier(const BarrierOptions &opts = BarrierOptions()) override;

  std::string toString();

  const std::map<ACCL::dataType, ACCL::dataType> &get_compression() const {
    return compression;
  }

  void
  set_compression(const std::map<ACCL::dataType, ACCL::dataType> &compression) {
    this->compression = compression;
  }

protected:
  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr<WorkACCL>>;
  // Worker thread loop
  void runLoop();
  // Helper function that is called by the destructor
  void destroy();

  c10::intrusive_ptr<Work>
  enqueue(std::unique_ptr<WorkEntry> entry,
          const char *profilingTitle = nullptr, OpType optype = OpType::UNKNOWN,
          const c10::optional<std::vector<at::Tensor>> &inputTensors =
              c10::nullopt);

  void run_send(at::Tensor tensor, int dstRank, int tag);
  void run_recv(at::Tensor tensor, int rcvRank, int tag);
  void run_broadcast(at::Tensor in_tensor, const BroadcastOptions &opts);
  void run_allreduce(at::Tensor in_tensor, const AllreduceOptions &opts);
  void run_reduce(at::Tensor in_tensor, const ReduceOptions &opts);
  void run_allgather(at::Tensor in_tensor,
                     const std::vector<at::Tensor> &dsttensors);
  void run_gather(at::Tensor in_tensor,
                  const std::vector<at::Tensor> &dsttensors,
                  const GatherOptions &opts);
  void run_scatter(std::vector<at::Tensor> &in_tensors, at::Tensor dsttensor,
                   const ScatterOptions &opts);

  void run_alltoall(at::Tensor in_tensor, at::Tensor dsttensor, const AllToAllOptions &opts);
  
  void run_alltoall_vec(std::vector<at::Tensor> &in_tensor_vec,
                                    std::vector<at::Tensor> &out_tensor_vec, const AllToAllOptions &opts);

  ACCL::dataType get_compressed_type(c10::ScalarType datatype);

  bool stop_;

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  // Global states
  static void initACCLOnce();
  static void acclExit();
  
  void init_input_tensor(at::Tensor &tensor, std::unique_ptr<ACCL::BaseBuffer> &data, bool do_on_root, bool do_on_others, int opts_root_rank = 0);
  
  void init_input_tensor_new(at::Tensor &tensor, ACCL::BaseBuffer *data, bool do_on_root, bool do_on_others, int opts_root_rank = 0);
  
  void init_input_data_vec(std::vector<at::Tensor> &tensor_vec, std::unique_ptr<ACCL::BaseBuffer> &data, const at::TensorOptions &options, bool do_on_root, bool do_on_others, int opts_root_rank = 0);
  
  void copy_back_tensor(at::Tensor tensor_original, std::unique_ptr<ACCL::BaseBuffer> &data, bool do_on_root, bool do_on_others, int opts_root_rank = 0);

  void copy_back_tensorvec(const std::vector<at::Tensor> &dsttensorvec, std::unique_ptr<ACCL::BaseBuffer> &data, at::Tensor &dsttensor, int numel, int offset, bool do_on_root, bool do_on_others, int opts_root_rank = 0);
  
  static std::once_flag onceFlagInitACCL;

  static std::mutex pgGlobalMutex_;

private:
  c10::intrusive_ptr<Store> store_;
  std::vector<ACCL::rank_t> ranks_;
  accl_network_utils::acclDesign design_;
  int device_index_;
  int nbufs_;
  uint64_t bufsize_;
  bool rsfec_;
  bool simulator_;
  const std::string xclbin_;

  ACCL::CoyoteDevice *cyt_device;
  std::vector<fpga::ibvQpConn*> ibvQpConn_vec;
  xrt::device xrt_device;

  std::unique_ptr<ACCL::ACCL> accl;
  uint64_t bufsize;
  bool p2p_enabled;
  bool coyote_enabled;
  std::map<ACCL::dataType, ACCL::dataType> compression;
  bool initialized;
  xrt::bo buf0;
  xrt::bo buf1;

  std::unique_ptr<ACCL::BaseBuffer> in_buf;
  std::unique_ptr<ACCL::BaseBuffer> out_buf;
};

} // namespace c10d
