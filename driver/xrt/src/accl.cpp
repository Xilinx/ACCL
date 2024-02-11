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

#include <bitset>
#include <cmath>
#include <set>
#include <stdexcept>
#include "accl.hpp"
#include "accl/dummybuffer.hpp"

// 64 MB
#define NETWORK_BUF_SIZE (64 << 20)

namespace ACCL {
ACCL::ACCL(xrt::device &device, xrt::ip &cclo_ip, xrt::kernel &hostctrl_ip,
           int devicemem, const std::vector<int> &rxbufmem,
           const arithConfigMap &arith_config)
    : arith_config(arith_config), sim_mode(false),
      _devicemem(devicemem), rxbufmem(rxbufmem) {
  cclo = new FPGADevice(cclo_ip, hostctrl_ip, device);
}

// Simulation constructor
ACCL::ACCL(unsigned int sim_start_port, unsigned int local_rank,
           const arithConfigMap &arith_config)
    : arith_config(arith_config), sim_mode(true),
      _devicemem(0), rxbufmem({}) {
  cclo = new SimDevice(sim_start_port, local_rank);
}

// constructor for coyote fpga device
ACCL::ACCL(CoyoteDevice *dev, const arithConfigMap &arith_config)
  : arith_config(arith_config), sim_mode(false),
    _devicemem(0), rxbufmem(0), cclo(dev) {}

// destructor
ACCL::~ACCL() {
  deinit();
  delete cclo;
}

void ACCL::soft_reset() {
  debug("Doing a soft reset");

  CCLO::Options options{};
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::reset_periph;
  ACCLRequest *handle = call_async(options);
  std::chrono::milliseconds timeout(100);
  if(!wait(handle, timeout)){
    throw std::runtime_error("CCLO failed to soft reset");
  }
  check_return_value("reset_periph", handle);
}

void ACCL::deinit() {
  debug("Removing CCLO object at " + debug_hex(cclo->get_base_addr()));

  cclo->printDebug();

  soft_reset();

  for (auto &buf : eager_rx_buffers) {
    buf->free_buffer();
    delete buf;
  }
  eager_rx_buffers.clear();

  for (auto &buf : utility_spares) {
    buf->free_buffer();
    delete buf;
  }
  utility_spares.clear();
}

ACCLRequest *ACCL::set_timeout(unsigned int value, bool run_async,
                               std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};
  options.scenario = operation::config;
  options.count = value;
  options.cfg_function = cfgFunc::set_timeout;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("set_timeout", handle);
  }

  return handle;
}

ACCLRequest *ACCL::set_rendezvous_threshold(unsigned int value, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};
  options.scenario = operation::config;
  options.count = value;
  options.cfg_function = cfgFunc::set_max_eager_msg_size;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("set_max_eager_msg_size", handle);
  }

  return handle;
}

ACCLRequest *ACCL::nop(bool run_async, std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};
  options.scenario = operation::nop;
  options.comm = communicators[GLOBAL_COMM].communicators_addr();
  options.count = 0;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);
  if (!run_async) {
    wait(handle);
    check_return_value("nop", handle);
  }

  return handle;
}

ACCLRequest *ACCL::send(BaseBuffer &srcbuf, unsigned int count,
                        unsigned int dst, unsigned int tag, communicatorId comm_id,
                        bool from_fpga, dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  if (from_fpga == false) {
    srcbuf.sync_to_device();
  }

  options.scenario = operation::send;
  options.comm = communicators[comm_id].communicators_addr();
  options.addr_0 = &srcbuf;
  options.count = count;
  options.root_src_dst = dst;
  options.tag = tag;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("send", handle);
  }

  return handle;
}

ACCLRequest *ACCL::send(dataType src_data_type, unsigned int count,
                        unsigned int dst, unsigned int tag, communicatorId comm_id,
                        dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  options.scenario = operation::send;
  options.comm = communicators[comm_id].communicators_addr();
  options.data_type_io_0 = src_data_type;
  options.count = count;
  options.root_src_dst = dst;
  options.tag = tag;
  options.stream_flags = streamFlags::OP0_STREAM;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("send", handle);
  }

  return handle;
}

ACCLRequest *ACCL::stream_put(BaseBuffer &srcbuf, unsigned int count,
                        unsigned int dst, unsigned int stream_id, communicatorId comm_id,
                        bool from_fpga, dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  if (stream_id > 246) {
    throw std::invalid_argument("Stream ID must < 247");
  }

  if (from_fpga == false) {
    srcbuf.sync_to_device();
  }
  options.scenario = operation::send;
  options.comm = communicators[comm_id].communicators_addr();
  options.addr_0 = &srcbuf;
  options.count = count;
  options.root_src_dst = dst;
  options.tag = stream_id;
  options.stream_flags = streamFlags::RES_STREAM;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("stream_put", handle);
  }

  return handle;
}

ACCLRequest *ACCL::stream_put(dataType src_data_type, unsigned int count,
                        unsigned int dst, unsigned int stream_id, communicatorId comm_id,
                        dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  if (stream_id > 246) {
    throw std::invalid_argument("Stream ID must < 247");
  }

  options.scenario = operation::send;
  options.comm = communicators[comm_id].communicators_addr();
  options.data_type_io_0 = src_data_type;
  options.count = count;
  options.root_src_dst = dst;
  options.tag = stream_id;
  options.stream_flags = streamFlags::OP0_STREAM | streamFlags::RES_STREAM;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("stream_put", handle);
  }

  return handle;
}

ACCLRequest *ACCL::recv(BaseBuffer &dstbuf, unsigned int count,
                        unsigned int src, unsigned int tag, communicatorId comm_id,
                        bool to_fpga, dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  options.scenario = operation::recv;
  options.comm = communicators[comm_id].communicators_addr();
  options.addr_2 = &dstbuf;
  options.count = count;
  options.root_src_dst = src;
  options.tag = tag;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      dstbuf.sync_from_device();
    }
    check_return_value("recv", handle);
  }

  return handle;
}

ACCLRequest *ACCL::recv(dataType dst_data_type, unsigned int count,
                        unsigned int src, unsigned int tag, communicatorId comm_id,
                        dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  options.scenario = operation::recv;
  options.comm = communicators[comm_id].communicators_addr();
  options.data_type_io_0 = dst_data_type;
  options.count = count;
  options.root_src_dst = src;
  options.tag = tag;
  options.stream_flags = streamFlags::RES_STREAM;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("recv", handle);
  }

  return handle;
}

ACCLRequest *ACCL::copy(BaseBuffer *srcbuf, BaseBuffer *dstbuf, unsigned int count,
                        bool from_fpga, bool to_fpga, streamFlags stream_flags,
                        dataType data_type, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (from_fpga == false) {
    srcbuf->sync_to_device();
  }

  options.scenario = operation::copy;
  options.addr_0 = srcbuf;
  options.addr_2 = dstbuf;
  options.data_type_io_0 = data_type;
  options.data_type_io_2 = data_type;
  options.count = count;
  options.stream_flags = stream_flags;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      dstbuf->sync_from_device();
    }
    check_return_value("copy", handle);
  }

  return handle;
}

ACCLRequest *ACCL::copy(BaseBuffer &srcbuf, BaseBuffer &dstbuf, unsigned int count,
                        bool from_fpga, bool to_fpga, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  return copy(&srcbuf, &dstbuf, count,
                 from_fpga, to_fpga, streamFlags::NO_STREAM,
                 dataType::none, run_async, waitfor);
}

ACCLRequest *ACCL::copy_from_stream(BaseBuffer &dstbuf, unsigned int count,
                        bool to_fpga, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  return copy(nullptr, &dstbuf, count,
                 true, to_fpga, streamFlags::OP0_STREAM,
                 dstbuf.type(), run_async, waitfor);
}

ACCLRequest *ACCL::copy_to_stream(BaseBuffer &srcbuf, unsigned int count,
                        bool from_fpga, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  return copy(&srcbuf, nullptr, count,
                 from_fpga, true, streamFlags::RES_STREAM,
                 srcbuf.type(), run_async, waitfor);
}

ACCLRequest *ACCL::copy_from_to_stream(dataType data_type, unsigned int count,
                        bool run_async, std::vector<ACCLRequest *> waitfor) {
  return copy(nullptr, nullptr, count,
                 true, true, streamFlags::OP0_STREAM | streamFlags::RES_STREAM,
                 data_type, run_async, waitfor);
}

ACCLRequest *ACCL::combine(unsigned int count, reduceFunction function,
                           BaseBuffer &val1, BaseBuffer &val2, BaseBuffer &result,
                           bool val1_from_fpga, bool val2_from_fpga, bool to_fpga,
                           bool run_async, std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (val1_from_fpga == false) {
    val1.sync_to_device();
  }

  if (val2_from_fpga == false) {
    val2.sync_to_device();
  }

  options.scenario = operation::combine;
  options.addr_0 = &val1;
  options.addr_1 = &val2;
  options.addr_2 = &result;
  options.reduce_function = function;
  options.count = count;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      result.sync_from_device();
    }
    check_return_value("combine", handle);
  }

  return handle;
}

ACCLRequest *ACCL::bcast(BaseBuffer &buf, unsigned int count,
                         unsigned int root, communicatorId comm_id, bool from_fpga,
                         bool to_fpga, dataType compress_dtype, bool run_async,
                         std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  bool is_root = communicator.local_rank() == root;

  if (to_fpga == false && is_root == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (from_fpga == false && is_root == true) {
    buf.sync_to_device();
  }

  options.scenario = operation::bcast;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &buf;
  options.addr_2 = &buf;
  options.count = count;
  options.root_src_dst = root;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      buf.sync_from_device();
    }
    check_return_value("bcast", handle);
  }

  return handle;
}

ACCLRequest *ACCL::scatter(BaseBuffer &sendbuf,
                           BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                           communicatorId comm_id, bool from_fpga, bool to_fpga,
                           dataType compress_dtype, bool run_async,
                           std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  bool is_root = communicator.local_rank() == root;

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (from_fpga == false && is_root == true) {
    auto slice = sendbuf.slice(0, count * communicator.get_ranks()->size());
    slice->sync_to_device();
  }

  options.scenario = operation::scatter;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &sendbuf;
  options.addr_2 = &recvbuf;
  options.count = count;
  options.root_src_dst = root;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("scatter", handle);
  }

  return handle;
}

ACCLRequest *ACCL::gather(BaseBuffer &sendbuf,
                          BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                          communicatorId comm_id, bool from_fpga, bool to_fpga,
                          dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  bool is_root = communicator.local_rank() == root;

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (ignore_safety_checks == false &&
      (count + eager_rx_buffer_size - 1) / eager_rx_buffer_size *
              communicator.get_ranks()->size() >
          eager_rx_buffers.size()) {
    std::cerr << "ACCL: gather can't be executed safely with this number of "
                 "spare buffers"
              << std::endl;
  }

  if (from_fpga == false) {
    auto slice = sendbuf.slice(0, count);
    slice->sync_to_device();
  }

  options.scenario = operation::gather;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &sendbuf;
  options.addr_2 = &recvbuf;
  options.count = count;
  options.root_src_dst = root;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false && is_root == true) {
      auto slice = recvbuf.slice(0, count * communicator.get_ranks()->size());
      slice->sync_from_device();
    }
    check_return_value("gather", handle);
  }

  return handle;
}

ACCLRequest *ACCL::allgather(BaseBuffer &sendbuf,
                             BaseBuffer &recvbuf, unsigned int count,
                             communicatorId comm_id, bool from_fpga, bool to_fpga,
                             dataType compress_dtype, bool run_async,
                             std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (ignore_safety_checks == false &&
      (count + eager_rx_buffer_size - 1) / eager_rx_buffer_size *
              communicator.get_ranks()->size() >
          eager_rx_buffers.size()) {
    std::cerr << "ACCL: gather can't be executed safely with this number of "
                 "spare buffers"
              << std::endl;
  }

  if (from_fpga == false) {
    auto slice = sendbuf.slice(0, count);
    slice->sync_to_device();
  }

  options.scenario = operation::allgather;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &sendbuf;
  options.addr_1 = &recvbuf; // recvbuf is used with dm1 rd, needs to pass host infomation
  options.addr_2 = &recvbuf;
  options.count = count;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count * communicator.get_ranks()->size());
      slice->sync_from_device();
    }
    check_return_value("allgather", handle);
  }

  return handle;
}

ACCLRequest *ACCL::reduce(BaseBuffer &sendbuf,
                          BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                          reduceFunction func, communicatorId comm_id, bool from_fpga,
                          bool to_fpga, dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  bool is_root = communicator.local_rank() == root;

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (from_fpga == false) {
    auto slice = sendbuf.slice(0, count);
    slice->sync_to_device();
  }

  options.scenario = operation::reduce;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &sendbuf;
  options.addr_2 = &recvbuf;
  options.count = count;
  options.reduce_function = func;
  options.root_src_dst = root;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false && is_root == true) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("reduce", handle);
  }

  return handle;
}

ACCLRequest *ACCL::reduce(dataType src_data_type,
                          BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                          reduceFunction func, communicatorId comm_id,
                          bool to_fpga, dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  bool is_root = communicator.local_rank() == root;

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  options.scenario = operation::reduce;
  options.comm = communicator.communicators_addr();
  options.data_type_io_0 = src_data_type;
  options.addr_2 = &recvbuf;
  options.count = count;
  options.reduce_function = func;
  options.root_src_dst = root;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  options.stream_flags = streamFlags::OP0_STREAM;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false && is_root == true) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("reduce", handle);
  }

  return handle;
}

ACCLRequest *ACCL::reduce(BaseBuffer &sendbuf, dataType dst_data_type,
                          unsigned int count, unsigned int root,
                          reduceFunction func, communicatorId comm_id, bool from_fpga,
                          dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (from_fpga == false) {
    auto slice = sendbuf.slice(0, count);
    slice->sync_to_device();
  }

  options.scenario = operation::reduce;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &sendbuf;
  options.data_type_io_1 = dst_data_type;
  options.count = count;
  options.reduce_function = func;
  options.root_src_dst = root;
  options.compress_dtype = compress_dtype;
  options.stream_flags = streamFlags::RES_STREAM;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("reduce", handle);
  }

  return handle;
}

ACCLRequest *ACCL::reduce(dataType src_data_type, dataType dst_data_type,
                          unsigned int count, unsigned int root,
                          reduceFunction func, communicatorId comm_id,
                          dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  options.scenario = operation::reduce;
  options.comm = communicator.communicators_addr();
  options.data_type_io_0 = src_data_type;
  options.data_type_io_1 = dst_data_type;
  options.count = count;
  options.reduce_function = func;
  options.root_src_dst = root;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  options.stream_flags = streamFlags::OP0_STREAM | streamFlags::RES_STREAM;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    check_return_value("reduce", handle);
  }

  return handle;
}

ACCLRequest *ACCL::allreduce(BaseBuffer &sendbuf,
                             BaseBuffer &recvbuf, unsigned int count,
                             reduceFunction func, communicatorId comm_id,
                             bool from_fpga, bool to_fpga, dataType compress_dtype,
                             bool run_async, std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (from_fpga == false) {
    auto slice = sendbuf.slice(0, count);
    slice->sync_to_device();
  }

  options.scenario = operation::allreduce;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &sendbuf;
  options.addr_1 = &recvbuf; // recvbuf is used with dm1 rd, needs to pass host infomation
  options.addr_2 = &recvbuf;
  options.count = count;
  options.reduce_function = func;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("allreduce", handle);
  }

  return handle;
}

ACCLRequest *ACCL::reduce_scatter(BaseBuffer &sendbuf,
                                  BaseBuffer &recvbuf, unsigned int count,
                                  reduceFunction func, communicatorId comm_id,
                                  bool from_fpga, bool to_fpga,
                                  dataType compress_dtype, bool run_async,
                                  std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (from_fpga == false) {
    auto slice = sendbuf.slice(0, count * communicator.get_ranks()->size());
    slice->sync_to_device();
  }

  options.scenario = operation::reduce_scatter;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &sendbuf;
  options.addr_2 = &recvbuf;
  options.count = count;
  options.reduce_function = func;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("reduce_scatter", handle);
  }

  return handle;
}

ACCLRequest *ACCL::alltoall(BaseBuffer &sendbuf, BaseBuffer &recvbuf, unsigned int count,
                            communicatorId comm_id, bool from_fpga, bool to_fpga,
                            dataType compress_dtype, bool run_async, std::vector<ACCLRequest *> waitfor){
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (count == 0) {
    std::cerr << "ACCL: zero size buffer" << std::endl;
    return nullptr;
  }

  if (ignore_safety_checks == false &&
      (count + eager_rx_buffer_size - 1) / eager_rx_buffer_size *
              communicator.get_ranks()->size() >
          eager_rx_buffers.size()) {
    std::cerr << "ACCL: gather can't be executed safely with this number of "
                 "spare buffers"
              << std::endl;
  }

  if (from_fpga == false) {
    auto slice = sendbuf.slice(0, count * communicator.get_ranks()->size());
    slice->sync_to_device();
  }

  options.scenario = operation::alltoall;
  options.comm = communicator.communicators_addr();
  options.addr_0 = &sendbuf;
  options.addr_1 = &recvbuf; // recvbuf is used with dm1 rd, needs to pass host infomation
  options.addr_2 = &recvbuf;
  options.count = count;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (!run_async) {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count * communicator.get_ranks()->size());
      slice->sync_from_device();
    }
    check_return_value("alltoall", handle);
  }

  return handle;

}

ACCLRequest *ACCL::barrier(communicatorId comm_id,
                   std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];
  options.scenario = operation::barrier;
  options.comm = communicator.communicators_addr();
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  wait(handle);
  check_return_value("barrier", handle);
  return handle;
}

std::vector<rank_t> ACCL::get_comm_group(communicatorId comm_id) {
  communicators[comm_id].readback();
  return *communicators[comm_id].get_ranks();
}

unsigned int ACCL::get_comm_rank(communicatorId comm_id) {
  return communicators[comm_id].local_rank();
}

communicatorId ACCL::create_communicator(const std::vector<rank_t> &ranks,
                                            int local_rank) {
  configure_communicator(ranks, local_rank);
  // Communicator ID is the index of the communicator in the
  // vector of communicators
  communicatorId new_comm_id = communicators.size() - 1;
  return communicators.size() - 1;
}

std::string ACCL::dump_exchange_memory() {
  std::stringstream stream;
  stream << "exchange mem:" << std::hex << std::endl;
  size_t num_word_per_line = 4;
  for (size_t i = 0; i < EXCHANGE_MEM_ADDRESS_RANGE;
       i += 4 * num_word_per_line) {
    stream << "0x" << EXCHANGE_MEM_OFFSET_ADDRESS + i << ": [";
    for (size_t j = 0; j < num_word_per_line; ++j) {
      stream << "0x" << cclo->read(EXCHANGE_MEM_OFFSET_ADDRESS + i + j * 4);
      if (j != num_word_per_line - 1) {
        stream << ", ";
      }
    }
    stream << "]" << std::endl;
  }

  return stream.str();
}

std::string ACCL::dump_eager_rx_buffers(size_t n_egr_rx_bufs, bool dump_data) {
  std::stringstream stream;
  stream << "CCLO address: " << std::hex << cclo->get_base_addr() << std::endl;
  n_egr_rx_bufs = std::min(n_egr_rx_bufs, eager_rx_buffers.size());

  addr_t address = CCLO_ADDR::EGR_RX_BUF_SIZE_OFFSET;
  val_t maxsize = cclo->read(address);
  stream << "rx address: " << address << std::dec << std::endl;
  for (size_t i = 0; i < n_egr_rx_bufs; ++i) {
    address += 4;
    std::string status;
    switch (cclo->read(address)) {
    case 0:
      status = "NOT USED";
      break;
    case 1:
      status = "ENQUEUED";
      break;
    case 2:
      status = "RESERVED";
      break;
    default:
      status = "UNKNOWN";
      break;
    }

    address += 4;
    val_t addrl = cclo->read(address);
    address += 4;
    val_t addrh = cclo->read(address);
    address += 4;
    val_t rxtag = cclo->read(address);
    address += 4;
    val_t rxlen = cclo->read(address);
    address += 4;
    val_t rxsrc = cclo->read(address);
    address += 4;
    val_t seq = cclo->read(address);

    stream << "Spare RX Buffer " << i << ":\t address: 0x" << std::hex
           << addrh * (1UL << 32) + addrl << std::dec
           << " \t status: " << status << " \t occupancy: " << rxlen << "/"
           << maxsize << " \t MPI tag: " << std::hex << rxtag << std::dec
           << " \t seq: " << seq << " \t src: " << rxsrc;

    if(dump_data) {
      eager_rx_buffers[i]->sync_from_device();

      stream << " \t data: " << std::hex << "[";
      for (size_t j = 0; j < eager_rx_buffers[i]->size(); ++j) {
        stream << "0x"
              << static_cast<uint16_t>(static_cast<uint8_t *>(
                      eager_rx_buffers[i]->byte_array())[j]);
        if (j != eager_rx_buffers[i]->size() - 1) {
          stream << ", ";
        }
      }
      stream << "]" << std::dec << std::endl;
    } else {
      stream << std::endl;
    }

  }

  return stream.str();
}

void ACCL::parse_hwid(){
  unsigned int hwid = get_hwid();
  debug("CCLO HWID: " + std::to_string(hwid) + " at 0x" + debug_hex(cclo->get_base_addr()));
  //   set idcode [expr { $externalDMA<<7 | ($debugLevel > 0)<<6 | $enableExtKrnlStream<<5 | $enableCompression<<4 | $enableArithmetic<<3 | $enableDMA<<2 | ($netStackType == "RDMA" ? 2 : $netStackType == "TCP" ? 1 : 0) }]
  debug("CCLO source commit (first 24b): " + debug_hex((hwid >> 8) & 0xffffff));
  debug("CCLO Capabilities:");
  unsigned int stype = hwid & 0x3;
  debug("Stack type: " + std::string((stype == 0) ? "UDP" : (stype == 1) ? "TCP" : (stype == 2) ? "RDMA" : "Unrecognized"));
  debug("Internal DMA:" + std::string(((hwid >> 2) & 0x1) ? "True" : "False"));
  debug("External DMA:" + std::string(((hwid >> 7) & 0x1) ? "True" : "False"));
  debug("Reduction:" + std::string(((hwid >> 3) & 0x1) ? "True" : "False"));
  debug("Compression:" + std::string(((hwid >> 4) & 0x1) ? "True" : "False"));
  debug("Kernel Streams:" + std::string(((hwid >> 5) & 0x1) ? "True" : "False"));
  debug("Debug:" + std::string(((hwid >> 6) & 0x1) ? "True" : "False"));
}

void ACCL::initialize(const std::vector<rank_t> &ranks, int local_rank,
                           int n_egr_rx_bufs, addr_t egr_rx_buf_size,
                           addr_t max_egr_size, addr_t max_rndzv_size) {

  parse_hwid();

  soft_reset();

  if (cclo->read(CCLO_ADDR::CFGRDY_OFFSET) != 0) {
    throw std::runtime_error("CCLO appears configured, might be in use. Please "
                             "reset the CCLO and retry");
  }

  debug("Configuring Eager RX Buffers");
  setup_eager_rx_buffers(n_egr_rx_bufs, egr_rx_buf_size, rxbufmem);

  debug("Configuring Rendezvous Spare Buffers");
  setup_rendezvous_spare_buffers(max_rndzv_size, rxbufmem);

  debug("Configuring a communicator");
  configure_communicator(ranks, local_rank);

  debug("Configuring arithmetic");
  configure_arithmetic();

  debug("Configuring collective tuning parameters");
  configure_tuning_parameters();

  // Mark CCLO as configured
  debug("CCLO configured");
  cclo->write(CCLO_ADDR::CFGRDY_OFFSET, 1);

  debug("Set timeout");
  set_timeout(1000000);

  debug("Set max eager size: " + std::to_string(max_egr_size));
  set_max_eager_msg_size(max_egr_size);
  debug("Set max rendezvous reduce size: " + std::to_string(max_rndzv_size));
  set_max_rendezvous_msg_size(max_rndzv_size);

  CCLO::Options options{};
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::enable_pkt;
  call_sync(options);

  config_rdy = true;

  debug("Accelerator ready!");
}

void ACCL::configure_arithmetic() {
  if (communicators.size() < 1) {
    throw new std::runtime_error("Communicators unconfigured, please call "
                                 "configure_communicator() first");
  }

  for (auto &[_key, arithcfg] : arith_config) {
    write_arithconfig(*cclo, arithcfg, &current_config_address);
  }
}

addr_t ACCL::get_arithmetic_config_addr(std::pair<dataType, dataType> id) {
  return arith_config.at(id).addr();
}

void ACCL::setup_eager_rx_buffers(size_t n_egr_rx_bufs, addr_t egr_rx_buf_size,
                            const std::vector<int> &devicemem) {
  addr_t address = CCLO_ADDR::EGR_RX_BUF_SIZE_OFFSET;
  eager_rx_buffer_size = egr_rx_buf_size;
  for (size_t i = 0; i < n_egr_rx_bufs; ++i) {
    // create, clear and sync buffers to device
    Buffer<int8_t> *buf;

    if (sim_mode) {
      buf = new SimBuffer(new int8_t[eager_rx_buffer_size](), eager_rx_buffer_size, dataType::int8,
                          static_cast<SimDevice *>(cclo)->get_context());
    } else if(cclo->get_device_type() == CCLO::xrt_device ){
      buf = new FPGABuffer<int8_t>(eager_rx_buffer_size, dataType::int8, *(static_cast<FPGADevice *>(cclo)->get_device()), devicemem[i % devicemem.size()]);
    } else if(cclo->get_device_type() == CCLO::coyote_device){
      buf = new CoyoteBuffer<int8_t>(eager_rx_buffer_size, dataType::int8, static_cast<CoyoteDevice *>(cclo));
    }

    buf->sync_to_device();
    eager_rx_buffers.emplace_back(buf);
    // program this buffer into the accelerator
    address += 4;
    cclo->write(address, 0);
    address += 4;
    cclo->write(address, buf->address() & 0xffffffff);
    address += 4;
    cclo->write(address, (buf->address() >> 32) & 0xffffffff);
    // clear remaining 4 fields
    for (size_t j = 0; j < 4; ++j) {
      address += 4;
      cclo->write(address, 0);
    }
  }

  //write buffer len
  cclo->write(CCLO_ADDR::EGR_RX_BUF_SIZE_OFFSET, eager_rx_buffer_size);

  // NOTE: the buffer count HAS to be written last (offload checks for this)
  cclo->write(CCLO_ADDR::NUM_EGR_RX_BUFS_OFFSET, n_egr_rx_bufs);

  current_config_address = address + 4;

}

void ACCL::setup_rendezvous_spare_buffers(addr_t rndzv_spare_buf_size, const std::vector<int> &devicemem){
  max_rndzv_msg_size = rndzv_spare_buf_size;
  for(int i=0; i<3; i++){
    Buffer<int8_t> *buf;
    if (sim_mode) {
      buf = new SimBuffer(new int8_t[max_rndzv_msg_size](), max_rndzv_msg_size, dataType::int8,
                        static_cast<SimDevice *>(cclo)->get_context());
    } else if(cclo->get_device_type() == CCLO::xrt_device ){
      buf = new FPGABuffer<int8_t>(max_rndzv_msg_size, dataType::int8,
                        *(static_cast<FPGADevice *>(cclo)->get_device()), devicemem[i % devicemem.size()]);
    } else if(cclo->get_device_type() == CCLO::coyote_device){
      buf = new CoyoteBuffer<int8_t>(max_rndzv_msg_size, dataType::int8, static_cast<CoyoteDevice *>(cclo));
    }
    buf->sync_to_device();
    utility_spares.emplace_back(buf);
  }
  cclo->write(CCLO_ADDR::SPARE1_OFFSET, utility_spares.at(0)->address() & 0xffffffff);
  cclo->write(CCLO_ADDR::SPARE1_OFFSET+4, (utility_spares.at(0)->address() >> 32) & 0xffffffff);
  cclo->write(CCLO_ADDR::SPARE2_OFFSET, utility_spares.at(1)->address() & 0xffffffff);
  cclo->write(CCLO_ADDR::SPARE2_OFFSET+4, (utility_spares.at(1)->address() >> 32) & 0xffffffff);
  cclo->write(CCLO_ADDR::SPARE3_OFFSET, utility_spares.at(2)->address() & 0xffffffff);
  cclo->write(CCLO_ADDR::SPARE3_OFFSET+4, (utility_spares.at(2)->address() >> 32) & 0xffffffff);
}

void ACCL::configure_tuning_parameters(){
  //tune gather to reduce fan-in of flat tree above certain message sizes
  cclo->write(CCLO_ADDR::GATHER_FLAT_TREE_MAX_FANIN_OFFSET, 2);
  cclo->write(CCLO_ADDR::GATHER_FLAT_TREE_MAX_COUNT_OFFSET, 32*1024);
  //tune bcast to execute flat tree up to 3 ranks
  cclo->write(CCLO_ADDR::BCAST_FLAT_TREE_MAX_RANKS_OFFSET, 3);
  //tune reduce to execute flat tree up to 4 ranks or up to 32KB
  unsigned int max_reduce_flat_tree_size = 4;
  cclo->write(CCLO_ADDR::REDUCE_FLAT_TREE_MAX_RANKS_OFFSET, max_reduce_flat_tree_size);
  cclo->write(CCLO_ADDR::REDUCE_FLAT_TREE_MAX_COUNT_OFFSET, std::min(max_rndzv_msg_size/max_reduce_flat_tree_size, (long unsigned int)32*1024));
}

void ACCL::check_return_value(const std::string function_name, ACCLRequest *request) {
  val_t retcode = cclo->get_retcode(request);
  if (retcode != 0) {
    std::stringstream stream;
    const std::bitset<error_code_bits> retcode_bitset{retcode};
    stream << "CCLO @0x" << std::hex << cclo->get_base_addr() << std::dec
           << ": during " << function_name
           << " the following error(s) occured: ";

    bool first = true;
    for (size_t i = 0; i < retcode_bitset.size(); ++i) {
      if (retcode_bitset[i]) {
        if (first) {
          first = false;
        } else {
          stream << ", ";
        }
        stream << error_code_to_string(static_cast<errorCode>(1 << i));
      }
    }

    stream << " (" << retcode_bitset << ")";
    throw std::runtime_error(stream.str());
  }
}

void ACCL::prepare_call(CCLO::Options &options) {
  const ArithConfig *arithcfg;

  std::set<dataType> dtypes;
  // set flags for host-only buffers
  options.host_flags = hostFlags::NO_HOST;

  if (options.addr_0 == nullptr) {
    options.addr_0 = &dummy_buffer;
    dtypes.insert(options.data_type_io_0);
  }
  else {
    dtypes.insert(options.addr_0->type());
    if(options.addr_0->is_host_only()) options.host_flags |= hostFlags::OP0_HOST;
  }

  if (options.addr_1 == nullptr) {
    options.addr_1 = &dummy_buffer;
    dtypes.insert(options.data_type_io_1);
  }
  else {
    dtypes.insert(options.addr_1->type());
    if(options.addr_1->is_host_only()) options.host_flags |= hostFlags::OP1_HOST;
  }

  if (options.addr_2 == nullptr) {
    options.addr_2 = &dummy_buffer;
    dtypes.insert(options.data_type_io_2);
  }
  else {
    dtypes.insert(options.addr_2->type());
    if(options.addr_2->is_host_only()) options.host_flags |= hostFlags::RES_HOST;
  }

  dtypes.erase(dataType::none);

  // if no compressed data type specified, set same as uncompressed
  options.compression_flags = compressionFlags::NO_COMPRESSION;

  if (dtypes.empty()) {
    options.arithcfg_addr = 0x0;
    return;
  }

  if (options.compress_dtype == dataType::none) {
    // no ethernet compression
    if (dtypes.size() == 1) {
      // no operand compression
      dataType dtype = *dtypes.begin();
      std::pair<dataType, dataType> key = {dtype, dtype};
      //barrier is a corner case, we don't care about the dtype
      if(options.scenario == operation::barrier) {
        arithcfg = &this->arith_config.begin()->second;
      } else {
        arithcfg = &this->arith_config.at(key);
      }
    } else {
      // with operand compression
      // determine compression dtype
      std::set<dataType>::iterator it = dtypes.begin();
      dataType dtype1 = *it;
      dataType dtype2 = *std::next(it);
      dataType compressed, uncompressed;

      if (dataTypeSize.at(dtype1) < dataTypeSize.at(dtype2)) {
        compressed = dtype1;
        uncompressed = dtype2;
      } else {
        compressed = dtype2;
        uncompressed = dtype1;
      }

      std::pair<dataType, dataType> key = {uncompressed, compressed};
      arithcfg = &this->arith_config.at(key);
      // determine which operand is compressed
      if (options.addr_0->type() == compressed) {
        options.compression_flags |= compressionFlags::OP0_COMPRESSED;
      }
      if (options.addr_1->type() == compressed) {
        options.compression_flags |= compressionFlags::OP1_COMPRESSED;
      }
      if (options.addr_2->type() == compressed) {
        options.compression_flags |= compressionFlags::RES_COMPRESSED;
      }
    }
  } else {
    // with ethernet compression
    options.compression_flags |= compressionFlags::ETH_COMPRESSED;
    if (dtypes.size() == 1) {
      // no operand compression
      dataType dtype = *dtypes.begin();
      std::pair<dataType, dataType> key = {dtype, options.compress_dtype};
      arithcfg = &this->arith_config.at(key);
    } else {
      // with operand compression
      if (dtypes.count(options.compress_dtype) == 0) {
        throw std::runtime_error("Unsupported data type combination");
      }
      dtypes.erase(options.compress_dtype);

      // determine compression dtype
      dataType uncompressed = *dtypes.begin();

      std::pair<dataType, dataType> key = {uncompressed,
                                           options.compress_dtype};
      arithcfg = &this->arith_config.at(key);
      // determine which operand is compressed
      if (options.addr_0->type() == options.compress_dtype) {
        options.compression_flags |= compressionFlags::OP0_COMPRESSED;
      }
      if (options.addr_1->type() == options.compress_dtype) {
        options.compression_flags |= compressionFlags::OP1_COMPRESSED;
      }
      if (options.addr_2->type() == options.compress_dtype) {
        options.compression_flags |= compressionFlags::RES_COMPRESSED;
      }
    }
  }

  options.arithcfg_addr = arithcfg->addr();
}

// Request handling
void ACCL::wait(ACCLRequest *request) {
  return cclo->wait(request);
}

bool ACCL::wait(ACCLRequest *request, std::chrono::milliseconds timeout) {
  return (cclo->wait(request, timeout) == timeoutStatus::no_timeout);
}

bool ACCL::test(ACCLRequest *request) {
  return cclo->test(request);
}

uint64_t ACCL::get_duration(ACCLRequest *request) {
  return cclo->get_duration(request);
}

void ACCL::free_request(ACCLRequest *request) {
  return cclo->free_request(request);
}

ACCLRequest *ACCL::call_async(CCLO::Options &options) {
  if (!config_rdy && options.scenario != operation::config) {
    throw std::runtime_error("CCLO not configured, cannot call. Please make sure that you are invoking initialize().");
  }

  prepare_call(options);
  ACCLRequest *req = cclo->start(options);
  return req;
}

ACCLRequest *ACCL::call_sync(CCLO::Options &options) {
  if (!config_rdy && options.scenario != operation::config) {
    throw std::runtime_error("CCLO not configured, cannot call. Please make sure that you are invoking initialize().");
  }

  prepare_call(options);
  ACCLRequest *req = cclo->call(options);
  return req;
}

void ACCL::set_max_eager_msg_size(unsigned int value) {
  CCLO::Options options{};
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::set_max_eager_msg_size;
  options.count = value;
  ACCLRequest *handle = call_sync(options);
  max_eager_msg_size = value;
  check_return_value("set_max_eager_msg_size", handle);
}

void ACCL::set_max_rendezvous_msg_size(unsigned int value) {
  CCLO::Options options{};
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::set_max_rendezvous_msg_size;
  options.count = value;
  ACCLRequest *handle = call_sync(options);
  max_eager_msg_size = value;
  check_return_value("set_max_rendezvous_msg_size", handle);
}

void ACCL::configure_communicator(const std::vector<rank_t> &ranks,
                                            int local_rank) {
  if (eager_rx_buffers.empty()) {
    throw std::runtime_error(
        "RX buffers unconfigured, please call setup_eager_rx_buffers() first.");
  }
  communicators.emplace_back(
      Communicator(cclo, ranks, local_rank, &current_config_address));
}

std::string ACCL::dump_communicator() {
  std::stringstream stream;
  for (size_t i = 0; i < communicators.size(); ++i) {
    stream << "Communicator " << i << " (0x" << std::hex
           << communicators[i].communicators_addr() << std::dec << ")"
           << ":" << std::endl
           << communicators[i].dump();
  }

  return stream.str();
}

addr_t ACCL::get_communicator_addr(communicatorId comm_id){
  return communicators[comm_id].communicators_addr();
}

} // namespace ACCL
