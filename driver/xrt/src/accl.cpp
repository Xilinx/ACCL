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
ACCL::ACCL(const std::vector<rank_t> &ranks, int local_rank,
           xrt::device &device, xrt::ip &cclo_ip, xrt::kernel &hostctrl_ip,
           int devicemem, const std::vector<int> &rxbufmem,
           networkProtocol protocol, int nbufs, addr_t bufsize, addr_t segsize,
           const arithConfigMap &arith_config)
    : arith_config(arith_config), protocol(protocol), sim_mode(false),
      _devicemem(devicemem), rxbufmem(rxbufmem), device(device) {
  cclo = new FPGADevice(cclo_ip, hostctrl_ip);
  initialize_accl(ranks, local_rank, nbufs, bufsize, segsize);
}

// Simulation constructor
ACCL::ACCL(const std::vector<rank_t> &ranks, int local_rank,
           unsigned int sim_start_port, networkProtocol protocol, int nbufs,
           addr_t bufsize, addr_t segsize, const arithConfigMap &arith_config)
    : arith_config(arith_config), protocol(protocol), sim_mode(true),
      _devicemem(0), rxbufmem({}) {
  cclo = new SimDevice(sim_start_port, local_rank);
  debug("initialize_accl");
  initialize_accl(ranks, local_rank, nbufs, bufsize, segsize);
}

ACCL::ACCL(const std::vector<rank_t> &ranks, int local_rank,
           unsigned int sim_start_port, xrt::device &device,
           networkProtocol protocol, int nbufs, addr_t bufsize, addr_t segsize,
           const arithConfigMap &arith_config)
    : arith_config(arith_config), protocol(protocol), sim_mode(true),
      _devicemem(0), rxbufmem({}), device(device) {
  cclo = new SimDevice(sim_start_port, local_rank);
  initialize_accl(ranks, local_rank, nbufs, bufsize, segsize);
}

ACCL::~ACCL() {
  deinit();
  delete cclo;
}

void ACCL::deinit() {
  debug("Removing CCLO object at " + debug_hex(cclo->get_base_addr()));

  CCLO::Options options{};
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::reset_periph;
  call_sync(options, false);

  for (auto &buf : rx_buffer_spares) {
    buf->free_buffer();
    delete buf;
  }
  rx_buffer_spares.clear();

  if (utility_spare != nullptr) {
    utility_spare->free_buffer();
    delete utility_spare;
    utility_spare = nullptr;
  }
}

ACCLRequest *ACCL::set_timeout(unsigned int value, bool run_async,
                               std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};
  options.scenario = operation::config;
  options.count = value;
  options.cfg_function = cfgFunc::set_timeout;
  ACCLRequest *handle = call_async(options, false);

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("set_timeout", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::nop(bool run_async, std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};
  options.scenario = operation::nop;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);
  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("nop", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::send(BaseBuffer &srcbuf, unsigned int count,
                        unsigned int dst, unsigned int tag, communicatorId comm_id,
                        bool from_fpga, dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  if (from_fpga == false) {
    srcbuf.sync_to_device();
  }

  unsigned count_bytes = count * (dataTypeSize.at(srcbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Send is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Send does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("send", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::send(dataType src_data_type, unsigned int count,
                        unsigned int dst, unsigned int tag, communicatorId comm_id,
                        dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(src_data_type) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Send is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Send does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("send", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::stream_put(BaseBuffer &srcbuf, unsigned int count,
                        unsigned int dst, unsigned int stream_id, communicatorId comm_id,
                        bool from_fpga, dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(srcbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Stream put is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Stream put does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("stream_put", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::stream_put(dataType src_data_type, unsigned int count,
                        unsigned int dst, unsigned int stream_id, communicatorId comm_id,
                        dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(src_data_type) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Stream put is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Stream put does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("stream_put", handle);
  }

  return nullptr;
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

  unsigned count_bytes = count * (dataTypeSize.at(dstbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Recv is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Recv does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false) {
      dstbuf.sync_from_device();
    }
    check_return_value("recv", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::recv(dataType dst_data_type, unsigned int count,
                        unsigned int src, unsigned int tag, communicatorId comm_id,
                        dataType compress_dtype, bool run_async,
                        std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(dst_data_type) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Recv is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Recv does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("recv", handle);
  }

  return nullptr;
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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false) {
      dstbuf->sync_from_device();
    }
    check_return_value("copy", handle);
  }

  return nullptr;
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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false) {
      result.sync_from_device();
    }
    check_return_value("combine", handle);
  }

  return nullptr;
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
  options.count = count;
  options.root_src_dst = root;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false) {
      buf.sync_from_device();
    }
    check_return_value("bcast", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::scatter(BaseBuffer &sendbuf,
                           BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                           communicatorId comm_id, bool from_fpga, bool to_fpga,
                           dataType compress_dtype, bool run_async,
                           std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(sendbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Scatter is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Scatter does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("scatter", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::gather(BaseBuffer &sendbuf,
                          BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                          communicatorId comm_id, bool from_fpga, bool to_fpga,
                          dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(sendbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Gather is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Gather does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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
      (count + segment_size - 1) / segment_size *
              communicator.get_ranks()->size() >
          rx_buffer_spares.size()) {
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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false && is_root == true) {
      auto slice = recvbuf.slice(0, count * communicator.get_ranks()->size());
      slice->sync_from_device();
    }
    check_return_value("gather", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::allgather(BaseBuffer &sendbuf,
                             BaseBuffer &recvbuf, unsigned int count,
                             communicatorId comm_id, bool from_fpga, bool to_fpga,
                             dataType compress_dtype, bool run_async,
                             std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(sendbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Allgather is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Allgather does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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
      (count + segment_size - 1) / segment_size *
              communicator.get_ranks()->size() >
          rx_buffer_spares.size()) {
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
  options.addr_2 = &recvbuf;
  options.count = count;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count * communicator.get_ranks()->size());
      slice->sync_from_device();
    }
    check_return_value("allgather", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::reduce(BaseBuffer &sendbuf,
                          BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                          reduceFunction func, communicatorId comm_id, bool from_fpga,
                          bool to_fpga, dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(sendbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Reduce is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Reduce does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false && is_root == true) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("reduce", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::reduce(dataType src_data_type,
                          BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                          reduceFunction func, communicatorId comm_id,
                          bool to_fpga, dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(src_data_type) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Reduce is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Reduce does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false && is_root == true) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("reduce", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::reduce(BaseBuffer &sendbuf, dataType dst_data_type,
                          unsigned int count, unsigned int root,
                          reduceFunction func, communicatorId comm_id, bool from_fpga,
                          dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(sendbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Reduce is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Reduce does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("reduce", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::reduce(dataType src_data_type, dataType dst_data_type,
                          unsigned int count, unsigned int root,
                          reduceFunction func, communicatorId comm_id,
                          dataType compress_dtype, bool run_async,
                          std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(src_data_type) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Reduce is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Reduce does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    check_return_value("reduce", handle);
  }

  return nullptr;
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
  options.addr_2 = &recvbuf;
  options.count = count;
  options.reduce_function = func;
  options.compress_dtype = compress_dtype;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("allreduce", handle);
  }

  return nullptr;
}

ACCLRequest *ACCL::reduce_scatter(BaseBuffer &sendbuf,
                                  BaseBuffer &recvbuf, unsigned int count,
                                  reduceFunction func, communicatorId comm_id,
                                  bool from_fpga, bool to_fpga,
                                  dataType compress_dtype, bool run_async,
                                  std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  unsigned count_bytes = count * (dataTypeSize.at(sendbuf.type()) / 8);
  if (count_bytes > segment_size) {
    std::cerr << "[ACCL] Warning: Reduce scatter is too large! (" << count_bytes
              << " B > " << segment_size << " B). "
              << "Reduce scatter does not currently support segmentation; "
              << "the operation might get stuck or data may be corrupted. "
              << std::endl;
  }

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

  if (run_async) {
    return handle;
  } else {
    wait(handle);
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("reduce_scatter", handle);
  }

  return nullptr;
}

void ACCL::barrier(communicatorId comm_id,
                   std::vector<ACCLRequest *> waitfor) {
  CCLO::Options options{};

  const Communicator &communicator = communicators[comm_id];
  options.scenario = operation::barrier;
  options.comm = communicator.communicators_addr();
  options.addr_0 = utility_spare;
  options.waitfor = waitfor;
  ACCLRequest *handle = call_async(options);

  wait(handle);
  check_return_value("barrier", handle);

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

std::string ACCL::dump_rx_buffers(size_t nbufs, bool dump_data) {
  std::stringstream stream;
  stream << "CCLO address: " << std::hex << cclo->get_base_addr() << std::endl;
  nbufs = std::min(nbufs, rx_buffer_spares.size());

  addr_t address = rx_buffers_adr;
  stream << "rx address: " << address << std::dec << std::endl;
  for (size_t i = 0; i < nbufs; ++i) {
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
    val_t maxsize = cclo->read(address);
    address += 4;
    val_t rxtag = cclo->read(address);
    address += 4;
    val_t rxlen = cclo->read(address);
    address += 4;
    val_t rxsrc = cclo->read(address);
    address += 4;
    val_t seq = cclo->read(address);

    rx_buffer_spares[i]->sync_from_device();

    stream << "Spare RX Buffer " << i << ":\t address: 0x" << std::hex
           << addrh * (1UL << 32) + addrl << std::dec
           << " \t status: " << status << " \t occupancy: " << rxlen << "/"
           << maxsize << " \t MPI tag: " << std::hex << rxtag << std::dec
           << " \t seq: " << seq << " \t src: " << rxsrc;

    if(dump_data) {
      stream << " \t data: " << std::hex << "[";
      for (size_t j = 0; j < rx_buffer_spares[i]->size(); ++j) {
        stream << "0x"
              << static_cast<uint16_t>(static_cast<uint8_t *>(
                      rx_buffer_spares[i]->byte_array())[j]);
        if (j != rx_buffer_spares[i]->size() - 1) {
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

void ACCL::initialize_accl(const std::vector<rank_t> &ranks, int local_rank,
                           int nbufs, addr_t bufsize , addr_t segsize) {
  reset_log();
  debug("CCLO HWID: " + std::to_string(get_hwid()) + " at 0x" +
        debug_hex(cclo->get_base_addr()));

  if (cclo->read(CFGRDY_OFFSET) != 0) {
    throw std::runtime_error("CCLO appears configured, might be in use. Please "
                             "reset the CCLO and retry");
  }

  debug("Configuring RX Buffers");
  setup_rx_buffers(nbufs, bufsize, rxbufmem);

  debug("Configuring a communicator");
  configure_communicator(ranks, local_rank);

  debug("Configuring arithmetic");
  configure_arithmetic();

  // Mark CCLO as configured
  debug("CCLO configured");
  cclo->write(CFGRDY_OFFSET, 1);
  config_rdy = true;

  debug("Set timeout");
  set_timeout(1000000);

  CCLO::Options options{};
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::enable_pkt;
  call_sync(options, false);

  debug("Set max segment size: " + std::to_string(segsize));
  set_max_segment_size(segsize);
  switch (protocol) {
  case networkProtocol::UDP:
    use_udp();
    break;
  case networkProtocol::TCP:
    use_tcp();
    break;
  default:
    throw std::runtime_error(
        "Requested network protocol is not yet supported.");
  }

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

void ACCL::setup_rx_buffers(size_t nbufs, addr_t bufsize,
                            const std::vector<int> &devicemem) {
  addr_t address = rx_buffers_adr;
  rx_buffer_size = bufsize;
  for (size_t i = 0; i < nbufs; ++i) {
    // create, clear and sync buffers to device
    Buffer<int8_t> *buf;

    if (sim_mode) {
      buf = new SimBuffer(new int8_t[bufsize](), bufsize, dataType::int8,
                          static_cast<SimDevice *>(cclo)->get_context());
    } else {
      buf = new FPGABuffer<int8_t>(bufsize, dataType::int8, device,
                                   devicemem[i % devicemem.size()]);
    }

    buf->sync_to_device();
    rx_buffer_spares.emplace_back(buf);
    // program this buffer into the accelerator
    address += 4;
    cclo->write(address, 0);
    address += 4;
    cclo->write(address, buf->physical_address() & 0xffffffff);
    address += 4;
    cclo->write(address, (buf->physical_address() >> 32) & 0xffffffff);
    address += 4;
    cclo->write(address, bufsize);

    // clear remaining fields
    for (size_t j = 4; j < 8; ++j) {
      address += 4;
      cclo->write(address, 0);
    }
  }

  // NOTE: the buffer count HAS to be written last (offload checks for this)
  cclo->write(rx_buffers_adr, nbufs);

  current_config_address = address + 4;
  if (sim_mode) {
    utility_spare =
        new SimBuffer(new int8_t[bufsize](), bufsize, dataType::int8,
                      static_cast<SimDevice *>(cclo)->get_context());
  } else {
    utility_spare =
        new FPGABuffer<int8_t>(bufsize, dataType::int8, device, devicemem[0]);
  }
}

void ACCL::check_return_value(const std::string function_name, ACCLRequest *request) {
  val_t retcode = request->get_retcode();
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

void ACCL::prepare_call(CCLO::Options &options, bool check_tcp) {
  const ArithConfig *arithcfg;

  if (check_tcp && protocol == networkProtocol::TCP) {
    if (!port_open) {
      throw std::runtime_error("ACCL not ready yet; port still closed!");
    }

    if (!con_open) {
      throw std::runtime_error("ACCL not ready yet; connection still closed!");
    }
  }

  std::set<dataType> dtypes;

  if (options.addr_0 == nullptr) {
    options.addr_0 = &dummy_buffer;
    dtypes.insert(options.data_type_io_0);
  }
  else {
    dtypes.insert(options.addr_0->type());
  }

  if (options.addr_1 == nullptr) {
    options.addr_1 = &dummy_buffer;
    dtypes.insert(options.data_type_io_1);
  }
  else {
    dtypes.insert(options.addr_1->type());
  }

  if (options.addr_2 == nullptr) {
    options.addr_2 = &dummy_buffer;
    dtypes.insert(options.data_type_io_2);
  }
  else {
    dtypes.insert(options.addr_2->type());
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

void ACCL::free_request(ACCLRequest *request) {
  return cclo->free_request(request);
}

ACCLRequest *ACCL::call_async(CCLO::Options &options, bool check_tcp) {
  if (!config_rdy) {
    throw std::runtime_error("CCLO not configured, cannot call");
  }

  prepare_call(options, check_tcp);
  ACCLRequest *req = cclo->start(options);
  return req;
}

ACCLRequest *ACCL::call_sync(CCLO::Options &options, bool check_tcp) {
  if (!config_rdy) {
    throw std::runtime_error("CCLO not configured, cannot call");
  }

  prepare_call(options, check_tcp);
  ACCLRequest *req = cclo->call(options);
  return req;
}

void ACCL::open_port(communicatorId comm_id) {
  if (port_open) {
    throw std::runtime_error("Port already open");
  }

  debug("Opening ports...");

  CCLO::Options options{};
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::open_port;
  ACCLRequest *handle = call_sync(options, false);
  check_return_value("open_port", handle);
  port_open = true;
  debug("Ports open!");
}

void ACCL::open_con(communicatorId comm_id) {
  if (con_open) {
    throw std::runtime_error("Connection already open");
  }

  debug("Opening connections...");

  CCLO::Options options{};
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::open_con;
  ACCLRequest *handle = call_sync(options, false);
  check_return_value("open_con", handle);
  con_open = true;
  debug("Connections open!");
}

void ACCL::close_con(communicatorId comm_id) {
    if (!con_open) {
    throw std::runtime_error("Connection not open yet");
  }

  CCLO::Options options{};
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::close_con;
  ACCLRequest *handle = call_sync(options, false);
  check_return_value("close_con", handle);
  con_open = false;
}

void ACCL::use_udp(communicatorId comm_id) {
  CCLO::Options options{};
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::set_stack_type;
  options.count = 0;
  ACCLRequest *handle = call_sync(options, false);
  check_return_value("use_udp", handle);
}

void ACCL::use_tcp(communicatorId comm_id) {
  CCLO::Options options{};
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::set_stack_type;
  options.count = 1;
  ACCLRequest *handle = call_sync(options, false);
  check_return_value("use_tcp", handle);
}

void ACCL::set_max_segment_size(unsigned int value) {
  if (value % 8 != 0) {
    std::cerr << "ACCL: dma transaction must be divisible by 8 to use reduce "
                 "collectives."
              << std::endl;
  }

  if (value > rx_buffer_size) {
    throw std::runtime_error("Transaction size should be less or equal "
                             "to configured buffer size.");
  }

  CCLO::Options options{};
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::set_max_segment_size;
  options.count = value;
  ACCLRequest *handle = call_sync(options, false);
  segment_size = value;
  check_return_value("set_max_segment_size", handle);
}

void ACCL::configure_communicator(const std::vector<rank_t> &ranks,
                                            int local_rank) {
  if (rx_buffer_spares.empty()) {
    throw std::runtime_error(
        "RX buffers unconfigured, please call setup_rx_buffers() first.");
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
