#include "accl.hpp"
#include "common.hpp"
#include "dummybuffer.hpp"
#include "simbuffer.hpp"
#include "simdevice.hpp"
#include <cmath>
#include <jsoncpp/json/json.h>
#include <set>

namespace ACCL {
#ifdef ACCL_HARDWARE_SUPPORT
ACCL::ACCL(const std::vector<rank_t> &ranks, int local_rank, int board_idx,
           int devicemem, std::vector<int> &rxbufmem, int networkmem,
           networkProtocol protocol, int nbufs, addr_t bufsize,
           const arithConfigMap &arith_config)
    : protocol(protocol), sim_mode(false), sim_sock(""), devicemem(devicemem),
      rxbufmem(rxbufmem), networkmem(networkmem), arith_config(arith_config) {
  // TODO: Create hardware constructor.
  throw std::logic_error("Hardware constructor currently not supported");
}
#endif

// Simulation constructor
ACCL::ACCL(const std::vector<rank_t> &ranks, int local_rank,
           const std::string &sim_sock, networkProtocol protocol, int nbufs,
           addr_t bufsize, const arithConfigMap &arith_config)
    : protocol(protocol), sim_mode(true), sim_sock(sim_sock), devicemem(0),
      rxbufmem({}), networkmem(0), arith_config(arith_config) {
  cclo = new SimDevice(sim_sock);
  initialize_accl(ranks, local_rank, nbufs, bufsize);
}

ACCL::~ACCL() {
  deinit();
  delete cclo;
  delete tx_buf_network;
  delete rx_buf_network;
}

void ACCL::deinit() {
  debug("Removing CCLO object at " + debug_hex(cclo->get_base_addr()));

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

CCLO *ACCL::set_timeout(unsigned int value, bool run_async,
                        std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();
  options.scenario = operation::config;
  options.count = value;
  options.cfg_function = cfgFunc::set_timeout;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    check_return_value("set_timeout");
  }

  return nullptr;
}

CCLO *ACCL::nop(bool run_async, std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();
  options.scenario = operation::nop;
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);
  if (run_async) {
    return handle;
  } else {
    handle->wait();
    check_return_value("nop");
  }

  return nullptr;
}

CCLO *ACCL::send(unsigned int comm_id, BaseBuffer &srcbuf, unsigned int count,
                 unsigned int dst, unsigned int tag, bool from_fpga,
                 streamFlags stream_flags, bool run_async,
                 std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();
  if (from_fpga == false) {
    srcbuf.sync_to_device();
  }
  options.scenario = operation::send;
  options.comm = communicators[comm_id].communicators_addr();
  options.addr_0 = &srcbuf;
  options.count = count;
  options.root_src_dst = dst;
  options.tag = tag;
  options.stream_flags = stream_flags;
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    check_return_value("send");
  }

  return nullptr;
}

CCLO *ACCL::recv(unsigned int comm_id, BaseBuffer &dstbuf, unsigned int count,
                 unsigned int src, unsigned int tag, bool to_fpga,
                 streamFlags stream_flags, bool run_async,
                 std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  options.stream_flags = stream_flags;
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      dstbuf.sync_from_device();
    }
    check_return_value("send");
  }

  return nullptr;
}

CCLO *ACCL::copy(BaseBuffer &srcbuf, BaseBuffer &dstbuf, unsigned int count,
                 bool from_fpga, bool to_fpga, bool run_async,
                 std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (from_fpga == false) {
    srcbuf.sync_to_device();
  }

  options.scenario = operation::copy;
  options.addr_0 = &srcbuf;
  options.addr_2 = &dstbuf;
  options.count = count;
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      dstbuf.sync_from_device();
    }
    check_return_value("copy");
  }

  return nullptr;
}

CCLO *ACCL::combine(unsigned int count, reduceFunction function,
                    BaseBuffer &val1, BaseBuffer &val2, BaseBuffer &result,
                    bool val1_from_fpga, bool val2_from_fpga, bool to_fpga,
                    bool run_async, std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      result.sync_from_device();
    }
    check_return_value("combine");
  }

  return nullptr;
}

CCLO *ACCL::external_stream_kernel(BaseBuffer &srcbuf, BaseBuffer &dstbuf,
                                   bool from_fpga, bool to_fpga, bool run_async,
                                   std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

  if (to_fpga == false && run_async == true) {
    std::cerr << "ACCL: async run returns data on FPGA, user must "
                 "sync_from_device() after waiting"
              << std::endl;
  }

  if (srcbuf.size() <= 4) {
    std::cerr << "ACCL: size of buffer not compatible" << std::endl;
  }

  if (from_fpga == false) {
    srcbuf.sync_to_device();
  }

  options.scenario = operation::ext_stream_krnl;
  options.addr_0 = &srcbuf;
  options.addr_1 = &dstbuf;
  options.count = srcbuf.size();
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      dstbuf.sync_from_device();
    }
    check_return_value("external_stream_kernel");
  }

  return nullptr;
}

CCLO *ACCL::bcast(unsigned int comm_id, BaseBuffer &buf, unsigned int count,
                  unsigned int root, bool from_fpga, bool to_fpga,
                  bool run_async, std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      buf.sync_from_device();
    }
    check_return_value("bcast");
  }

  return nullptr;
}

CCLO *ACCL::scatter(unsigned int comm_id, BaseBuffer &sendbuf,
                    BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                    bool from_fpga, bool to_fpga, bool run_async,
                    std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("scatter");
  }

  return nullptr;
}

CCLO *ACCL::gather(unsigned int comm_id, BaseBuffer &sendbuf,
                   BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                   bool from_fpga, bool to_fpga, bool run_async,
                   std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false && is_root == true) {
      auto slice = recvbuf.slice(0, count * communicator.get_ranks()->size());
      slice->sync_from_device();
    }
    check_return_value("gather");
  }

  return nullptr;
}

CCLO *ACCL::allgather(unsigned int comm_id, BaseBuffer &sendbuf,
                      BaseBuffer &recvbuf, unsigned int count, bool from_fpga,
                      bool to_fpga, bool run_async,
                      std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count * communicator.get_ranks()->size());
      slice->sync_from_device();
    }
    check_return_value("allgather");
  }

  return nullptr;
}

CCLO *ACCL::reduce(unsigned int comm_id, BaseBuffer &sendbuf,
                   BaseBuffer &recvbuf, unsigned int count, unsigned int root,
                   reduceFunction func, bool from_fpga, bool to_fpga,
                   bool run_async, std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false && is_root == true) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("reduce");
  }

  return nullptr;
}

CCLO *ACCL::allreduce(unsigned int comm_id, BaseBuffer &sendbuf,
                      BaseBuffer &recvbuf, unsigned int count,
                      reduceFunction func, bool from_fpga, bool to_fpga,
                      bool run_async, std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("allgather");
  }

  return nullptr;
}

CCLO *ACCL::reduce_scatter(unsigned int comm_id, BaseBuffer &sendbuf,
                           BaseBuffer &recvbuf, unsigned int count,
                           reduceFunction func, bool from_fpga, bool to_fpga,
                           bool run_async, std::vector<CCLO *> waitfor) {
  CCLO::Options options = CCLO::Options();

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
  options.waitfor = waitfor;
  CCLO *handle = call_async(options);

  if (run_async) {
    return handle;
  } else {
    handle->wait();
    if (to_fpga == false) {
      auto slice = recvbuf.slice(0, count);
      slice->sync_from_device();
    }
    check_return_value("reduce_scatter");
  }

  return nullptr;
}

std::string ACCL::dump_exchange_memory() {
  std::stringstream stream;
  stream << "exchange mem:" << std::endl;
  size_t num_word_per_line = 4;
  for (size_t i = 0; i < EXCHANGE_MEM_ADDRESS_RANGE;
       i += 4 * num_word_per_line) {
    stream << std::hex << EXCHANGE_MEM_OFFSET_ADDRESS + i << ": ";
    for (size_t j = 0; j < num_word_per_line; ++j) {
      stream << std::hex << cclo->read(EXCHANGE_MEM_OFFSET_ADDRESS + i + j * 4);
    }
    stream << std::endl;
  }

  return stream.str();
}

std::string ACCL::dump_rx_buffers(size_t nbufs) {
  std::stringstream stream;
  stream << "CCLO address: " << std::hex << cclo->get_base_addr() << std::endl;
  nbufs = std::min(nbufs, rx_buffer_spares.size());

  addr_t address = rx_buffers_adr;
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
           << addrh * (1UL << 32) + addrl << " \t status: " << status
           << " \t occupancy: " << rxlen << "/" << maxsize
           << " \t MPI tag: " << std::hex << rxtag << " \t seq: " << seq
           << " \t src: " << rxsrc << " \t data: ";
    for (size_t j = 0; j < rx_buffer_spares[i]->size(); ++j) {
      stream << std::hex
             << static_cast<uint8_t *>(rx_buffer_spares[i]->byte_array())[j];
    }
    stream << std::endl;
  }

  return stream.str();
}

void ACCL::initialize_accl(const std::vector<rank_t> &ranks, int local_rank,
                           int nbufs, addr_t bufsize) {
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
  cclo->write(CFGRDY_OFFSET, 1);
  config_rdy = true;

  set_timeout(1000000);

  CCLO::Options options = CCLO::Options();
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::enable_pkt;
  call_sync(options);

  set_max_segment_size(bufsize);
  switch (protocol) {
  case networkProtocol::UDP:
    use_udp();
    break;
  case networkProtocol::TCP:
#ifdef ACCL_HARDWARE_SUPPORT
    if (!sim_mode) {
      throw new std::runtime_error("TODO: Allocate buffers.");
      tx_buf_network->sync_to_device();
      rx_buf_network->sync_to_device();
    }
#endif
    use_tcp();
    break;
  default:
    throw std::runtime_error(
        "Requested network protocol is not yet supported.");
  }

  if (protocol == networkProtocol::TCP) {
    debug("Starting connections to communicator ranks");
  }

  debug("Accelerator ready!");
}

void ACCL::configure_arithmetic() {
  if (communicators.size() < 1) {
    throw new std::runtime_error("Communicators unconfigured, please call "
                                 "configure_communicator() first");
  }

  addr_t address = arithcfg_addr;
  for (auto &[_key, arithcfg] : arith_config) {
    write_arithconfig(*cclo, arithcfg, &address);
  }
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
                          static_cast<SimDevice *>(cclo)->get_socket());
    }
#ifdef ACCL_HARDWARE_SUPPORT
    else {
      std::runtime_error("TODO: allocate hw buffer.");
    }
#endif

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

  communicators_addr = address + 4;
  if (sim_mode) {
    utility_spare =
        new SimBuffer(new int8_t[bufsize](), bufsize, dataType::int8,
                      static_cast<SimDevice *>(cclo)->get_socket());
  }
#ifdef ACCL_HARDWARE_SUPPORT
  else {
    std::runtime_error("TODO: allocate hw buffer.");
  }
#endif
}

void ACCL::check_return_value(const std::string function_name) {
  val_t retcode = get_retcode();
  if (retcode != 0) {
    std::stringstream stream;
    stream << std::hex << cclo->get_base_addr();
    throw std::runtime_error(
        "CCLO @0x" + stream.str() + ": during " + function_name + " error " +
        std::to_string(retcode) +
        " occured. You should consider resetting mpi_offload.");
  }
}

void ACCL::prepare_call(CCLO::Options &options) {
  const ArithConfig *arithcfg;
  /* TODO: Does this work on hardware? */
  if (options.addr_0 == nullptr) {
    options.addr_0 = &dummy_buffer;
  }

  if (options.addr_1 == nullptr) {
    options.addr_1 = &dummy_buffer;
  }

  if (options.addr_2 == nullptr) {
    options.addr_2 = &dummy_buffer;
  }

  std::set<dataType> dtypes = {options.addr_0->type(), options.addr_1->type(),
                               options.addr_2->type()};
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
      arithcfg = &this->arith_config.at(key);
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

CCLO *ACCL::call_async(CCLO::Options &options) {
  if (!config_rdy) {
    throw std::runtime_error("CCLO not configured, cannot call");
  }

  prepare_call(options);
  cclo->start(options);
  return cclo;
}

CCLO *ACCL::call_sync(CCLO::Options &options) {
  if (!config_rdy) {
    throw std::runtime_error("CCLO not configured, cannot call");
  }

  prepare_call(options);
  cclo->call(options);
  return cclo;
}

void ACCL::init_connection(unsigned int comm_id) {
  debug("Opening ports to communicator ranks");
  open_port(comm_id);
  debug("Starting session to communicator ranks");
  open_con(comm_id);
}

void ACCL::open_port(unsigned int comm_id) {
  CCLO::Options options = CCLO::Options();
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::open_port;
  call_sync(options);
  check_return_value("open_port");
}

void ACCL::open_con(unsigned int comm_id) {
  CCLO::Options options = CCLO::Options();
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::open_con;
  call_sync(options);
  check_return_value("open_con");
}

void ACCL::use_udp(unsigned int comm_id) {
  CCLO::Options options = CCLO::Options();
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::set_stack_type;
  options.count = 0;
  call_sync(options);
  check_return_value("use_udp");
}

void ACCL::use_tcp(unsigned int comm_id) {
  CCLO::Options options = CCLO::Options();
  options.scenario = operation::config;
  options.comm = communicators[comm_id].communicators_addr();
  options.cfg_function = cfgFunc::set_stack_type;
  options.count = 1;
  call_sync(options);
  check_return_value("use_tcp");
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

  CCLO::Options options = CCLO::Options();
  options.scenario = operation::config;
  options.cfg_function = cfgFunc::set_max_segment_size;
  options.count = value;
  call_sync(options);
  segment_size = value;
  check_return_value("set_max_segment_size");
}

void ACCL::configure_communicator(const std::vector<rank_t> &ranks,
                                  int local_rank) {
  if (rx_buffer_spares.empty()) {
    throw std::runtime_error(
        "RX buffers unconfigured, please call setup_rx_buffers() first.");
  }

  addr_t addr;

  if (communicators.empty()) {
    addr = communicators_addr;
  } else {
    addr = communicators.back().communicators_addr();
  }

  communicators.emplace_back(Communicator(cclo, ranks, local_rank, addr));
}

std::string ACCL::dump_communicator() {
  std::stringstream stream;
  for (size_t i = 0; i < communicators.size(); ++i) {
    stream << "Communicator " << i << ":" << std::endl
           << communicators[i].dump();
  }

  return stream.str();
}

} // namespace ACCL
