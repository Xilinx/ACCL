#include "accl.hpp"
#include "common.hpp"
#include "dummybuffer.hpp"
#include "simbuffer.hpp"
#include "simdevice.hpp"
#include <jsoncpp/json/json.h>
#include <math.h>
#include <set>

namespace ACCL {
// Hardware constructor
ACCL::ACCL(const std::vector<rank_t> &ranks, int local_rank, int board_idx,
           int devicemem, std::vector<int> &rxbufmem, int networkmem,
           networkProtocol protocol, int nbufs, addr_t bufsize,
           const arithConfigMap &arith_config)
    : protocol(protocol), sim_mode(false), sim_sock(""), devicemem(devicemem),
      rxbufmem(rxbufmem), networkmem(networkmem), arith_config(arith_config) {
  // TODO: Create hardware constructor.
  throw std::logic_error("Hardware constructor currently not supported");
}

// Simulation constructor
ACCL::ACCL(const std::vector<rank_t> &ranks, int local_rank,
           std::string sim_sock, networkProtocol protocol, int nbufs,
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
  call_async(options);

  if (run_async) {
    return cclo;
  } else {
    cclo->wait();
  }

  return NULL;
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

  if (cclo->read(CFGRDY_OFFSET) == 0) {
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
  call_async(options);

  set_max_segment_size(bufsize);
  switch (protocol) {
  case networkProtocol::UDP:
    use_udp();
    break;
  case networkProtocol::TCP:
    if (!sim_mode) {
      throw new std::runtime_error("TODO: Allocate buffers.");
      tx_buf_network->sync_to_device();
      rx_buf_network->sync_to_device();
    }
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
  for (const auto &[_key, arithcfg] : arith_config) {
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
      buf = new SimBuffer(new int8_t[bufsize], bufsize, dataType::int8,
                          static_cast<SimDevice *>(cclo)->get_socket());
    } else {
      std::runtime_error("TODO: allocate hw buffer.");
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

    cclo->write(rx_buffers_adr, nbufs);

    communicators_addr = address + 4;
    if (sim_mode) {
      utility_spare =
          new SimBuffer(new int8_t[bufsize], bufsize, dataType::int8,
                        static_cast<SimDevice *>(cclo)->get_socket());
    } else {
      std::runtime_error("TODO: allocate hw buffer.");
    }
  }
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
  options.comm = communicators[comm_id].comm_addr();
  options.cfg_function = cfgFunc::open_port;
  call_sync(options);
}

void ACCL::open_con(unsigned int comm_id) {
  CCLO::Options options = CCLO::Options();
  options.scenario = operation::config;
  options.comm = communicators[comm_id].comm_addr();
  options.cfg_function = cfgFunc::open_con;
  call_sync(options);
}

} // namespace ACCL
