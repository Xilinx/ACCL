#pragma once

#include "arithconfig.hpp"
#include "buffer.hpp"
#include "cclo.hpp"
#include "communicator.hpp"
#include "constants.hpp"
#include <stdexcept>
#include <string>
#include <vector>

/** @file accl.hpp */

namespace ACCL {
class ACCL {
public:
#ifdef ACCL_HARDWARE_SUPPORT
  ACCL(const std::vector<rank_t> &ranks, int local_rank, int board_idx,
       int devicemem, std::vector<int> &rxbufmem, int networkmem,
       networkProtocol protocol = networkProtocol::TCP, int nbufs = 16,
       addr_t bufsize = 1024,
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);
#endif
  // Simulation constructor
  ACCL(const std::vector<rank_t> &ranks, int local_rank,
       const std::string &sim_sock,
       networkProtocol protocol = networkProtocol::TCP, int nbufs = 16,
       addr_t bufsize = 1024,
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);

  ~ACCL();

  void deinit();

  val_t get_retcode() { return this->cclo->read(RETCODE_OFFSET); }

  val_t get_hwid() { return this->cclo->read(IDCODE_OFFSET); }

  CCLO *set_timeout(unsigned int value, bool run_async = false,
                    std::vector<CCLO *> waitfor = {});

  CCLO *nop(bool run_async = false, std::vector<CCLO *> waitfor = {});

  CCLO *send(unsigned int comm_id, BaseBuffer &srcbuf, unsigned int count,
             unsigned int dst, unsigned int tag = TAG_ANY,
             bool from_fpga = false,
             streamFlags stream_flags = streamFlags::NO_STREAM,
             bool run_async = false, std::vector<CCLO *> waitfor = {});

  CCLO *recv(unsigned int comm_id, BaseBuffer &dstbuf, unsigned int count,
             unsigned int src, unsigned int tag = TAG_ANY, bool to_fpga = false,
             streamFlags stream_flags = streamFlags::NO_STREAM,
             bool run_async = false, std::vector<CCLO *> waitfor = {});

  CCLO *copy(BaseBuffer &srcbuf, BaseBuffer &dstbuf, unsigned int count,
             bool from_fpga = false, bool to_fpga = false,
             bool run_async = false, std::vector<CCLO *> waitfor = {});

  CCLO *combine(unsigned int count, reduceFunction function, BaseBuffer &val1,
                BaseBuffer &val2, BaseBuffer &result,
                bool val1_from_fpga = false, bool val2_from_fpga = false,
                bool to_fpga = false, bool run_async = false,
                std::vector<CCLO *> waitfor = {});

  CCLO *external_stream_kernel(BaseBuffer &srcbuf, BaseBuffer &dstbuf,
                               bool from_fpga = false, bool to_fpga = false,
                               bool run_async = false,
                               std::vector<CCLO *> waitfor = {});

  CCLO *bcast(unsigned int comm_id, BaseBuffer &buf, unsigned int count,
              unsigned int root, bool from_fpga = false, bool to_fpga = false,
              bool run_async = false, std::vector<CCLO *> waitfor = {});

  CCLO *scatter(unsigned int comm_id, BaseBuffer &sendbuf, BaseBuffer &recvbuf,
                unsigned int count, unsigned int root, bool from_fpga = false,
                bool to_fpga = false, bool run_async = false,
                std::vector<CCLO *> waitfor = {});

  CCLO *gather(unsigned int comm_id, BaseBuffer &sendbuf, BaseBuffer &recvbuf,
               unsigned int count, unsigned int root, bool from_fpga = false,
               bool to_fpga = false, bool run_async = false,
               std::vector<CCLO *> waitfor = {});

  CCLO *allgather(unsigned int comm_id, BaseBuffer &sendbuf,
                  BaseBuffer &recvbuf, unsigned int count,
                  bool from_fpga = false, bool to_fpga = false,
                  bool run_async = false, std::vector<CCLO *> waitfor = {});

  CCLO *reduce(unsigned int comm_id, BaseBuffer &sendbuf, BaseBuffer &recvbuf,
               unsigned int count, unsigned int root, reduceFunction func,
               bool from_fpga = false, bool to_fpga = false,
               bool run_async = false, std::vector<CCLO *> waitfor = {});

  CCLO *allreduce(unsigned int comm_id, BaseBuffer &sendbuf,
                  BaseBuffer &recvbuf, unsigned int count, reduceFunction func,
                  bool from_fpga = false, bool to_fpga = false,
                  bool run_async = false, std::vector<CCLO *> waitfor = {});

  CCLO *reduce_scatter(unsigned int comm_id, BaseBuffer &sendbuf,
                       BaseBuffer &recvbuf, unsigned int count,
                       reduceFunction func, bool from_fpga = false,
                       bool to_fpga = false, bool run_async = false,
                       std::vector<CCLO *> waitfor = {});

private:
  CCLO *cclo{};
  // Supported types and corresponding arithmetic config
  arithConfigMap arith_config;
  addr_t arithcfg_addr{};
  // RX spare buffers
  std::vector<Buffer<int8_t> *> rx_buffer_spares;
  addr_t rx_buffer_size{};
  addr_t rx_buffers_adr{};
  // Buffers for POE
  Buffer<int8_t> *tx_buf_network{};
  Buffer<int8_t> *rx_buf_network{};
  // Spare buffer for general use
  Buffer<int8_t> *utility_spare{};
  // List of communicators, to which users will add
  std::vector<Communicator> communicators;
  addr_t communicators_addr{};
  // safety checks
  bool check_return_value_flag{};
  bool ignore_safety_checks{};
  // TODO: use description to gather info about where to allocate spare buffers
  addr_t segment_size{};
  // protocol being used
  const networkProtocol protocol;
  // flag to indicate whether we've finished config
  bool config_rdy{};
  // flag to indicate whether we're simulating
  const bool sim_mode;
  const std::string sim_sock;
  // memory banks for hardware
  const int devicemem;
  const std::vector<int> rxbufmem;
  const int networkmem;

  std::string dump_exchange_memory();

  std::string dump_rx_buffers(size_t nbufs);
  std::string dump_rx_buffers() {
    if (cclo->read(rx_buffers_adr) != rx_buffer_spares.size()) {
      throw std::runtime_error("CCLO inconsistent");
    }
    return dump_rx_buffers(rx_buffer_spares.size());
  }

  void initialize_accl(const std::vector<rank_t> &ranks, int local_rank,
                       int nbufs, addr_t bufsize);

  void configure_arithmetic();

  void setup_rx_buffers(size_t nbufs, addr_t bufsize,
                        const std::vector<int> &devicemem);
  void setup_rx_buffers(size_t nbufs, addr_t bufsize, int devicemem) {
    std::vector<int> mems = {devicemem};
    return setup_rx_buffers(nbufs, bufsize, mems);
  }

  void check_return_value(const std::string function_name);

  void prepare_call(CCLO::Options &options);

  CCLO *call_async(CCLO::Options &options);

  CCLO *call_sync(CCLO::Options &options);

  void init_connection(unsigned int comm_id = 0);

  void open_port(unsigned int comm_id = 0);

  void open_con(unsigned int comm_id = 0);

  void use_udp(unsigned int comm_id = 0);

  void use_tcp(unsigned int comm_id = 0);

  void set_max_segment_size(unsigned int value = 0);

  void configure_communicator(const std::vector<rank_t> &ranks, int local_rank);

  std::string dump_communicator();
};

} // namespace ACCL
