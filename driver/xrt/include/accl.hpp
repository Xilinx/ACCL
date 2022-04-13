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

#pragma once

#include "arithconfig.hpp"
#include "buffer.hpp"
#include "cclo.hpp"
#include "communicator.hpp"
#include "constants.hpp"
#include "fpgabuffer.hpp"
#include "fpgadevice.hpp"
#include "simbuffer.hpp"
#include "simdevice.hpp"
#include <stdexcept>
#include <string>
#include <vector>

/** @file accl.hpp */

namespace ACCL {

/**
 * Main ACCL class that talks to the CCLO on hardware or emulation/simulation.
 *
 */
class ACCL {
public:
#ifdef ACCL_HARDWARE_SUPPORT
  /**
   * Construct a new ACCL object that talks to hardware.
   *
   * @param ranks         All ranks on the network
   * @param local_rank    Rank of this process
   * @param board_idx     Board id
   * @param devicemem     Memory bank of device memory
   * @param rxbufmem      Memory banks of rxbuf memory
   * @param networkmem    Memory bank of network memory
   * @param protocol      Network protocol to use
   * @param nbufs         Amount of buffers to use
   * @param bufsize       Size of buffers
   * @param arith_config  Arithmetic configuration to use
   */
  ACCL(const std::vector<rank_t> &ranks, int local_rank, xrt::device &device,
       xrt::ip &cclo_ip, xrt::kernel &hostctrl_ip, int devicemem,
       const std::vector<int> &rxbufmem, int networkmem,
       networkProtocol protocol = networkProtocol::TCP, int nbufs = 16,
       addr_t bufsize = 1024,
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);
#endif
  /**
   * Construct a new ACCL object that talks to emulator/simulator.
   *
   * @param ranks         All ranks on the network
   * @param local_rank    Rank of this process
   * @param sim_sock      ZMQ socket of emulator/simulator.
   * @param protocol      Network protocol to use
   * @param nbufs         Amount of buffers to use
   * @param bufsize       Size of buffers
   * @param arith_config  Arithmetic configuration to use

   */
  ACCL(const std::vector<rank_t> &ranks, int local_rank,
       const std::string &sim_sock,
       networkProtocol protocol = networkProtocol::TCP, int nbufs = 16,
       addr_t bufsize = 1024,
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);

  /**
   * Destroy the ACCL object. Automatically deinitializes the CCLO.
   *
   */
  ~ACCL();

  /**
   * Deinitializes the CCLO.
   *
   */
  void deinit();

  /**
   * Get the return code of the last ACCL call.
   *
   * @return val_t The return code
   */
  val_t get_retcode() { return this->cclo->read(RETCODE_OFFSET); }

  /**
   * Get the hardware id from the FPGA.
   *
   * @return val_t The hardware id
   */
  val_t get_hwid() { return this->cclo->read(IDCODE_OFFSET); }

  /**
   * Set the timeout of ACCL calls.
   *
   * @param value      Timeout in miliseconds
   * @param run_async  Run the ACCL call asynchronously.
   * @param waitfor    ACCL call will wait for these operations before it will
   *                   start.
   * @return CCLO*     CCLO object that can be passed to waitfor;
   *                   nullptr if run_async is false.
   */
  CCLO *set_timeout(unsigned int value, bool run_async = false,
                    std::vector<CCLO *> waitfor = {});

  /**
   * Performs the nop operation on the FPGA.
   *
   * @param run_async  Run the ACCL call asynchronously.
   * @param waitfor    ACCL call will wait for these operations before it will
   *                   start.
   * @return CCLO*     CCLO object that can be passed to waitfor;
   *                   nullptr if run_async is false.
   */
  CCLO *nop(bool run_async = false, std::vector<CCLO *> waitfor = {});

  /**
   * Performs the send operation on the FPGA.
   *
   * @param comm_id      Index of communicator to use.
   * @param srcbuf       Buffer that contains the data to be send. Create a
   *                     buffer using ACCL::create_buffer.
   * @param count        Amount of elements in buffer to send.
   * @param dst          Destination rank to send data to.
   * @param tag          Tag of send operation.
   * @param from_fpga    Set to true if the data is already on the FPGA.
   * @param stream_flags Stream flags to use.
   * @param run_async    Run the ACCL call asynchronously.
   * @param waitfor      ACCL call will wait for these operations before it will
   *                     start.
   * @return CCLO*       CCLO object that can be passed to waitfor;
   *                     nullptr if run_async is false.
   */
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

  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(dtype *host_buffer,
                                               size_t length, dataType type,
                                               unsigned mem_grp) {
    if (sim_mode) {
      return std::unique_ptr<Buffer<dtype>>(
          new SimBuffer<dtype>(host_buffer, length, type,
                               static_cast<SimDevice *>(cclo)->get_socket()));
    }
#ifdef ACCL_HARDWARE_SUPPORT
    else {
      return std::unique_ptr<Buffer<dtype>>(new FPGABuffer<dtype>(
          host_buffer, length, type, device, (xrt::memory_group)mem_grp));
    }
#endif
    return std::unique_ptr<Buffer<dtype>>(nullptr);
  }

  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(size_t length, dataType type,
                                               unsigned mem_grp) {
    if (sim_mode) {
      return std::unique_ptr<Buffer<dtype>>(new SimBuffer<dtype>(
          length, type, static_cast<SimDevice *>(cclo)->get_socket()));
    }
#ifdef ACCL_HARDWARE_SUPPORT
    else {
      return std::unique_ptr<Buffer<dtype>>(new FPGABuffer<dtype>(
          length, type, device, (xrt::memory_group)mem_grp));
    }
#endif
    return std::unique_ptr<Buffer<dtype>>(nullptr);
  }

  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(dtype *host_buffer,
                                               size_t length, dataType type) {
    return create_buffer(host_buffer, length, type, devicemem);
  }

  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(size_t length, dataType type) {
    return create_buffer<dtype>(length, type, devicemem);
  }

  std::string dump_exchange_memory();

  std::string dump_rx_buffers(size_t nbufs);
  std::string dump_rx_buffers() {
    if (cclo->read(rx_buffers_adr) != rx_buffer_spares.size()) {
      throw std::runtime_error("CCLO inconsistent");
    }
    return dump_rx_buffers(rx_buffer_spares.size());
  }

  std::string dump_communicator();

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
#ifdef ACCL_HARDWARE_SUPPORT
  xrt::device device;
#endif

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
};

} // namespace ACCL
