/*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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
# *******************************************************************************/

#pragma once

#include "xlnx-comm.hpp"

#include "experimental/xrt_aie.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <uuid/uuid.h>
#include <vector>

// XXX Implement all supported ops

enum fgFunc {
	enable_irq  = 0,
    disable_irq = 1,
    reset_periph= 2,
    enable_pkt  = 3,
    set_timeout = 4,
    init_connection = 5,
    open_port       = 6,
    open_con        = 7,
    use_tcp_stack   = 8,
    use_udp_stack   = 9,
    start_profiling = 10,
    end_profiling   = 11,
    set_dma_transaction_size = 12
};

enum operation_t {
	config                  = 0,
    sendop                  = 1,
    recvop                  = 2,
    bcast                   = 3,
    scatter                 = 4,
    gather                  = 5,
    reduce                  = 6,
    allgather               = 7,
    allreduce               = 8,
    accumulate              = 9,
    copy                    = 10,
    reduce_ring             = 11,
    allreduce_fused_ring    = 12,
    gather_ring             = 13,
    allgather_ring          = 14,
    ext_stream_krnl         = 15,
    ext_reduce              = 16,
    bcast_rr                = 17,
    scatter_rr              = 18,
    allreduce_share_ring    = 19,
    nop                     = 255
};

enum network_protocol_t { TCP, UDP, ROCE };

enum mode { DUAL = 2, QUAD = 4 };

class FPGA {

private:
  std::vector<std::vector<int8_t>> _host_bufs;
  int _nbufs;
  int _buffer_size;
  std::vector<xrt::bo> _rx_buffers;
  xrt::device _device;
  xrt::kernel _krnl;
  communicator _comm;
  const uint64_t _base_addr = 0x800;
  uint64_t _comm_addr = 0;
  enum mode _mode;
  int _rank;

public:
  FPGA(unsigned int idx = 1) { get_xilinx_device(idx); }

  FPGA(int nbufs, int buffersize, unsigned int idx, enum mode m)
      : _nbufs(nbufs), _host_bufs(nbufs), _rx_buffers(nbufs),
        _buffer_size(buffersize), _mode(m) {
    get_xilinx_device(idx);
  }

  ~FPGA() {
	std::cout << "Removing CCLO object at " << std::hex <<  get_mmio_addr() << std::endl;
    execute_kernel(true, config, 0, 0, 0, reset_periph, 0, 0, 0, _rx_buffers[0], _rx_buffers[0]);
  }

  /*
   *      Get the first or idx Xilinx device
   */
  void get_xilinx_device(unsigned int idx) {
    _device = xrt::device(idx);
    // Do some checks to see if we're on the U280, error otherwise
    //	std::string name = _device.get_info<xrt::info::device::name>();
    //	if(name.find("u280") == std::string::npos) {
    //		std::cerr << "Device selected is not a U280." << std::endl;
    //		exit(-1);
    //	}
  }

  uint64_t get_mmio_addr() {
	return 0; //XXX Implement this
  }

  void config_comm(int ranks) { _comm = {ranks, _comm_addr, _krnl}; }

  void load_bitstream(const std::string xclbin) {
    char *local_rank_string = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = atoi(local_rank_string);
    //	if(local_rank==0) {
    auto uuid = _device.load_xclbin(xclbin.c_str());
    //	}
    MPI_Barrier(MPI_COMM_WORLD);
    cout << local_rank_string << endl;
    _krnl = xrt::kernel(
        _device, uuid,
        "ccl_offload:ccl_offload_inst", //+string{local_rank_string},
        xrt::kernel::cu_access_mode::exclusive);
  }

  void write_reg(uint64_t addr, uint64_t data) {
    _krnl.write_register(addr, data);
  }

  int read_reg(uint64_t addr) { return _krnl.read_register(addr); }

  template <typename... Args> void execute_kernel(bool wait, Args... args) {
    auto run = _krnl(args...);
   	run.start(); 
    if (wait) {
      run.wait();
    }
  }

  void dump_rx_buffers() {
    int64_t addr = _base_addr;
    for (int i = 0; i < _nbufs; i++) {
      std::cout << "===========================" << std::endl;
      std::cout << "Dumping spare RX buffer: " << i << std::endl;
      std::cout << "===========================" << std::endl;
      int8_t res;

      addr += 4;
      res = read_reg(addr);
      std::cout << "ADDRL: " << static_cast<int32_t>(res) << std::endl;

      addr += 4;
      res = read_reg(addr);
      std::cout << "ADDRH: " << static_cast<int32_t>(res) << std::endl;

      addr += 4;
      res = read_reg(addr);
      std::cout << "MAXSIZE: " << static_cast<int32_t>(res) << std::endl;

      addr += 4;
      res = read_reg(addr);
      std::cout << "DMA TAG: " << static_cast<int32_t>(res) << std::endl;

      addr += 4;
      res = read_reg(addr);
      std::cout << "ENQUEUED: " << static_cast<int32_t>(res) << std::endl;

      addr += 4;
      res = read_reg(addr);
      std::cout << "RX_TAG: " << static_cast<int32_t>(res) << std::endl;

      addr += 4;
      res = read_reg(addr);
      std::cout << "RESERVED: " << static_cast<int32_t>(res) << std::endl;

      addr += 4;
      res = read_reg(addr);
      std::cout << "RX_LEN: " << static_cast<int32_t>(res) << std::endl;

      addr += 4;
      res = read_reg(addr);
      std::cout << "RX_SRC: " << static_cast<int32_t>(res) << std::endl;
    }
  }

  void prep_rx_buffers(int bank_id = 1) {
    const auto SIZE = _buffer_size / sizeof(int8_t);
    int64_t addr = _base_addr;
    write_reg(addr, _nbufs);
    for (int i = 0; i < _nbufs; i++) {
      // Alloc and fill buffer
      const xrtMemoryGroup bank_grp_idx = bank_id;
      auto bo = xrt::bo(_device, _buffer_size, bank_grp_idx);
      auto hostmap = bo.map<int8_t *>();
      std::fill(hostmap, hostmap + (_buffer_size), static_cast<int8_t>(0));
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, SIZE, 0);
      _rx_buffers.insert(_rx_buffers.cbegin() + i, bo);

      // Write meta data
      addr += 4;
      write_reg(addr, bo.address() & 0xffffffff);

      addr += 4;
      write_reg(addr, (bo.address() >> 32) & 0xffffffff);

      addr += 4;
      write_reg(addr, _buffer_size);

      for (int i = 3; i < 9; i++) {
        addr += 4;
        write_reg(addr, 0);
      }
      _comm_addr = addr + 4;
      // Start irq-driven RX buffer scheduler and (de)packetizer
      execute_kernel(true, config, 0, 0, 0, enable_irq, 0, 0, 0, _rx_buffers[0],
                     _rx_buffers[0]);
      execute_kernel(true, config, 0, 0, 0, enable_pkt, 0, 0, 0, _rx_buffers[0],
                     _rx_buffers[0]);
    }
  }
  uint64_t get_retcode() { return read_reg(0xFFC); }

  uint64_t get_hwid() { return read_reg(0xFF8); }

//XXX Continue here
  void nop_op(bool run_async=false) {//, waitfor=[]) {
        auto handle = _krnl(nop, 0, 0, 0, 0, 0, 0, 0, _rx_buffers[0], _rx_buffers[0]);//, waitfor=waitfor);
        handle.start();
            handle.wait();
  }

};
