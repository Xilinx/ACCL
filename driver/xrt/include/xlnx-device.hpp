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
#
*******************************************************************************/

#pragma once

#include "xlnx-comm.hpp"
#include "xlnx-consts.hpp"

#include "experimental/xrt_aie.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <uuid/uuid.h>
#include <vector>
#include <sstream>      // std::stringstream

bool compatible_size(size_t nbytes, accl_reduce_func type) {
  if (type == fp || type == i32) {
    return (nbytes % 4) == 0 ? true : false;
  } else if (type == dp || type == i64) {
    return (nbytes % 8) == 0 ? true : false;
  }
}

enum network_protocol_t { TCP, UDP, ROCE };

enum mode { DUAL = 2, QUAD = 4 };

class ACCL {

private:
  std::vector<std::vector<int8_t>> _rx_host_bufs;
  int _nbufs;
  int _rx_buffers_adr;
  int _rx_buffer_size;
  std::vector<xrt::bo> _rx_buffer_spares;
  xrt::bo _utility_spare; 
  xrt::device _device;
  xrt::kernel _krnl;
 // std::vector<communicator> _comm;
  const uint64_t _base_addr = 0x800;
  uint64_t _exchange_mem = _base_addr;
  uint64_t _comm_addr = 0;
  enum mode _mode;
  int _rank;

public:
  ACCL(unsigned int idx = 1) { get_xilinx_device(idx); }

  ACCL(int nbufs, int buffersize, unsigned int idx, enum mode m)
      : _nbufs(nbufs), _rx_host_bufs(nbufs), _rx_buffer_spares(nbufs),
        _rx_buffer_size(buffersize), _mode(m) {
    get_xilinx_device(idx);
	_exchange_mem = read_reg(_base_addr + EXCHANGE_MEM_OFFSET_ADDRESS);
  }

  ~ACCL() {
    std::cout << "Removing CCLO object at " << std::hex << get_mmio_addr()
              << std::endl;
    execute_kernel(true, config, 0, 0, 0, reset_periph, 0, 0, 0, _rx_buffer_spares[0],
                   _rx_buffer_spares[0]);
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
    return 0; // XXX Implement this
  }

  //void config_comm(int ranks) { _comm = {ranks, _comm_addr, _krnl}; }

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
        "ACCL:ccl_offload:1.0", //+string{local_rank_string},
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

	void dump_exchange_memory() {
        std::cout << "exchange mem: "<< std::endl;
        const int num_word_per_line=4;
        for(int i =0; i< EXCHANGE_MEM_ADDRESS_RANGE; i+=4*num_word_per_line){
            std::stringstream ss;
			for(int j=0; j<num_word_per_line; j++) {
             //   ss << read_register(_exchange_mem.base_addr(i+(j*4)));
			}
    //        std::cout << std::hex << _exchange_mem.base_addr + i <<  ss << endl;
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
    const auto SIZE = _rx_buffer_size / sizeof(int8_t);
    int64_t addr = _base_addr;
    write_reg(addr, _nbufs);
    for (int i = 0; i < _nbufs; i++) {
      // Alloc and fill buffer
      const xrtMemoryGroup bank_grp_idx = bank_id;
      auto bo = xrt::bo(_device, _rx_buffer_size, bank_grp_idx);
      auto hostmap = bo.map<int8_t *>();
      std::fill(hostmap, hostmap + (_rx_buffer_size), static_cast<int8_t>(0));
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, SIZE, 0);
      _rx_buffer_spares.insert(_rx_buffer_spares.cbegin() + i, bo);

      // Write meta data
      addr += 4;
      write_reg(addr, bo.address() & 0xffffffff);

      addr += 4;
      write_reg(addr, (bo.address() >> 32) & 0xffffffff);

      addr += 4;
      write_reg(addr, _rx_buffer_size);

      for (int i = 3; i < 9; i++) {
        addr += 4;
        write_reg(addr, 0);
      }
      _comm_addr = addr + 4;
      // Start irq-driven RX buffer scheduler and (de)packetizer
      execute_kernel(true, config, 0, 0, 0, enable_irq, 0, 0, 0, _rx_buffer_spares[0],
                     _rx_buffer_spares[0]);
      execute_kernel(true, config, 0, 0, 0, enable_pkt, 0, 0, 0, _rx_buffer_spares[0],
                     _rx_buffer_spares[0]);
    }
  }
  uint64_t get_retcode() { return read_reg(0xFFC); }

  uint64_t get_hwid() { return read_reg(0xFF8); }

  // XXX Continue here
  void nop_op(bool run_async = false) { //, waitfor=[]) {
    auto handle = _krnl(nop, 0, 0, 0, 0, 0, 0, 0, _rx_buffer_spares[0],
                        _rx_buffer_spares[0]); //, waitfor=waitfor);
    handle.start();
    handle.wait();
  }
};
