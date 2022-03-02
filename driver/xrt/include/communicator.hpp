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
#include "constants.hpp"
#include <iostream>
#include <map>

#include <arpa/inet.h>

/** @file communicator.hpp */

namespace ACCL {

struct rank_t {
  std::string ip;
  int port;
  int session_id;
  addr_t max_segment_size;
};

class Communicator {
private:
  std::string base_ipaddr = "197.11.27.";
  int start_ip = 1;
  int _world_size;
  int _local_rank;
  int _rank;
  bool _vnx;
  std::map<int, std::string> rank_to_ip;
  addr_t _comm_addr;

public:

  addr_t comm_addr() const { return _comm_addr; }

  // communicator() {}

  // communicator(int world_size, uint64_t comm_addr, xrt::kernel krnl,
  //              bool vnx = false)
  //     : _world_size(world_size), _comm_addr(comm_addr), _vnx(vnx) {

  //   MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  //   char *local_rank_string = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  //   _local_rank = atoi(local_rank_string);

  //   uint64_t addr = _comm_addr;

  //   krnl.write_register(addr, world_size);
  //   addr += 4;
  //   krnl.write_register(addr, _local_rank);
  //   for (int i = 0; i < _world_size; i++) {
  //     string ip = base_ipaddr + to_string(i + start_ip);
  //     rank_to_ip.insert(pair<int, string>(_rank, ip));
  //     addr += 4;
  //     krnl.write_register(addr, ip_encode(ip_from_rank(i)));
  //     addr += 4;
  //     if (_vnx) {
  //       krnl.write_register(addr, i);
  //     } else {
  //       krnl.write_register(addr, port_from_rank(i));
  //     }
  //   }
  //   //  self.communicators.append(communicator)
  // }

  // int port_from_rank(int rank) {
  //   throw std::logic_error("Function not yet implemented");
  //   return 0;
  // }

  // uint32_t ip_encode(string ip) { return inet_addr(ip.c_str()); }

  // string ip_from_rank(int rank) { return rank_to_ip[rank]; }
};
} // namespace ACCL
