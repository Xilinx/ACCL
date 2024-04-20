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

#include "coyote_init.hpp"
#include <arpa/inet.h>
#include <iostream>

namespace {
inline void swap_endianness(uint32_t *ip) {
  uint8_t *ip_bytes = reinterpret_cast<uint8_t *>(ip);
  *ip = (ip_bytes[3] << 0) | (ip_bytes[2] << 8) | (ip_bytes[1] << 16) |
        (ip_bytes[0] << 24);
}

uint32_t _ip_encode(std::string ip) {
  struct sockaddr_in sa;
  inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr));
  swap_endianness(&sa.sin_addr.s_addr);
  return sa.sin_addr.s_addr;
}

std::string ip_decode(uint32_t ip) {
  char buffer[INET_ADDRSTRLEN];
  struct in_addr sa;
  sa.s_addr = ip;
  swap_endianness(&sa.s_addr);
  inet_ntop(AF_INET, &sa, buffer, INET_ADDRSTRLEN);
  return std::string(buffer, INET_ADDRSTRLEN);
}

void exchange_qp(unsigned int first_rank, unsigned int second_rank,
                 unsigned int local_rank,
                 std::vector<fpga::ibvQpConn *> &ibvQpConn_vec,
                 std::vector<ACCL::rank_t> &ranks) {
  // write established connection to hardware and perform arp lookup
  if (local_rank == first_rank) {
    int connection =
        (ibvQpConn_vec[second_rank]->getQpairStruct()->local.qpn & 0xFFFF) |
        ((ibvQpConn_vec[second_rank]->getQpairStruct()->remote.qpn & 0xFFFF)
         << 16);
    ibvQpConn_vec[second_rank]->setConnection(connection);
    ibvQpConn_vec[second_rank]->writeContext(ranks[second_rank].port);
    ibvQpConn_vec[second_rank]->doArpLookup();
    ranks[second_rank].session_id =
        ibvQpConn_vec[second_rank]->getQpairStruct()->local.qpn;
  } else if (local_rank == second_rank) {
    int connection =
        (ibvQpConn_vec[first_rank]->getQpairStruct()->local.qpn & 0xFFFF) |
        ((ibvQpConn_vec[first_rank]->getQpairStruct()->remote.qpn & 0xFFFF)
         << 16);
    ibvQpConn_vec[first_rank]->setConnection(connection);
    ibvQpConn_vec[first_rank]->writeContext(ranks[first_rank].port);
    ibvQpConn_vec[first_rank]->doArpLookup();
    ranks[first_rank].session_id =
        ibvQpConn_vec[first_rank]->getQpairStruct()->local.qpn;
  }
}

} // namespace

namespace coyote_init {
void setup_cyt_rdma(std::vector<fpga::ibvQpConn *> &ibvQpConn_vec,
                    std::vector<ACCL::rank_t> &ranks, int local_rank,
                    ACCL::CoyoteDevice &device) {
  std::cout << "[ACCL Coyote] Initializing QP..." << std::endl;
  // create single page dummy memory space for each qp
  uint32_t n_pages = 1;
  for (int i = 0; i < ranks.size(); i++) {
    fpga::ibvQpConn *qpConn = new fpga::ibvQpConn(
        device.coyote_qProc_vec[i], ranks[local_rank].ip, n_pages);
    ibvQpConn_vec.push_back(qpConn);
  }
}

void configure_cyt_rdma(std::vector<fpga::ibvQpConn *> &ibvQpConn_vec,
                        std::vector<ACCL::rank_t> &ranks, int local_rank) {
  std::cout << "[ACCL Coyote] Test3..." << std::endl;
  std::cout << "[ACCL Coyote] Exchanging QP..." << std::endl;
  for (int first_rank = 0; first_rank < ranks.size(); first_rank++) {
    for (int second_rank = first_rank + 1; second_rank < ranks.size();
         second_rank++) {
      exchange_qp(first_rank, second_rank, local_rank, ibvQpConn_vec, ranks);
      this_thread::sleep_for(500ms);
    }
  }

  this_thread::sleep_for(3s);
  std::cout << "[ACCL Coyote] Finished exchanging QP!" << std::endl;
}
} // namespace coyote_init
