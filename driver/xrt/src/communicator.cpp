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

#include "accl/common.hpp"
#include "accl/communicator.hpp"
#include "accl/constants.hpp"
#include <sstream>

namespace ACCL {
Communicator::Communicator(CCLO *cclo, const std::vector<rank_t> &ranks,
                           unsigned int rank, addr_t *addr)
    : cclo(cclo), _ranks(ranks), _rank(rank), _communicators_addr(*addr) {
  cclo->write(*addr, _ranks.size());
  *addr += 4;
  cclo->write(*addr, _rank);

  for (auto &r : _ranks) {
    *addr += 4;
    cclo->write(*addr, ip_encode(r.ip));
    *addr += 4;
    cclo->write(*addr, r.port);
    // leave 2 32 bit space for inbound/outbound_seq_number
    *addr += 4;
    cclo->write(*addr, 0);
    *addr += 4;
    cclo->write(*addr, 0);
    *addr += 4;
    if (r.session_id < 0) {
      cclo->write(*addr, 0xFFFFFFFF);
    } else {
      cclo->write(*addr, r.session_id);
    }
    *addr += 4;
    cclo->write(*addr, r.max_segment_size);
  }
  *addr += 4;
}

void Communicator::readback() {
  addr_t addr = this->_communicators_addr;
  size_t nr_ranks = cclo->read(addr);
  addr += 4;
  this->_rank = cclo->read(addr);

  if (nr_ranks != _ranks.size()) {
    std::cerr << "ACCL: Number of ranks on device does not match number of "
                 "ranks specified on host!"
              << std::endl;
  }
  for (auto &r : _ranks) {
    addr += 4;
    r.ip = ip_decode(cclo->read(addr));
    addr += 4;
    r.port = cclo->read(addr);
    // leave 2 32 bit space for inbound/outbound_seq_number
    addr += 4;
    addr += 4;
    addr += 4;
    r.session_id = cclo->read(addr);
    addr += 4;
    r.max_segment_size = cclo->read(addr);
  }
}

std::string Communicator::dump() {
  std::stringstream stream;
  addr_t addr = _communicators_addr;

  size_t n = cclo->read(addr);
  addr += 4;
  size_t local_rank = cclo->read(addr);

  stream << "local rank: " << local_rank << " \t number of ranks: " << n
         << std::endl;

  for (size_t i = 0; i < n; ++i) {
    addr += 4;
    std::string ip_address = ip_decode(cclo->read(addr));
    addr += 4;
    // when using the UDP stack, write the rank number into the port register
    // the actual port is programmed into the stack itself
    val_t port = cclo->read(addr);

    // leave 2 32 bit space for inbound/outbound_seq_number
    addr += 4;
    val_t inbound_seq_number = cclo->read(addr);
    addr += 4;
    val_t outbound_seq_number = cclo->read(addr);
    // a 32 bit integer is dedicated to session id
    addr += 4;
    val_t session = cclo->read(addr);
    addr += 4;
    val_t max_seg_size = cclo->read(addr);
    stream << "> rank " << i << " (ip " << ip_address << ":" << port
           << " ; session " << session << " ; max segment size " << max_seg_size
           << ") : <- inbound seq number " << inbound_seq_number
           << ", -> outbound seq number " << outbound_seq_number << std::endl;
  }

  return stream.str();
}
} // namespace ACCL
