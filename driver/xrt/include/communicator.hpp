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
#include "cclo.hpp"
#include "constants.hpp"

#include <arpa/inet.h>
#include <iostream>
#include <vector>

/** @file communicator.hpp */

namespace ACCL {

struct rank_t {
  std::string ip;
  int port;
  int session_id;
  addr_t max_segment_size;
};

uint32_t ip_encode(std::string ip);
std::string ip_decode(uint32_t ip);

class Communicator {
private:
  CCLO *cclo;
  const std::vector<rank_t> _ranks;
  unsigned int _rank;
  addr_t _communicators_addr;

public:
  addr_t communicators_addr() const { return _communicators_addr; }

  Communicator(CCLO *cclo, const std::vector<rank_t> &ranks, unsigned int rank,
               addr_t *addr);

  unsigned int local_rank() const {
    return _rank;
  }

  const std::vector<rank_t> *get_ranks() const {
    return &_ranks;
  }

  std::string dump();
};
} // namespace ACCL
