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

#include <iostream>
#include <vector>

/** @file communicator.hpp */

namespace ACCL {
/**
 * The rank_t struct contains information ACCL needs from the ranks in the
 * network.
 *
 */
struct rank_t {
  std::string ip;          /**< IP address of rank */
  int port;                /**< Port of rank */
  int session_id;          /**< Session id of rank */
  addr_t max_segment_size; /**< Max segment size of rank */
};

/**
 * Communicators store configurations on the CCLO necessary for communicating
 * with other ranks.
 *
 */
class Communicator {
private:
  CCLO *cclo;
  std::vector<rank_t> _ranks;
  unsigned int _rank;
  addr_t _communicators_addr;

public:
  /**
   * Construct a new communicator object and store configuration on the CCLO.
   *
   * @param cclo  CCLO to store configuration on.
   * @param ranks Configuration of all ranks in the communicator.
   * @param rank  Rank of this process.
   * @param addr  Address on the CCLO to write configuration to.
   */
  Communicator(CCLO *cclo, const std::vector<rank_t> &ranks, unsigned int rank,
               addr_t *addr);

  /**
   * Retrieve the communicators address on the FPGA from this communicator.
   *
   * @return addr_t communicators address on the FPGA.
   */
  addr_t communicators_addr() const { return _communicators_addr; }

  /**
   * Retrieve the local rank stored in the communicator.
   *
   * @return unsigned int The local rank stored in the communicator.
   */
  unsigned int local_rank() const { return _rank; }

  /**
   * Get the configuration of the ranks in this communicator.
   *
   * @return const std::vector<rank_t>* Configuration of the ranks in this
   *     communicator.
   */
  const std::vector<rank_t> *get_ranks() const { return &_ranks; }

  void readback();

  /**
   * Dump the configuration of this communicator stored in the CCLO to a string.
   *
   * @return std::string A dump of the configuration of this communicator.
   */
  std::string dump();
};
} // namespace ACCL
