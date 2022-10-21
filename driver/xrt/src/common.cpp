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
#include <cmath>
#include <cstdlib>
#ifdef ACCL_DEBUG
#include <fstream>
#include <string> 
#endif

#define ACCL_LOG_FILE(i)                                                  \
  (std::string("accl") + i + std::string(".log"))

namespace ACCL {

void *allocate_aligned_buffer(size_t size, size_t alignment) {
  size_t aligned_size =
      ((size_t)std::ceil(size / (double)alignment)) * alignment;
  return std::aligned_alloc(alignment, aligned_size);
}

void write_arithconfig(CCLO &cclo, ArithConfig &arithcfg, addr_t *addr) {
  arithcfg.set_exchmem(*addr);

  cclo.write(*addr, arithcfg.uncompressed_elem_bytes);
  *addr += 4;
  cclo.write(*addr, arithcfg.compressed_elem_bytes);
  *addr += 4;
  cclo.write(*addr, arithcfg.elem_ratio_log);
  *addr += 4;
  cclo.write(*addr, arithcfg.compressor_tdest);
  *addr += 4;
  cclo.write(*addr, arithcfg.decompressor_tdest);
  *addr += 4;
  cclo.write(*addr,
             arithcfg.arith_tdest.size()); // amount of arithmetic functions
  *addr += 4;
  cclo.write(*addr, arithcfg.arith_is_compressed);
  *addr += 4;

  for (auto &elem : arithcfg.arith_tdest) {
    cclo.write(*addr, elem);
    *addr += 4;
  }
}

#ifdef ACCL_DEBUG

void reset_log(int rank) {
  std::string str_rank = std::to_string(rank);
  std::string filename = ACCL_LOG_FILE(str_rank);
  std::ofstream outfile;
  outfile.open(filename, std::ios::out);
  outfile<<"collective,number_of_nodes,rank_id,number_of_banks,size[B],rx_buffer_size[B],segment_size[B],execution_time[us],throughput[Gbps]"<<std::endl;
  outfile.close();
}

void accl_log(int rank, const std::string &message) {
  std::string str_rank = std::to_string(rank);
  std::string filename = ACCL_LOG_FILE(str_rank);
  std::ofstream outfile;
  outfile.open(filename, std::ios::out | std::ios_base::app);
  outfile << message << std::endl;
  outfile.close();
}
#endif

} // namespace ACCL
