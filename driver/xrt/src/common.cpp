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

#include "common.hpp"
#ifdef ACCL_DEBUG
#include <fstream>
#endif

#define ACCL_SEND_LOG_FILE(i)                                                  \
  (std::string("accl_send") + i + std::string(".log"))

namespace ACCL {

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
std::string get_rank() {
  char *ompi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
  if (!ompi_rank) {
    return "0";
  } else {
    return ompi_rank;
  }
}

void reset_log() {
  std::string rank = get_rank();
  std::string filename = ACCL_SEND_LOG_FILE(rank);
  std::ofstream outfile;
  outfile.open(filename, std::ios::out);
  outfile.close();
}

void accl_send_log(const std::string &label, const std::string &message) {
  std::string rank = get_rank();
  std::string filename = ACCL_SEND_LOG_FILE(rank);
  std::ofstream outfile;
  outfile.open(filename, std::ios::out | std::ios_base::app);
  outfile << "Json request " << label << ":" << std::endl
          << message << std::endl
          << std::endl;
  outfile.close();
}
#endif

} // namespace ACCL
