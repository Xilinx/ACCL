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
#include <arpa/inet.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#ifdef ACCL_DEBUG
#include <fstream>
#endif

#define ACCL_SEND_LOG_FILE(i)                                                  \
  (std::string("accl_send") + i + std::string(".log"))
#define ACCL_LOG_FILE(i)                                                       \
  (std::string("accl") + i + std::string(".log"))

namespace {
inline void swap_endianness(uint32_t *ip) {
  uint8_t *ip_bytes = reinterpret_cast<uint8_t *>(ip);
  *ip = (ip_bytes[3] << 0) | (ip_bytes[2] << 8) | (ip_bytes[1] << 16) |
        (ip_bytes[0] << 24);
}
} // namespace

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

uint32_t ip_encode(std::string ip) {
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
  return std::string(buffer, std::strlen(buffer));
}

#ifdef ACCL_DEBUG
std::string get_rank() {
  char *rank = std::getenv("RANK");
  if (!rank) {
    char *ompi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
    if (!ompi_rank) {
      char *mpich_rank = std::getenv("PMI_RANK");
      if (!mpich_rank) {
        return "0";
      } else {
        return mpich_rank;
      }
    } else {
      return ompi_rank;
    }
  } else {
    return rank;
  }
}

void reset_log() {
  std::string rank = get_rank();
  std::string filename = ACCL_SEND_LOG_FILE(rank);
  std::ofstream outfile;
  outfile.open(filename, std::ios::out);
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
