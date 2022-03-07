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
#include "constants.hpp"
#include <map>
#include <stdexcept>
#include <vector>

/** @file arithconfig.hpp */

namespace ACCL {
/**
 * Arithmetic configuration needed for CCLO.
 *
 */
class ArithConfig {
public:
  const unsigned int uncompressed_elem_bytes;
  const unsigned int compressed_elem_bytes;
  const unsigned int elem_ratio_log;
  const unsigned int compressor_tdest;
  const unsigned int decompressor_tdest;
  const unsigned int arith_is_compressed;
  const std::vector<unsigned int> arith_tdest;

  /**
   * Construct a new Arithmetic Configuration object.
   *
   * TODO: Describe paramaters
   *
   * @param uncompressed_elem_bytes
   * @param compressed_elem_bytes
   * @param elem_ratio_log
   * @param compressor_tdest
   * @param decompressor_tdest
   * @param arith_is_compressed
   * @param arith_tdest
   */
  ArithConfig(unsigned int uncompressed_elem_bytes,
              unsigned int compressed_elem_bytes, unsigned int elem_ratio_log,
              unsigned int compressor_tdest, unsigned int decompressor_tdest,
              unsigned int arith_is_compressed,
              std::vector<unsigned int> arith_tdest)
      : uncompressed_elem_bytes(uncompressed_elem_bytes),
        compressed_elem_bytes(compressed_elem_bytes),
        elem_ratio_log(elem_ratio_log), compressor_tdest(compressor_tdest),
        decompressor_tdest(decompressor_tdest),
        arith_is_compressed(arith_is_compressed), arith_tdest(arith_tdest),
        exchmem_set(false) {}

  void set_exchmem(addr_t address) {
    this->exchmem_addr = address;
    this->exchmem_set = true;
  }
  /**
   * Get the address on the FPGA where the arithmetic configuration is stored.
   *
   * @return addr_t  Address on the FPGA where the arithmetic configuration is
   *                 stored.
   */
  addr_t addr() const {
    if (!this->exchmem_set) {
      std::runtime_error(
          "Arithmetic config address requested before it was set.");
    }

    return this->exchmem_addr;
  }

private:
  addr_t exchmem_addr;
  bool exchmem_set;
};

typedef std::map<std::pair<dataType, dataType>, ArithConfig> arithConfigMap;

const arithConfigMap DEFAULT_ARITH_CONFIG = {
    {{dataType::float16, dataType::float16},
     ArithConfig(2, 2, 0, 0, 0, 0, {4})},
    {{dataType::float16, dataType::float32},
     ArithConfig(4, 2, 0, 1, 1, 1, {4})},
    {{dataType::float32, dataType::float32},
     ArithConfig(4, 4, 0, 0, 0, 0, {0})},
    {{dataType::float64, dataType::float64},
     ArithConfig(8, 8, 0, 0, 0, 0, {1})},
    {{dataType::int32, dataType::int32}, ArithConfig(4, 4, 0, 0, 0, 0, {2})},
    {{dataType::int64, dataType::int64}, ArithConfig(8, 8, 0, 0, 0, 0, {3})},
};
} // namespace ACCL
