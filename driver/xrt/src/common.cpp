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

namespace ACCL {
Json::Value arithcfg_to_json(ACCL::ArithConfig const &arithcfg) {
  Json::Value root;

  root["uncompressed_elem_bytes"] = arithcfg.uncompressed_elem_bytes;
  root["compressed_elem_bytes"] = arithcfg.compressed_elem_bytes;
  root["elem_ratio_log"] = arithcfg.elem_ratio_log;
  root["compressor_tdest"] = arithcfg.compressor_tdest;
  root["decompressor_tdest"] = arithcfg.decompressor_tdest;
  root["arith_is_compressed"] = arithcfg.arith_is_compressed;
  root["arith_tdest"] = iterable_to_json(arithcfg.arith_tdest);

  return root;
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
  cclo.write(*addr, arithcfg.arith_tdest.size()); // amount of arithmetic functions
  *addr += 4;
  cclo.write(*addr, arithcfg.arith_is_compressed);
  *addr += 4;

  for (auto &elem : arithcfg.arith_tdest) {
    cclo.write(*addr, elem);
    *addr += 4;
  }
}

void check_return_status(const Json::Value &status) {
  ACCL::val_t status_val = status.asUInt64();
  if (status_val != 0) {
    throw std::runtime_error("ZMQ call error (" + std::to_string(status_val) +
                             ")");
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

Json::Value parse_json(const std::string &raw_json) {
  Json::CharReaderBuilder builder;
  const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value json;
  reader->parse(raw_json.c_str(), raw_json.c_str() + raw_json.length(), &json,
                nullptr);
  return json;
}

void check_return_status(const zmq::message_t &reply) {
  Json::CharReaderBuilder builder;
  const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value reply_json = parse_json(reply.to_string());
  check_return_status(reply_json["status"]);
}
} // namespace ACCL
