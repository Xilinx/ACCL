#include "common.hpp"

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

Json::Value parse_json(const std::string &raw_json) {
  Json::CharReaderBuilder builder;
  const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value json;
  reader->parse(raw_json.c_str(), raw_json.c_str() + raw_json.length(),
                &json, nullptr);
  return json;
}

void check_return_status(const zmq::message_t &reply) {
  Json::CharReaderBuilder builder;
  const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value reply_json = parse_json(reply.to_string());
  check_return_status(reply_json["status"]);
}
} // namespace ACCL
