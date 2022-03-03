#pragma once
#include "arithconfig.hpp"
#include "cclo.hpp"
#include "constants.hpp"
#include <iostream>
#include <jsoncpp/json/json.h>
#include <zmq.hpp>

/** @file common.hpp */

namespace ACCL {
template <typename Iterable>
inline Json::Value iterable_to_json(Iterable const &iter) {
  Json::Value list;

  for (auto &&value : iter) {
    list.append(value);
  }

  return list;
}

Json::Value arithcfg_to_json(ArithConfig const &arithcfg);

#ifdef ACCL_DEBUG
inline void debug(std::string message) { std::cerr << message << std::endl; }

inline std::string debug_hex(addr_t value) {
  std::stringstream stream;
  stream << std::hex << value;
  return stream.str();
}
#else
inline void debug(std::string message) {
  // NOP
}

inline std::string debug_hex(addr_t value) { return ""; }
#endif

/**
 * Writes the arithmetic configuration to the CCLO on the FPGA at address
 * location.
 *
 * @param cclo     CCLO object to write arithmetic configuration to.
 * @param arithcfg Arithmetic configuration to write.
 * @param addr     Address on the FPGA to write arithmetic configuration to.
 */
void write_arithconfig(CCLO &cclo, ArithConfig &arithcfg, addr_t *addr);

Json::Value parse_json(const std::string &raw_json);

void check_return_status(const Json::Value &status);

void check_return_status(const zmq::message_t &reply);

} // namespace ACCL
