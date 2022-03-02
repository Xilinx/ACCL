#pragma once
#include "cclo.hpp"
#include "constants.hpp"
#include <string>
#include <zmq.hpp>

/** @file simdevice.hpp */

namespace ACCL {
/**
 * Implementation of CCLO that uses an external CCLO simulator or emulator.
 *
 */
class SimDevice : public CCLO {
public:
  /**
   * Construct a new Simulated Device object.
   *
   * @param zmqadr  Address of simulator or emulator to connect to.
   */
  SimDevice(std::string zmqadr = "tcp://localhost:5555");

  /**
   * See ACCL::CCLO::call().
   *
   */
  void call(const Options &options) override;

  /**
   * See ACCL::CCLO::start().
   *
   */
  void start(const Options &options) override;

  val_t read(addr_t offset) override;

  void write(addr_t offset, val_t val) override;

  void wait() override;

  addr_t get_base_addr() override { return 0x0; }

  zmq::socket_t *get_socket() { return &socket; }

private:
  zmq::context_t context;
  zmq::socket_t socket;
};
} // namespace ACCL
