#include "accl.hpp"
#include <jsoncpp/json/json.h>
#include <math.h>

/***************************************************
 *                HELP FUNCTIONS                   *
 ***************************************************/
template <typename Iterable>
Json::Value iterable_to_json(Iterable const &iter) {
  Json::Value list;

  for (auto &&value : iter) {
    list.append(value);
  }

  return list;
}

Json::Value arithcfg_to_json(ACCL::arithConfig const &arithcfg) {
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

void check_return_status(const Json::Value &status) {
  ACCL::val_t status_val = status.asUInt64();
  if (status_val != 0) {
    throw std::runtime_error("ZMQ call error (" + std::to_string(status_val) +
                             ")");
  }
}

void check_return_status(const zmq::message_t &reply) {
  Json::Value reply_json(reply.to_string());
  check_return_status(reply_json["status"]);
}

namespace ACCL {
/***************************************************
 *                    SimMMIO                      *
 ***************************************************/
// MMIO read request  {"type": 0, "addr": <uint>}
// MMIO read response {"status": OK|ERR, "rdata": <uint>}
val_t SimMMIO::read(addr_t offset) {
  Json::Value request_json;
  request_json["type"] = 0;
  request_json["addr"] = (Json::Value::UInt64)offset;
  std::string request = request_json.asString();
  this->socket->send(request.c_str(), request.size());

  zmq::message_t reply;
  this->socket->recv(reply);
  Json::Value reply_json(reply.to_string());
  check_return_status(reply_json["status"]);
  return reply_json["rdata"].asUInt64();
}

// MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
// MMIO write response {"status": OK|ERR}
void SimMMIO::write(addr_t offset, val_t val) {
  Json::Value request_json;
  request_json["type"] = 1;
  request_json["addr"] = (Json::Value::UInt64)offset;
  request_json["wdata"] = (Json::Value::UInt64)val;
  std::string request = request_json.asString();
  this->socket->send(request.c_str(), request.size());

  zmq::message_t reply;
  this->socket->recv(reply);
  check_return_status(reply);
}

/***************************************************
 *                   SimBuffer                     *
 ***************************************************/
template <typename dtype> class SimBuffer : Buffer<dtype> {
private:
  addr_t next_free_address{0x0};
  zmq::socket_t *const socket;
  addr_t physical_address;

  addr_t get_next_free_address() {
    addr_t address = SimBuffer::next_free_address;
    // allocate on 4K boundaries
    // not sure how realistic this is, but it does help
    // work around some addressing limitations in RTLsim
    SimBuffer::next_free_address +=
        ((addr_t)ceil(this->buffer.size() / 4096.0)) * 4096;

    return address;
  }

public:
  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq::socket_t *const socket, const addr_t physical_address)
      : Buffer<dtype>(buffer, length, type), socket(socket),
        physical_address(physical_address) {}

  SimBuffer(dtype *buffer, size_t length, dataType type,
            zmq::socket_t *const socket)
      : SimBuffer(buffer, length, type, socket, this->get_next_free_address()) {
  }

  // Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>}
  // Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
  void sync_from_device() override {
    Json::Value request_json;
    request_json["type"] = 2;
    request_json["addr"] = (Json::Value::UInt64)this->physical_address;
    request_json["len"] = (Json::Value::UInt64)this->size();
    std::string request = request_json.asString();
    this->socket->send(request.c_str(), request.size());

    zmq::message_t reply;
    this->socket->recv(reply);
    Json::Value reply_json(reply.to_string());
    check_return_status(reply_json["status"]);

    size_t array_size = reply_json["rdata"].size();
    uint8_t *data = this->buffer.data();
    for (size_t i = 0; i < array_size; ++i) {
      data[i] = (uint8_t)reply_json["rdata"][(Json::ArrayIndex)i].asInt();
    }
  }

  // Devicemem write request  {"type": 3, "addr": <uint>,
  //                           "wdata": <array of uint>}
  // Devicemem write response {"status": OK|ERR}
  void sync_to_device() override {
    Json::Value request_json;
    request_json["type"] = 3;
    request_json["addr"] = (Json::Value::UInt64)this->physical_address;

    Json::Value array;
    uint8_t *data = this->buffer.data();
    for (size_t i = 0; i < this->buffer.size(); ++i) {
      array[(Json::ArrayIndex)i] = (Json::Value::Int)data[i];
    }
    request_json["wdata"] = array;
    std::string request = request_json.asString();
    this->socket->send(request.c_str(), request.size());
  }

  void free_buffer() override { return; }

  Buffer<dtype> slice(size_t start, size_t end) override {
    return SimBuffer(&this->buffer[start], end - start, this->type,
                     this->socket, this->physical_address + start);
  }
};

/***************************************************
 *                    SimDevice                    *
 ***************************************************/
SimDevice::SimDevice(std::string zmqadr) {
#ifdef ACCL_DEBUG
  std::cerr << "SimDevice connecting to ZMQ on " << zmqadr << std::endl;
#endif
  this->context = zmq::context_t();
  this->socket = zmq::socket_t(this->context, zmq::socket_type::req);
  this->socket.connect(zmqadr);
  this->mmio = new SimMMIO(&this->socket);
#ifdef ACCL_DEBUG
  std::cerr << "SimDevice connected" << std::endl;
#endif
};

void SimDevice::start(operation scenario, unsigned int count, unsigned int comm,
                      unsigned int root_src_dst, fgFunc function,
                      unsigned int tag, arithConfig arithcfg,
                      compressionFlags compression_flags,
                      streamFlags stream_flags, addr_t addr_0, addr_t addr_1,
                      addr_t addr_2, std::vector<CCLO> *waitfor) {
  if (waitfor != nullptr) {
    throw std::runtime_error("SimDevice does not support chaining");
  }

  Json::Value request_json;

  request_json["type"] = 4;
  request_json["scenario"] = scenario;
  request_json["count"] = count;
  request_json["comm"] = comm;
  request_json["root_src_dst"] = root_src_dst;
  request_json["function"] = function;
  request_json["arithcfg"] = arithcfg_to_json(arithcfg);
  request_json["compression_flags"] = compression_flags;
  request_json["stream_flags"] = stream_flags;
  request_json["addr_0"] = (Json::Value::UInt64)addr_0;
  request_json["addr_1"] = (Json::Value::UInt64)addr_1;
  request_json["addr_2"] = (Json::Value::UInt64)addr_2;

  std::string message = request_json.asString();
  zmq::message_t request(message.c_str(), message.size());
  this->socket.send(request);
};

void SimDevice::wait() {
  zmq::message_t reply;
  this->socket.recv(&reply);
  check_return_status(reply);
}

void SimDevice::call(operation scenario, unsigned int count, unsigned int comm,
                     unsigned int root_src_dst, fgFunc function,
                     unsigned int tag, arithConfig arithcfg,
                     compressionFlags compression_flags,
                     streamFlags stream_flags, addr_t addr_0, addr_t addr_1,
                     addr_t addr_2, std::vector<CCLO> *waitfor) {
  this->start(scenario, count, comm, root_src_dst, function, tag, arithcfg,
              compression_flags, stream_flags, addr_0, addr_1, addr_2, waitfor);
  this->wait();
};

val_t SimDevice::read(addr_t offset) { return this->mmio->read(offset); }

void SimDevice::write(addr_t offset, val_t val) {
  this->mmio->write(offset, val);
}

} // namespace ACCL
