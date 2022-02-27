#pragma once
#include "xlnx-comm.hpp"
#include "xlnx-consts.hpp"
#include <map>
#include <string>
#include <valarray>
#include <vector>
#include <zmq.hpp>

namespace ACCL {

struct arithConfig {
  unsigned int uncompressed_elem_bytes;
  unsigned int compressed_elem_bytes;
  unsigned int elem_ratio_log;
  unsigned int compressor_tdest;
  unsigned int decompressor_tdest;
  unsigned int arith_is_compressed;
  std::vector<unsigned int> arith_tdest;
};

typedef std::map<std::pair<dataType, dataType>, arithConfig> arithConfigMap;

const arithConfigMap DEFAULT_ARITH_CONFIG = {
    {{dataType::float16, dataType::float16}, {2, 2, 0, 0, 0, 0, {4}}},
    {{dataType::float16, dataType::float32}, {4, 2, 0, 1, 1, 1, {4}}},
    {{dataType::float32, dataType::float32}, {4, 4, 0, 0, 0, 0, {0}}},
    {{dataType::float64, dataType::float64}, {8, 8, 0, 0, 0, 0, {1}}},
    {{dataType::int32, dataType::int32}, {4, 4, 0, 0, 0, 0, {2}}},
    {{dataType::int64, dataType::int64}, {8, 8, 0, 0, 0, 0, {3}}},
};

class SimMMIO {
private:
  zmq::socket_t *const socket;

public:
  SimMMIO(zmq::socket_t *const socket) : socket(socket){};

  val_t read(addr_t offset);
  void write(addr_t offset, val_t val);
};

class CCLO {
public:
  CCLO() {}
  virtual void call(operation scenario, unsigned int count, unsigned int comm,
                    unsigned int root_src_dst, fgFunc function,
                    unsigned int tag, arithConfig arithcfg,
                    compressionFlags compression_flags,
                    streamFlags stream_flags, addr_t addr_0, addr_t addr_1,
                    addr_t addr_2, std::vector<CCLO> *waitfor = nullptr) = 0;

  virtual void start(operation scenario, unsigned int count, unsigned int comm,
                     unsigned int root_src_dst, fgFunc function,
                     unsigned int tag, arithConfig arithcfg,
                     compressionFlags compression_flags,
                     streamFlags stream_flags, addr_t addr_0, addr_t addr_1,
                     addr_t addr_2, std::vector<CCLO> *waitfor = nullptr) = 0;

  virtual val_t read(addr_t offset) = 0;

  virtual void write(addr_t offset, val_t val) = 0;

  virtual void wait() = 0;
};

template <typename dtype> class Buffer {
private:
protected:
  void *const byte_array;
  dtype *const buffer;
  const size_t _length;
  const size_t _size;
  const dataType _type;

public:
  Buffer(dtype *buffer, size_t length, dataType type)
      : buffer(buffer), byte_array((void *)buffer), _length(length),
        _size(length * sizeof(dtype)), _type(type){};

  virtual void sync_from_device();

  virtual void sync_to_device();

  virtual void free_buffer();

  size_t length() { return _length; }

  size_t size() { return _size; }

  dataType type() { return _type; }

  virtual Buffer<dtype> slice(size_t start, size_t end);

  uint8_t operator[](size_t i) { return this->buffer[i]; }

  uint8_t &operator[](size_t i) const { return this->buffer[i]; }
};

class SimDevice : CCLO {
private:
  zmq::context_t context;
  zmq::socket_t socket;
  SimMMIO *mmio;

public:
  SimDevice(std::string zmqadr = "tcp://localhost:5555");

  ~SimDevice() { delete mmio; };

  void call(operation scenario, unsigned int count, unsigned int comm,
            unsigned int root_src_dst, fgFunc function, unsigned int tag,
            arithConfig arithcfg, compressionFlags compression_flags,
            streamFlags stream_flags, addr_t addr_0, addr_t addr_1,
            addr_t addr_2, std::vector<CCLO> *waitfor = nullptr) override;

  void start(operation scenario, unsigned int count, unsigned int comm,
             unsigned int root_src_dst, fgFunc function, unsigned int tag,
             arithConfig arithcfg, compressionFlags compression_flags,
             streamFlags stream_flags, addr_t addr_0, addr_t addr_1,
             addr_t addr_2, std::vector<CCLO> *waitfor = nullptr) override;

  val_t read(addr_t offset) override;

  void write(addr_t offset, val_t val) override;

  void wait() override;
};

enum networkProtocol { TCP, UDP };

class ACCL {
private:
  CCLO *cclo;
  // Supported types and corresponding arithmetic config
  const arithConfigMap &arith_config;
  addr_t arithcfg_addr;
  // RX spare buffers
  std::vector<Buffer<int8_t>> rx_buffer_spares;
  addr_t rx_buffer_size;
  addr_t rx_buffers_adr;
  // Buffers for POE
  std::vector<Buffer<int8_t>> tx_buf_network;
  std::vector<Buffer<int8_t>> rx_buf_network;
  // Spare buffer for general use
  std::vector<Buffer<int8_t>> utility_spare;
  // List of communicators, to which users will add
  std::vector<Communicator> communicators;
  addr_t communicators_addr;
  // safety checks
  bool check_return_value_flag;
  bool ignore_safety_checks;
  // TODO: use description to gather info about where to allocate spare buffers
  addr_t segment_size;
  // protocol being used
  networkProtocol protocol;
  // flag to indicate whether we've finished config
  bool config_rdy;
  // flag to indicate whether we're simulating
  bool sim_mode;
  std::string sim_sock;
  // memory banks for hardware
  Buffer<int8_t> &devicemem;
  std::vector<Buffer<int8_t>> &rxbufmem;
  Buffer<int8_t> &networkmem;

public:
  // Hardware constructor
  ACCL(const std::vector<rank_t> &ranks, int local_rank, int board_idx,
       Buffer<int8_t> &devicemem, std::vector<Buffer<int8_t>> &rxbufmem,
       Buffer<int8_t> &networkmem, networkProtocol protocol = TCP,
       int nbufs = 16, addr_t bufsize = 1024,
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);
  // Simulation constructor
  ACCL(const std::vector<rank_t> &ranks, int local_rank, std::string sim_sock,
       networkProtocol protocol = TCP, int nbufs = 16, addr_t bufsize = 1024,
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);
};

} // namespace ACCL
