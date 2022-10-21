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

#include <accl.hpp>
#include <cstdlib>
#include <experimental/xrt_ip.h>
#include <functional>
#include <mpi.h>
#include <random>
#include <sstream>
#include <tclap/CmdLine.h>
#include <vector>
#include <vnx/cmac.hpp>
#include <vnx/mac.hpp>
#include <vnx/networklayer.hpp>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <fstream>
#include <arpa/inet.h>

using namespace ACCL;
using namespace vnx;

// Set the tolerance for compressed datatypes high enough, since we do currently
// not replicate the float32 -> float16 conversion for our reference results
#define FLOAT16RTOL 0.005
#define FLOAT16ATOL 0.05

#define MAX_HW_BENCH_RECORD 10
#define FREQ 200

int rank, size;
unsigned failed_tests;
unsigned skipped_tests;

struct options_t {
  int start_port;
  unsigned int rxbuf_size;
  unsigned int seg_size;
  unsigned int count;
  unsigned int nruns;
  unsigned int device_index;
  unsigned int num_rxbufmem;
  unsigned int test_mode;
  bool test_xrt_simulator;
  bool debug;
  bool hardware;
  bool axis3;
  bool udp;
	bool tcp;
  bool hw_bench;
  bool enableUserKernel;
  std::string xclbin;
	std::string fpgaIP;
};

struct timestamp_t {
  uint64_t cmdSeq;
  uint64_t scenario;
  uint64_t len;
  uint64_t comm;
  uint64_t root_src_dst;
  uint64_t function;
  uint64_t msg_tag;
  uint64_t datapath_cfg;
  uint64_t compression_flags;
  uint64_t stream_flags;
  uint64_t addra_l;
  uint64_t addra_h;
  uint64_t addrb_l;
  uint64_t addrb_h;
  uint64_t addrc_l;
  uint64_t addrc_h;
  uint64_t cmdTimestamp;
  uint64_t cmdEnd;
  uint64_t stsSeq;
  uint64_t sts;
  uint64_t stsTimestamp;
  uint64_t stsEnd;
};

//******************************
//**  XCC Operations          **
//******************************
//Housekeeping
#define ACCL_CONFIG         0
//Primitives
#define ACCL_COPY           1
#define ACCL_COMBINE        2
#define ACCL_SEND           3
#define ACCL_RECV           4
//Collectives
#define ACCL_BCAST          5
#define ACCL_SCATTER        6
#define ACCL_GATHER         7
#define ACCL_REDUCE         8
#define ACCL_ALLGATHER      9
#define ACCL_ALLREDUCE      10
#define ACCL_REDUCE_SCATTER 11
#define ACCL_BARRIER        12
#define ACCL_ALLTOALL       13

//ACCL_CONFIG SUBFUNCTIONS
#define HOUSEKEEP_SWRST                0
#define HOUSEKEEP_PKTEN                1
#define HOUSEKEEP_TIMEOUT              2
#define HOUSEKEEP_OPEN_PORT            3
#define HOUSEKEEP_OPEN_CON             4
#define HOUSEKEEP_SET_STACK_TYPE       5
#define HOUSEKEEP_SET_MAX_SEGMENT_SIZE 6
#define HOUSEKEEP_CLOSE_CON            7

std::string format_log(std::string collective, options_t options, double time, double tput){
  std::string log_str = collective+","+std::to_string(size)+","+std::to_string(rank)+","+std::to_string(options.num_rxbufmem)+","+std::to_string(options.count*sizeof(float))+","+std::to_string(options.rxbuf_size)+","+std::to_string(options.rxbuf_size)+","+std::to_string(time)+","+std::to_string(tput);
  return log_str;
}

timestamp_t readTimeStamp(uint64_t* host_ptr_hw_bench_cmd, uint64_t* host_ptr_hw_bench_sts, unsigned int& cmd_mem_offset, unsigned int& sts_mem_offset)
{
  timestamp_t timestamp;

  timestamp.cmdSeq = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.scenario = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.len = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.comm = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.root_src_dst = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.function = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.msg_tag = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.datapath_cfg = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.compression_flags = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.stream_flags = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addra_l = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addra_h = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addrb_l = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addrb_h = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addrc_l = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addrc_h = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.cmdTimestamp = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.cmdEnd = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;

  timestamp.stsSeq = (uint64_t)host_ptr_hw_bench_sts[sts_mem_offset];
  sts_mem_offset++;
  timestamp.sts = (uint64_t)host_ptr_hw_bench_sts[sts_mem_offset];
  sts_mem_offset++;
  timestamp.stsTimestamp = (uint64_t)host_ptr_hw_bench_sts[sts_mem_offset];
  sts_mem_offset++;
  timestamp.stsEnd = (uint64_t)host_ptr_hw_bench_sts[sts_mem_offset];
  sts_mem_offset++;

  return timestamp;
}

void printTimeStamp(timestamp_t timestamp, options_t &options)
{
  std::string exp;
  bool writeToLog = false;
  std::cout<<"cmdSeq: "<<timestamp.cmdSeq<<" ";
  switch (timestamp.scenario)
  {
      case ACCL_COPY:
          std::cout<<"ACCL_COPY";
          break;
      case ACCL_COMBINE:
          std::cout<<"ACCL_COMBINE";
          break;
      case ACCL_SEND:
          std::cout<<"ACCL_SEND";
          exp="sendrecv_K2K";
          writeToLog=true;
          break;
      case ACCL_RECV:
          std::cout<<"ACCL_RECV";
          exp="sendrecv_K2K";
          writeToLog=true;
          break;
      case ACCL_BCAST:
          std::cout<<"ACCL_BCAST";
          exp="bcast_K2K";
          writeToLog=true;
          break;
      case ACCL_SCATTER:
          std::cout<<"ACCL_SCATTER";
          exp="scatter_K2K";
          writeToLog=true;
          break;
      case ACCL_GATHER:
          std::cout<<"ACCL_GATHER";
          exp="allgather_K2K";
          writeToLog=true;
          break;
      case ACCL_REDUCE:
          std::cout<<"ACCL_REDUCE";
          exp="reduce_K2K";
          writeToLog=true;
          break;
      case ACCL_ALLGATHER:
          std::cout<<"ACCL_ALLGATHER";
          exp="allgather_K2K";
          writeToLog=true;
          break;
      case ACCL_REDUCE_SCATTER:
          std::cout<<"ACCL_REDUCE_SCATTER";
          break;
      case ACCL_ALLREDUCE:
          std::cout<<"ACCL_ALLREDUCE";
          exp="allreduce_K2K";
          writeToLog=true;
          break;
      case ACCL_BARRIER:
          std::cout<<"ACCL_COPY";
          break;
      case ACCL_ALLTOALL:
          std::cout<<"ACCL_ALLTOALL";
          break;
      case ACCL_CONFIG:
          std::cout<<"ACCL_CONFIG";
          switch (timestamp.function)
          {
              case HOUSEKEEP_SWRST:
                  std::cout<<" HOUSEKEEP_SWRST";
                  break;
              case HOUSEKEEP_PKTEN:
                  std::cout<<" HOUSEKEEP_PKTEN";
                  break;
              case HOUSEKEEP_TIMEOUT:
                  std::cout<<" HOUSEKEEP_TIMEOUT";
                  break;
              case HOUSEKEEP_OPEN_PORT:
                  std::cout<<" HOUSEKEEP_OPEN_PORT";
                  break;
              case HOUSEKEEP_OPEN_CON:
                  std::cout<<" HOUSEKEEP_OPEN_CON";
                  break;
              case HOUSEKEEP_CLOSE_CON:
                  std::cout<<" HOUSEKEEP_CLOSE_CON";
                  break;
              case HOUSEKEEP_SET_STACK_TYPE:
                  std::cout<<" HOUSEKEEP_SET_STACK_TYPE";
                  break;
              case HOUSEKEEP_SET_MAX_SEGMENT_SIZE:
                  std::cout<<" HOUSEKEEP_SET_MAX_SEGMENT_SIZE";
                  break;
              default:
                  std::cout<<" Not Recognized Function:"<<timestamp.function;
                  break;
          }
          break;
      default:
          std::cout<<"Not Recognized Scenario:"<<timestamp.scenario;
          writeToLog=false;
          break;
  }
  std::cout<<" len: "<<timestamp.len<<" ";
  std::cout<<" cmdTimestamp: "<<timestamp.cmdTimestamp<<" ";
  std::cout<<" cmdEnd: "<<timestamp.cmdEnd<<" ";
  std::cout<<" sts: "<<timestamp.sts<<" ";
  std::cout<<" stsTimestamp: "<<timestamp.stsTimestamp<<" ";
  std::cout<<" stsEnd: "<<timestamp.stsEnd<<" ";
  std::cout<<std::endl;

  if ((timestamp.cmdEnd != 0xFFFFFFFFFFFFFFFF) || (timestamp.stsEnd != 0xFFFFFFFFFFFFFFFF))
  {
    writeToLog=false;
  }
  if (writeToLog)
  {
    uint64_t start_cycle = timestamp.cmdTimestamp;
    uint64_t end_cycle = timestamp.stsTimestamp;
    double durationUs = (end_cycle-start_cycle)/(double)FREQ;
    double tput = (options.count*sizeof(float)*8.0)/(durationUs*1000.0); // only useful for send/recv
    accl_log(rank, format_log(exp, options, durationUs, tput));
  }

}


inline void swap_endianness(uint32_t *ip) {
  uint8_t *ip_bytes = reinterpret_cast<uint8_t *>(ip);
  *ip = (ip_bytes[3] << 0) | (ip_bytes[2] << 8) | (ip_bytes[1] << 16) |
        (ip_bytes[0] << 24);
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
  return std::string(buffer, INET_ADDRSTRLEN);
}


void test_debug(std::string message, options_t &options) {
  if (options.debug) {
    std::cerr << message << std::endl;
  }
}

void check_usage(int argc, char *argv[]) {}

std::string prepend_process() {
  return "[process " + std::to_string(rank) + "] ";
}

template <typename T>
bool is_close(T a, T b, double rtol = 1e-5, double atol = 1e-8) {
  // std::cout << abs(a - b) << " <= " << (atol + rtol * abs(b)) << "? " <<
  // (abs(a - b) <= (atol + rtol * abs(b))) << std::endl;
  return abs(a - b) <= (atol + rtol * abs(b));
}

template <typename T> static void random_array(T *data, size_t count) {
  std::uniform_real_distribution<T> distribution(-1000, 1000);
  std::mt19937 engine;
  auto generator = std::bind(distribution, engine);
  for (size_t i = 0; i < count; ++i) {
    data[i] = generator();
  }
}

template <typename T> std::unique_ptr<T> random_array(size_t count) {
  std::unique_ptr<T> data(new T[count]);
  random_array(data.get(), count);
  return data;
}

void test_copy(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start copy test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  accl.copy(*op_buf, *res_buf, count);
  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float ref = (*op_buf)[i];
    float res = (*res_buf)[i];
    if (res != ref) {
      std::cout << i + 1
                << "th item is incorrect! (" + std::to_string(res) +
                       " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << errors << " errors!" << std::endl;
  } else {
    std::cout << "Test succesfull!" << std::endl;
  }
}

void test_copy_p2p(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start copy p2p test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> p2p_buf;
  try {
    p2p_buf = accl.create_buffer_p2p<float>(count, dataType::float32);
  } catch (const std::bad_alloc &e) {
    std::cout << "Can't allocate p2p buffer (" << e.what() << "). "
              << "This probably means p2p is disabled on the FPGA.\n"
              << "Skipping p2p test..." << std::endl;
    skipped_tests += 1;
    return;
  }
  random_array(op_buf->buffer(), count);

  accl.copy(*op_buf, *p2p_buf, count);
  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float ref = (*op_buf)[i];
    float res = (*p2p_buf)[i];
    if (res != ref) {
      std::cout << i + 1
                << "th item is incorrect! (" + std::to_string(res) +
                       " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << errors << " errors!" << std::endl;
  } else {
    std::cout << "Test succesfull!" << std::endl;
  }
}

void test_combine_sum(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start combine SUM test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf1 = accl.create_buffer<float>(count, dataType::float32);
  auto op_buf2 = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf1->buffer(), count);
  random_array(op_buf2->buffer(), count);

  accl.combine(count, reduceFunction::SUM, *op_buf1, *op_buf2, *res_buf);
  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float ref = (*op_buf1)[i] + (*op_buf2)[i];
    float res = (*res_buf)[i];
    if (res != ref) {
      std::cout << i + 1
                << "th item is incorrect! (" + std::to_string(res) +
                       " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << errors << " errors!" << std::endl;
  } else {
    std::cout << "Test succesfull!" << std::endl;
  }
}

void test_combine_max(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start combine MAX test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf1 = accl.create_buffer<float>(count, dataType::float32);
  auto op_buf2 = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf1->buffer(), count);
  random_array(op_buf2->buffer(), count);

  accl.combine(count, reduceFunction::MAX, *op_buf1, *op_buf2, *res_buf);
  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float ref = ((*op_buf1)[i] > (*op_buf2)[i]) ? (*op_buf1)[i] : (*op_buf2)[i];
    float res = (*res_buf)[i];
    if (res != ref) {
      std::cout << i + 1
                << "th item is incorrect! (" + std::to_string(res) +
                       " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << errors << " errors!" << std::endl;
  } else {
    std::cout << "Test succesfull!" << std::endl;
  }
}

void test_sendrcv_bo(ACCL::ACCL &accl, xrt::device &dev, options_t &options) {
  std::cout << "Start send recv test bo..." << std::endl;
  unsigned int count = options.count;

  // Initialize bo
  float *data =
      static_cast<float *>(std::aligned_alloc(4096, count * sizeof(float)));
  float *validation_data =
      static_cast<float *>(std::aligned_alloc(4096, count * sizeof(float)));
  random_array(data, count);

  xrt::bo send_bo(dev, data, count * sizeof(float), accl.devicemem());
  xrt::bo recv_bo(dev, validation_data, count * sizeof(float),
                  accl.devicemem());
  auto op_buf = accl.create_buffer<float>(send_bo, count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(recv_bo, count, dataType::float32);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Syncing buffers...", options);
  send_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl.send(*op_buf, count, next_rank, 0, GLOBAL_COMM, true);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl.recv(*op_buf, count, prev_rank, 0, GLOBAL_COMM, true);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl.send(*op_buf, count, prev_rank, 1, GLOBAL_COMM, true);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl.recv(*op_buf, count, next_rank, 1, GLOBAL_COMM, true);

  accl.copy(*op_buf, *res_buf, count, true, true);

  recv_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float ref = validation_data[i];
    float res = data[i];
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }

  std::free(data);
  std::free(validation_data);
}

void test_sendrcv(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start send recv test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl.send(*op_buf, count, next_rank, 0);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl.recv(*res_buf, count, prev_rank, 0);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl.send(*res_buf, count, prev_rank, 1);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl.recv(*res_buf, count, next_rank, 1);

  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i];
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}



void test_sendrcv_bench(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start send recv H2H test with 2 ranks..." << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  double tput = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  if (rank == 0)
  {
    test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
    accl.send(*op_buf, count, next_rank, 0);
  } else if (rank == 1)
  {
    test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
    accl.recv(*res_buf, count, prev_rank, 0);
  }
  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
  tput = (options.count*sizeof(float)*8.0)/(durationUs*1000.0);

  if (rank == 0 || rank == 1){
    accl_log(rank, format_log("sendrecv_H2H", options, durationUs, tput));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Start send recv F2F test with 2 ranks..." << std::endl;
  start = std::chrono::high_resolution_clock::now();
  if (rank == 0)
  {
    test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
    accl.send(*op_buf, count, next_rank, 0, GLOBAL_COMM, true);
  } else if (rank == 1)
  {
    test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
    accl.recv(*res_buf, count, prev_rank, 0, GLOBAL_COMM, true);
  }
  end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
  tput = (options.count*sizeof(float)*8.0)/(durationUs*1000.0);

  if (rank == 0 || rank == 1){
    accl_log(rank, format_log("sendrecv_F2F", options, durationUs, tput));
  }
}

void test_sendrcv_stream(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start streaming send recv test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Sending data on " + std::to_string(rank) + " to " +
             std::to_string(next_rank) + "...", options);
  accl.send(*op_buf, count, next_rank, 0);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
             std::to_string(prev_rank) + "...", options);
  accl.recv(dataType::float32, count, prev_rank, 0, GLOBAL_COMM);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
             std::to_string(prev_rank) + "...", options);
  accl.send(dataType::float32, count, prev_rank, 1, GLOBAL_COMM);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
             std::to_string(next_rank) + "...", options);
  accl.recv(*res_buf, count, next_rank, 1);

  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i];
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_stream_put(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start stream put test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Sending data on " + std::to_string(rank) + " to stream 0 on " +
             std::to_string(next_rank) + "...", options);
  accl.stream_put(*op_buf, count, next_rank, 9);

  test_debug("Sending data on " + std::to_string(rank) + " from stream to " +
             std::to_string(prev_rank) + "...", options);
  accl.send(dataType::float32, count, prev_rank, 1, GLOBAL_COMM);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
             std::to_string(next_rank) + "...", options);
  accl.recv(*res_buf, count, next_rank, 1);

  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i];
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_sendrcv_compressed(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start send recv compression test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  int errors = 0;

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl.send(*op_buf, count, next_rank, 0, GLOBAL_COMM, false,
            dataType::float16);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl.recv(*res_buf, count, prev_rank, 0, GLOBAL_COMM, false,
            dataType::float16);

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i];
    if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl.send(*op_buf, count, prev_rank, 1, GLOBAL_COMM, false,
            dataType::float16);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl.recv(*res_buf, count, next_rank, 1, GLOBAL_COMM, false,
            dataType::float16);

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i];
    if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_bcast(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start bcast test H2H with root " + std::to_string(root) + " ..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  if (rank == root) {
    test_debug("Broadcasting data from " + std::to_string(rank) + "...",
               options);
    accl.bcast(*op_buf, count, root);
  } else {
    test_debug("Getting broadcast data from " + std::to_string(root) + "...",
               options);
    accl.bcast(*res_buf, count, root);
  }
  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("bcast_H2H", options, durationUs, 0));

  if (rank != root) {
    int errors = 0;
    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i];
      float ref = (*op_buf)[i];
      if (res != ref) {
        std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                         std::to_string(res) + " != " + std::to_string(ref) +
                         ")"
                  << std::endl;
        errors += 1;
      }
    }

    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }

  // test bcast F2F
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Start bcast test F2F with root " + std::to_string(root) + " ..."
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  if (rank == root) {
    test_debug("Broadcasting data from " + std::to_string(rank) + "...",
               options);
    accl.bcast(*op_buf, count, root, GLOBAL_COMM, true, true);
  } else {
    test_debug("Getting broadcast data from " + std::to_string(root) + "...",
               options);
    accl.bcast(*res_buf, count, root, GLOBAL_COMM, true, true);
  }
  end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("bcast_F2F", options, durationUs, 0));

}

void test_bcast_compressed(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start bcast compression test with root " +
                   std::to_string(root) + " ..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  if (rank == root) {
    test_debug("Broadcasting data from " + std::to_string(rank) + "...",
               options);
    accl.bcast(*op_buf, count, root, GLOBAL_COMM, false, false,
               dataType::float16);
  } else {
    test_debug("Getting broadcast data from " + std::to_string(root) + "...",
               options);
    accl.bcast(*res_buf, count, root, GLOBAL_COMM, false, false,
               dataType::float16);
  }

  if (rank != root) {
    int errors = 0;
    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i];
      float ref = (*op_buf)[i];
      if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
        std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                         std::to_string(res) + " != " + std::to_string(ref) +
                         ")"
                  << std::endl;
        errors += 1;
      }
    }

    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }
}

void test_scatter(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start scatter test H2H with root " + std::to_string(root) + " ..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count * size, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * size);

  test_debug("Scatter data from " + std::to_string(rank) + "...", options);

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  accl.scatter(*op_buf, *res_buf, count, root);

  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("scatter_H2H", options, durationUs, 0));

  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i + rank * count];
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Start scatter test F2F with root " + std::to_string(root) + " ..."
            << std::endl;
  durationUs = 0.0;
  start = std::chrono::high_resolution_clock::now();
  accl.scatter(*op_buf, *res_buf, count, root, GLOBAL_COMM, true, true);

  end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("scatter_F2F", options, durationUs, 0));

}

void test_scatter_compressed(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start scatter compression test with root " +
                   std::to_string(root) + " ..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count * size, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * size);

  test_debug("Scatter data from " + std::to_string(rank) + "...", options);
  accl.scatter(*op_buf, *res_buf, count, root, GLOBAL_COMM, false, false,
               dataType::float16);

  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i + rank * count];
    if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_gather(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start gather test H2H with root " + std::to_string(root) + "..."
            << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array<float>(count * size);
  auto op_buf = accl.create_buffer(host_op_buf.get() + count * rank, count,
                                   dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> res_buf;
  if (rank == root) {
    res_buf = accl.create_buffer<float>(count * size, dataType::float32);
  } else {
    res_buf = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }

  test_debug("Gather data from " + std::to_string(rank) + "...", options);

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();

  accl.gather(*op_buf, *res_buf, count, root);

  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("gather_H2H", options, durationUs, 0));

  if (rank == root) {
    int errors = 0;
    for (unsigned int i = 0; i < count * size; ++i) {
      float res = (*res_buf)[i];
      float ref = host_op_buf.get()[i];
      if (res != ref) {
        std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                         std::to_string(res) + " != " + std::to_string(ref) +
                         ")"
                  << std::endl;
        errors += 1;
      }
    }

    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }

  std::cout << "Start gather test F2F with root " + std::to_string(root) + "..."
            << std::endl;

  test_debug("Gather data from " + std::to_string(rank) + "...", options);

  MPI_Barrier(MPI_COMM_WORLD);
  durationUs = 0.0;
  start = std::chrono::high_resolution_clock::now();

  accl.gather(*op_buf, *res_buf, count, root, GLOBAL_COMM, true, true);

  end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("gather_F2F", options, durationUs, 0));

}

void test_gather_compressed(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start gather compression test with root " +
                   std::to_string(root) + "..."
            << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array<float>(count * size);
  auto op_buf = accl.create_buffer(host_op_buf.get() + count * rank, count,
                                   dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> res_buf;
  if (rank == root) {
    res_buf = accl.create_buffer<float>(count * size, dataType::float32);
  } else {
    res_buf = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }

  test_debug("Gather data from " + std::to_string(rank) + "...", options);
  accl.gather(*op_buf, *res_buf, count, root, GLOBAL_COMM, false, false,
              dataType::float16);

  if (rank == root) {
    int errors = 0;
    for (unsigned int i = 0; i < count * size; ++i) {
      float res = (*res_buf)[i];
      float ref = host_op_buf.get()[i];
      if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
        std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                         std::to_string(res) + " != " + std::to_string(ref) +
                         ")"
                  << std::endl;
        errors += 1;
      }
    }

    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }
}

void test_allgather(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start allgather H2H test..." << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array<float>(count * size);
  auto op_buf = accl.create_buffer(host_op_buf.get() + count * rank, count,
                                   dataType::float32);
  auto res_buf = accl.create_buffer<float>(count * size, dataType::float32);

  test_debug("Gathering data...", options);

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();

  accl.allgather(*op_buf, *res_buf, count);

  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("allgather_H2H", options, durationUs, 0));

  int errors = 0;
  for (unsigned int i = 0; i < count * size; ++i) {
    float res = (*res_buf)[i];
    float ref = host_op_buf.get()[i];
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }

  std::cout << "Start allgather F2F test..." << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  durationUs = 0.0;
  start = std::chrono::high_resolution_clock::now();

  accl.allgather(*op_buf, *res_buf, count, GLOBAL_COMM, true, true);

  end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("allgather_F2F", options, durationUs, 0));
}

void test_allgather_compressed(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start allgather compression test..." << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array<float>(count * size);
  auto op_buf = accl.create_buffer(host_op_buf.get() + count * rank, count,
                                   dataType::float32);
  auto res_buf = accl.create_buffer<float>(count * size, dataType::float32);

  test_debug("Gathering data...", options);
  accl.allgather(*op_buf, *res_buf, count, GLOBAL_COMM, false, false,
                 dataType::float16);

  int errors = 0;
  for (unsigned int i = 0; i < count * size; ++i) {
    float res = (*res_buf)[i];
    float ref = host_op_buf.get()[i];
    if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_allgather_comms(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start allgather test with communicators..." << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf(new float[count * size]);
  auto op_buf = accl.create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count * size, dataType::float32);

  for (unsigned int i = 0; i < count * size; i++) {
    host_op_buf.get()[i] = rank + i;
  }
  std::fill(res_buf->buffer(), res_buf->buffer() + count * size, 0);

  test_debug("Setting up communicators...", options);
  auto group = accl.get_comm_group(GLOBAL_COMM);
  unsigned int own_rank = accl.get_comm_rank(GLOBAL_COMM);
  unsigned int split = group.size() / 2;
  test_debug("Split is " + std::to_string(split), options);
  std::vector<rank_t>::iterator new_group_start;
  std::vector<rank_t>::iterator new_group_end;
  unsigned int new_rank = own_rank;
  bool is_in_lower_part = own_rank < split;
  if (is_in_lower_part) {
    new_group_start = group.begin();
    new_group_end = group.begin() + split;
  } else {
    new_group_start = group.begin() + split;
    new_group_end = group.end();
    new_rank = own_rank - split;
  }
  std::vector<rank_t> new_group(new_group_start, new_group_end);
  communicatorId new_comm = accl.create_communicator(new_group, new_rank);
  test_debug(accl.dump_communicator(), options);
  test_debug("Gathering data... count=" + std::to_string(count) +
                 ", comm=" + std::to_string(new_comm),
             options);
  accl.allgather(*op_buf, *res_buf, count, new_comm);
  test_debug("Validate data...", options);

  unsigned int data_split =
      is_in_lower_part ? count * split : count * size - count * split;
  int errors = 0;
  for (unsigned int i = 0; i < count * size; ++i) {
    float res = (*res_buf)[i];
    float ref;
    if (i < data_split) {
      ref = is_in_lower_part ? 0 : split;
      ref += (i / count) + (i % count);
    } else {
      ref = 0.0;
    }
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_multicomm(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start multi communicator test..." << std::endl;
  unsigned int count = options.count;
  auto group = accl.get_comm_group(GLOBAL_COMM);
  unsigned int own_rank = accl.get_comm_rank(GLOBAL_COMM);
  int errors = 0;
  if (group.size() < 4) {
    return;
  }
  if (own_rank == 1 || own_rank > 3) {
    return;
  }
  std::vector<rank_t> new_group;
  new_group.push_back(group[0]);
  new_group.push_back(group[2]);
  new_group.push_back(group[3]);
  unsigned int new_rank = (own_rank == 0) ? 0 : own_rank - 1;
  communicatorId new_comm = accl.create_communicator(new_group, new_rank);
  test_debug(accl.dump_communicator(), options);
  std::unique_ptr<float> host_op_buf = random_array<float>(count);
  auto op_buf = accl.create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  // start with a send/recv between ranks 0 and 2 (0 and 1 in the new
  // communicator)
  if (new_rank == 0) {
    accl.send(*op_buf, count, 1, 0, new_comm);
    accl.recv(*res_buf, count, 1, 1, new_comm);
    test_debug("Second recv completed", options);
    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i];
      float ref = host_op_buf.get()[i];
      if (res != ref) {
        std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                         std::to_string(res) + " != " + std::to_string(ref) +
                         ")"
                  << std::endl;
        errors += 1;
      }
    }
  } else if (new_rank == 1) {
    accl.recv(*res_buf, count, 0, 0, new_comm);
    test_debug("First recv completed", options);
    accl.send(*op_buf, count, 0, 1, new_comm);
  }
  std::cout << "Send/Recv with comms succeeded" << std::endl;
  // do an all-reduce on the new communicator
  for (unsigned int i = 0; i < count; ++i) {
    host_op_buf.get()[i] = i;
  }
  accl.allreduce(*op_buf, *res_buf, count, ACCL::reduceFunction::SUM, new_comm);
  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = 3 * host_op_buf.get()[i];
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }
  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_reduce(ACCL::ACCL &accl, options_t &options, int root,
                 reduceFunction function) {
  std::cout << "Start reduce test H2H with root " + std::to_string(root) +
                   " and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reduce data to " + std::to_string(root) + "...", options);

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  accl.reduce(*op_buf, *res_buf, count, root, function);

  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("reduce_H2H", options, durationUs, 0));

  if (rank == root) {
    int errors = 0;

    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i];
      float ref = (*op_buf)[i] * size;

      if (res != ref) {
        std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                         std::to_string(res) + " != " + std::to_string(ref) +
                         ")"
                  << std::endl;
        errors += 1;
      }
    }

    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }

  std::cout << "Start reduce test F2F with root " + std::to_string(root) +
                   " and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  durationUs = 0.0;
  start = std::chrono::high_resolution_clock::now();
  accl.reduce(*op_buf, *res_buf, count, root, function, GLOBAL_COMM, true, true);

  end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("reduce_F2F", options, durationUs, 0));
}

void test_reduce_compressed(ACCL::ACCL &accl, options_t &options, int root,
                            reduceFunction function) {
  std::cout << "Start reduce compression test with root " +
                   std::to_string(root) + " and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl.reduce(*op_buf, *res_buf, count, root, function, GLOBAL_COMM, false,
              false, dataType::float16);

  if (rank == root) {
    int errors = 0;

    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i];
      float ref = (*op_buf)[i] * size;

      if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
        std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                         std::to_string(res) + " != " + std::to_string(ref) +
                         ")"
                  << std::endl;
        errors += 1;
      }
    }

    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }
}

void test_reduce_scatter(ACCL::ACCL &accl, options_t &options,
                         reduceFunction function) {
  std::cout << "Start reduce scatter test and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count * size, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * size);

  test_debug("Reducing data...", options);
  accl.reduce_scatter(*op_buf, *res_buf, count, function);

  int errors = 0;
  int rank_prod = 0;

  for (int i = 0; i < rank; ++i) {
    rank_prod += i;
  }

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i + rank * count] * size;

    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_reduce_scatter_compressed(ACCL::ACCL &accl, options_t &options,
                                    reduceFunction function) {
  std::cout << "Start reduce scatter compression test and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count * size, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * size);

  test_debug("Reducing data...", options);
  accl.reduce_scatter(*op_buf, *res_buf, count, function, GLOBAL_COMM, false,
                      false, dataType::float16);

  int errors = 0;
  int rank_prod = 0;

  for (int i = 0; i < rank; ++i) {
    rank_prod += i;
  }

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i + rank * count] * size;

    if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_allreduce(ACCL::ACCL &accl, options_t &options,
                    reduceFunction function) {
  std::cout << "Start allreduce H2H test and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reducing data...", options);

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  accl.allreduce(*op_buf, *res_buf, count, function);

  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("allreduce_H2H", options, durationUs, 0));

  int errors = 0;

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i] * size;

    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }

  std::cout << "Start allreduce F2F test and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  durationUs = 0.0;
  start = std::chrono::high_resolution_clock::now();
  accl.allreduce(*op_buf, *res_buf, count, function, GLOBAL_COMM, true, true);

  end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(rank, format_log("allreduce_F2F", options, durationUs, 0));
}

void test_allreduce_compressed(ACCL::ACCL &accl, options_t &options,
                               reduceFunction function) {
  std::cout << "Start allreduce compression test and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reducing data...", options);
  accl.allreduce(*op_buf, *res_buf, count, function, GLOBAL_COMM, false, false,
                 dataType::float16);

  int errors = 0;

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i] * size;

    if (!is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL)) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }
  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_user_kernel( xrt::device &device, ACCL::ACCL &accl, options_t &options)
{
  std::cout << "Start user kernel test "
            << std::endl;

  auto user_kernel = xrt::kernel(device, device.get_xclbin_uuid(), "vadd_put:{vadd_0_0}",
                    xrt::kernel::cu_access_mode::exclusive);

  float src[options.count], dst[options.count];
  for(int i=0; i<(int)options.count; i++){
      src[i] = 1.0*(options.count*rank+i);
  }
  auto src_bo = xrt::bo(device, sizeof(float)*options.count, user_kernel.group_id(0));
  auto dst_bo = xrt::bo(device, sizeof(float)*options.count, user_kernel.group_id(1));

  src_bo.write(src);
  src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  MPI_Barrier(MPI_COMM_WORLD);
  auto run = user_kernel(src_bo, dst_bo, options.count, (rank+1)%size, accl.get_communicator_adr(),
                    accl.get_arithmetic_config_addr({dataType::float32, dataType::float32}));
  run.wait(10000);

  dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  dst_bo.read(dst);

  //check HLS function outputs
  unsigned int err_count = 0;
  for(int i=0; i<(int)options.count; i++){
      float expected = 1.0*(options.count*((rank+size-1)%size)+i) + 1;
      if(dst[i] != expected){
          err_count++;
          std::cout << "Mismatch at [" << i << "]: got " << dst[i] << " vs expected " << expected << std::endl;
      }
  }

  std::cout << "Test finished with " << err_count << " errors" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

}

void test_barrier(ACCL::ACCL &accl) {
  std::cout << "Start barrier test " << std::endl;
  accl.barrier();
  std::cout << "Test is successful!" << std::endl;
}

bool check_arp(Networklayer &network_layer, std::vector<rank_t> &ranks,
               options_t &options) {
  std::map<unsigned, bool> ranks_checked;
  for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
    ranks_checked[i] = false;
  }

  bool sanity_check = true;
  const std::map<int, std::pair<std::string, std::string>> arp =
      network_layer.read_arp_table(size);

  std::ostringstream ss_arp;
  ss_arp << "ARP table:";

  for (const std::pair<const int, std::pair<std::string, std::string>> &elem :
       arp) {
    const unsigned index = elem.first;
    const std::pair<std::string, std::string> &entry = elem.second;
    const std::string &mac = entry.first;
    const std::string &ip = entry.second;
    ss_arp << "\n(" << index << ") " << mac << ": " << ip;

    for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
      if (ranks[i].ip == ip) {
        if (ranks_checked[i]) {
          std::cout << "Double entry for " << ip << " in arp table!"
                    << std::endl;
          sanity_check = false;
        } else {
          ranks_checked[i] = true;
        }
      }
    }
  }

  test_debug(ss_arp.str(), options);

  if (!sanity_check) {
    return false;
  }

  unsigned hosts = 0;
  for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
    if (ranks_checked[i]) {
      hosts += 1;
    }
  }
  if (hosts < static_cast<unsigned>(size) - 1) {
    std::cout << "Found only " << hosts << " hosts out of " << size - 1 << "!"
              << std::endl;
    return false;
  }

  return true;
}

void configure_vnx(CMAC &cmac, Networklayer &network_layer,
                   std::vector<rank_t> &ranks, options_t &options) {
  if (ranks.size() > max_sockets_size) {
    throw std::runtime_error("Too many ranks. VNX supports up to " +
                             std::to_string(max_sockets_size) + " sockets.");
  }

  std::cout << "Testing UDP link status: ";

  const auto link_status = cmac.link_status();

  if (link_status.at("rx_status")) {
    std::cout << "Link successful!" << std::endl;
  } else {
    std::cout << "No link found." << std::endl;
  }

  std::ostringstream ss;

  ss << "Link interface 1 : {";
  for (const auto &elem : link_status) {
    ss << elem.first << ": " << elem.second << ", ";
  }
  ss << "}" << std::endl;
  test_debug(ss.str(), options);

  if (!link_status.at("rx_status")) {
    // Give time for other ranks to setup link.
    std::this_thread::sleep_for(std::chrono::seconds(3));
    exit(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "Populating socket table..." << std::endl;

  network_layer.update_ip_address(ranks[rank].ip);
  for (size_t i = 0; i < ranks.size(); ++i) {
    if (i == static_cast<size_t>(rank)) {
      continue;
    }

    network_layer.configure_socket(i, ranks[i].ip, ranks[i].port,
                                   ranks[rank].port, true);
  }

  network_layer.populate_socket_table();

  std::cout << "Starting ARP discovery..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(4));
  MPI_Barrier(MPI_COMM_WORLD);
  network_layer.arp_discovery();
  std::cout << "Finishing ARP discovery..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(2));
  MPI_Barrier(MPI_COMM_WORLD);
  network_layer.arp_discovery();
  std::cout << "ARP discovery finished!" << std::endl;

  if (!check_arp(network_layer, ranks, options)) {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    exit(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

int start_test(options_t options) {
  std::vector<rank_t> ranks = {};
  failed_tests = 0;
  skipped_tests = 0;
	std::ifstream myfile;
  myfile.open(options.fpgaIP);
	if(!myfile.is_open()) {
      perror("Error open fpgaIP file");
      exit(EXIT_FAILURE);
  }
	std::vector<std::string> ipList;
  for (int i = 0; i < size; ++i) {
    std::string ip;
    if (options.hardware && !options.axis3) {
      // ip = "10.10.10." + std::to_string(i);
			getline(myfile, ip);
			// std::cout << ip << std::endl;
			ipList.push_back(ip);
    } else {
      ip = "127.0.0.1";
    }
    rank_t new_rank = {ip, options.start_port + i, i, options.rxbuf_size};
    ranks.emplace_back(new_rank);
  }

  std::unique_ptr<ACCL::ACCL> accl;

  xrt::device device;

  if (options.hardware || options.test_xrt_simulator) {
    device = xrt::device(options.device_index);
  }

  // uint64_t *host_ptr_hw_bench_cmd;
  // xrt::bo buf_hw_bench_cmd;
  // uint64_t *host_ptr_hw_bench_sts;
  // xrt::bo buf_hw_bench_sts;

  if (options.hardware) {
    std::string cclo_id;
    if (options.axis3) {
      cclo_id = std::to_string(rank);
    } else {
      cclo_id = "0";
    }
    auto xclbin_uuid = device.load_xclbin(options.xclbin);
    auto cclo_ip = xrt::ip(device, xclbin_uuid,
                           "ccl_offload:{ccl_offload_" + cclo_id + "}");
    auto hostctrl_ip = xrt::kernel(device, xclbin_uuid,
                                   "hostctrl:{hostctrl_" + cclo_id + "_0}",
                                   xrt::kernel::cu_access_mode::exclusive);

    int devicemem;
    std::vector<int> rxbufmem;
    int networkmem;
    if (options.axis3) {
      devicemem = rank * 6;
      rxbufmem = {rank * 6 + 1};
      networkmem = rank * 6 + 2;
    } else {
      devicemem = 0;
      for (int i=0; i<(int)options.num_rxbufmem; i++)
      {
        if(i<5){
          rxbufmem.push_back(i+1);
        }
      }
      networkmem = 6;
    }

    if (options.udp) {
      auto cmac = CMAC(xrt::ip(device, xclbin_uuid, "cmac_0:{cmac_0}"));
      auto network_layer = Networklayer(
          xrt::ip(device, xclbin_uuid, "networklayer:{networklayer_0}"));

      configure_vnx(cmac, network_layer, ranks, options);
    } else if (options.tcp)
		{
			std::cout << "Configure TCP Network Kernel" << std::endl;
			auto network_krnl = xrt::kernel(device, xclbin_uuid, "network_krnl:{network_krnl_0}",
                    xrt::kernel::cu_access_mode::exclusive);
			Buffer<int8_t> *tx_buf_network = new FPGABuffer<int8_t>(64*1024*1024, dataType::int8,
                                              device, networkmem);
			Buffer<int8_t> *rx_buf_network = new FPGABuffer<int8_t>(64*1024*1024, dataType::int8,
                                              device, networkmem);
			tx_buf_network->sync_to_device();
      rx_buf_network->sync_to_device();

			uint localFPGAIP = ip_encode(ipList[rank]);
			std::cout << "rank: "<< rank << " FPGA IP: "<<std::hex << localFPGAIP << std::endl;

			network_krnl(localFPGAIP, uint(rank), localFPGAIP, tx_buf_network->bo(), rx_buf_network->bo());

      uint32_t ip_reg = network_krnl.read_register(0x010);
      uint32_t board_reg = network_krnl.read_register(0x018);
      std::cout<< std::hex << "ip_reg: "<< ip_reg << " board_reg IP: " << board_reg << std::endl;
		}

		MPI_Barrier(MPI_COMM_WORLD);

    accl = std::make_unique<ACCL::ACCL>(
        ranks, rank, device, cclo_ip, hostctrl_ip, devicemem, rxbufmem,
        options.udp || options.axis3 ? networkProtocol::UDP
                                     : networkProtocol::TCP,
        16, options.rxbuf_size, options.seg_size);

    if (options.tcp){
      debug("Starting connections to communicator ranks");
      debug("Opening ports to communicator ranks");
      accl->open_port();
      MPI_Barrier(MPI_COMM_WORLD);
      debug("Starting session to communicator ranks");
      accl->open_con();
      debug(accl->dump_communicator());
    }

    if (options.enableUserKernel)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      test_user_kernel(device, *accl, options);
    }

    // if (options.hw_bench)
    // {
    //   std::cout << "Enable hw bench kernel" << std::endl;
    //   auto hw_bench_krnl = xrt::kernel(device, xclbin_uuid, "collector:{collector_0}",xrt::kernel::cu_access_mode::exclusive);

    //   // Host Memory pointer aligned to 4K boundary
    //   posix_memalign((void**)&host_ptr_hw_bench_cmd,4096,8*1024*1024*sizeof(uint64_t));
    //   posix_memalign((void**)&host_ptr_hw_bench_sts,4096,8*1024*1024*sizeof(uint64_t));
    //   // Sample example filling the allocated host memory
    //   for(int i=0; i<8*1024*1024; i++) {
    //     host_ptr_hw_bench_cmd[i] = 0;
    //     host_ptr_hw_bench_sts[i] = 0;
    //   }
    //   buf_hw_bench_cmd = xrt::bo (device, host_ptr_hw_bench_cmd, 8*1024*1024*sizeof(uint64_t), hw_bench_krnl.group_id(1));
    //   buf_hw_bench_cmd.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    //   buf_hw_bench_sts = xrt::bo (device, host_ptr_hw_bench_sts, 8*1024*1024*sizeof(uint64_t), hw_bench_krnl.group_id(2));
    //   buf_hw_bench_sts.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //   hw_bench_krnl(MAX_HW_BENCH_RECORD, buf_hw_bench_cmd, buf_hw_bench_sts);
    //   uint32_t round_reg = hw_bench_krnl.read_register(0x010);
    //   uint32_t cmd_addr_reg = hw_bench_krnl.read_register(0x018);
    //   uint32_t sts_addr_reg = hw_bench_krnl.read_register(0x024);
    //   std::cout<< std::hex << "round_reg: "<< round_reg <<" cmd_addr_reg: "<<cmd_addr_reg<<" sts_addr_reg: "<<sts_addr_reg<<std::endl;

    // }

  } else {
    accl = std::make_unique<ACCL::ACCL>(ranks, rank, options.start_port, device,
                                        options.udp ? networkProtocol::UDP
                                                    : networkProtocol::TCP,
                                        16, options.rxbuf_size);
  }
  if (!options.udp) {
    MPI_Barrier(MPI_COMM_WORLD);
    accl->open_port();
    MPI_Barrier(MPI_COMM_WORLD);
    accl->open_con();
  }

  accl->set_timeout(1e6);

  // barrier here to make sure all the devices are configured before testing
  MPI_Barrier(MPI_COMM_WORLD);
  accl->nop();

  // Run a specific collective
  if(options.test_mode == ACCL_COPY || options.test_mode == 0)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    test_copy(*accl, options);
  }
  if(options.test_mode == ACCL_COMBINE || options.test_mode == 0)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    test_combine_sum(*accl, options);
    MPI_Barrier(MPI_COMM_WORLD);
    test_combine_max(*accl, options);
  }
  if(options.test_mode == ACCL_SEND || options.test_mode == ACCL_RECV || options.test_mode == 0)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    test_sendrcv_bench(*accl, options);
  }
  if(options.test_mode == ACCL_BCAST || options.test_mode == 0)
  {
    int root = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    test_bcast(*accl, options, root);
  }
  if(options.test_mode == ACCL_SCATTER || options.test_mode == 0)
  {
    int root = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    test_scatter(*accl, options, root);
  }
  if(options.test_mode == ACCL_GATHER || options.test_mode == 0)
  {
    int root = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    test_gather(*accl, options, root);
  }
  if(options.test_mode == ACCL_REDUCE || options.test_mode == 0)
  {
    int root = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    test_reduce(*accl, options, root, reduceFunction::SUM);
  }
  if(options.test_mode == ACCL_ALLGATHER || options.test_mode == 0)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    test_allgather(*accl, options);
  }
  if(options.test_mode == ACCL_ALLREDUCE || options.test_mode == 0)
  {
    test_allreduce(*accl, options, reduceFunction::SUM);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if(options.test_mode == ACCL_REDUCE_SCATTER || options.test_mode == 0)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    test_reduce_scatter(*accl, options, reduceFunction::SUM);
  }
  if(options.test_mode == ACCL_BARRIER || options.test_mode == 0)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    test_barrier(*accl);
  }

  // test_mode 0 runs all
  if (options.test_mode == 0)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    test_copy_p2p(*accl, options);
    if (options.test_xrt_simulator) {
      MPI_Barrier(MPI_COMM_WORLD);
      test_sendrcv_bo(*accl, device, options);
    } else {
      std::cout << "Skipping xrt::bo test. We are not running on hardware and "
                  "XCL emulation is disabled. Make sure XILINX_VITIS and "
                  "XCL_EMULATION_MODE are set."
                << std::endl;
      ++skipped_tests;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    test_sendrcv(*accl, options);
    MPI_Barrier(MPI_COMM_WORLD);
    test_sendrcv_compressed(*accl, options);
    MPI_Barrier(MPI_COMM_WORLD);
    test_stream_put(*accl, options);
    MPI_Barrier(MPI_COMM_WORLD);
    test_allgather_compressed(*accl, options);
    MPI_Barrier(MPI_COMM_WORLD);
    test_allreduce_compressed(*accl, options, reduceFunction::SUM);
    MPI_Barrier(MPI_COMM_WORLD);
    test_reduce_scatter_compressed(*accl, options, reduceFunction::SUM);
    MPI_Barrier(MPI_COMM_WORLD);
    test_multicomm(*accl, options);
    MPI_Barrier(MPI_COMM_WORLD);
    test_allgather_comms(*accl, options);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int root = 0; root < size; ++root) {

      MPI_Barrier(MPI_COMM_WORLD);
      test_bcast_compressed(*accl, options, root);
      MPI_Barrier(MPI_COMM_WORLD);
      test_scatter_compressed(*accl, options, root);
      MPI_Barrier(MPI_COMM_WORLD);
      test_gather_compressed(*accl, options, root);
      MPI_Barrier(MPI_COMM_WORLD);
      test_reduce_compressed(*accl, options, root, reduceFunction::SUM);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  std::cout << failed_tests << " tests failed on rank " << rank;
  if (skipped_tests > 0) {
    std::cout << " (skipped " << skipped_tests << " tests)";
  }
  std::cout << "." << std::endl;

  if (options.hw_bench)
  {
    std::cout << "Enable hw bench kernel" << std::endl;
    auto hw_bench_krnl = xrt::kernel(device, device.get_xclbin_uuid(), "collector:{collector_0}",xrt::kernel::cu_access_mode::exclusive);

    std::cout << "Allocate Buffer in Global Memory\n";
    auto buf_hw_bench_cmd = xrt::bo (device, 8*1024*1024*sizeof(uint64_t), hw_bench_krnl.group_id(1));
    auto buf_hw_bench_sts = xrt::bo (device, 8*1024*1024*sizeof(uint64_t), hw_bench_krnl.group_id(2));

    auto host_ptr_hw_bench_cmd = buf_hw_bench_cmd.map<uint64_t*>();
    auto host_ptr_hw_bench_sts = buf_hw_bench_sts.map<uint64_t*>();

    std::fill(host_ptr_hw_bench_cmd, host_ptr_hw_bench_cmd + 8*1024*1024, 0);
    std::fill(host_ptr_hw_bench_sts, host_ptr_hw_bench_sts + 8*1024*1024, 0);

    std::cout << "synchronize input buffer data to device global memory\n";
    buf_hw_bench_cmd.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buf_hw_bench_sts.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Execution of the kernel\n";
    auto run = hw_bench_krnl(MAX_HW_BENCH_RECORD, buf_hw_bench_cmd, buf_hw_bench_sts);
    uint32_t round_reg = hw_bench_krnl.read_register(0x010);
    uint32_t cmd_addr_reg = hw_bench_krnl.read_register(0x018);
    uint32_t sts_addr_reg = hw_bench_krnl.read_register(0x024);
    std::cout<< std::hex << "round_reg: "<< round_reg <<" cmd_addr_reg: "<<cmd_addr_reg<<" sts_addr_reg: "<<sts_addr_reg<<std::endl;

    run.wait(1000);

    std::cout <<"Sync hw_bench mem to host"<< std::endl;

    buf_hw_bench_cmd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    unsigned int cmd_mem_offset = 0;
    buf_hw_bench_sts.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    unsigned int sts_mem_offset = 0;
    for (unsigned int i = 0; i < MAX_HW_BENCH_RECORD; i++)
    {
      timestamp_t timestamp = readTimeStamp(host_ptr_hw_bench_cmd, host_ptr_hw_bench_sts, cmd_mem_offset, sts_mem_offset);
      printTimeStamp(timestamp, options);
    }

  }


  MPI_Barrier(MPI_COMM_WORLD);
  return failed_tests;
}

bool xrt_simulator_ready(const options_t &opts) {
  if (opts.hardware) {
    return true;
  }

  const char *vitis = std::getenv("XILINX_VITIS");

  if (vitis == nullptr) {
    return false;
  }

  const char *emu = std::getenv("XCL_EMULATION_MODE");
  if (emu == nullptr) {
    return false;
  }

  return std::string(emu) == "sw_emu" || std::string(emu) == "hw_emu";
}

options_t parse_options(int argc, char *argv[]) {
  try {
    TCLAP::CmdLine cmd("Test ACCL C++ driver");
    TCLAP::ValueArg<unsigned int> nruns_arg("n", "nruns",
                                            "How many times to run each test",
                                            false, 1, "positive integer");
    cmd.add(nruns_arg);
    TCLAP::ValueArg<uint16_t> start_port_arg(
        "s", "start-port", "Start of range of ports usable for sim", false, 5005,
        "positive integer");
    cmd.add(start_port_arg);
    TCLAP::ValueArg<uint32_t> count_arg("c", "count", "How many element per buffer",
                                        false, 16, "positive integer");
    cmd.add(count_arg);
    TCLAP::ValueArg<uint16_t> bufsize_arg("b", "rxbuf-size",
                                          "How many KB per RX buffer", false, 4096,
                                          "positive integer");
    cmd.add(bufsize_arg);
    TCLAP::ValueArg<uint32_t> seg_arg("g", "max_segment_size",
                                          "Maximum segmentation size in KB (should be samller than Max DMA transaction)", false, 4096,
                                          "positive integer");
    cmd.add(seg_arg);
    TCLAP::ValueArg<uint16_t> num_rxbufmem_arg ("m", "num_rxbufmem",
                                          "Number of memory banks used for rxbuf", false, 6,
                                          "positive integer");
    cmd.add(num_rxbufmem_arg);
    TCLAP::ValueArg<uint16_t> test_mode_arg ("y", "test_mode",
                                          "Test mode, by default run all the collective tests", false, 0,
                                          "integer");
    cmd.add(test_mode_arg);
    TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
    TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd,
                                  false);
    TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd,
                              false);
    TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
    TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
    TCLAP::SwitchArg hwbench_arg("z", "hwbench", "Enable hwbench, the maximum CCLO commands (~20) is limited by the FIFO depth to the bench kernel", cmd, false);
    TCLAP::SwitchArg userkernel_arg("k", "userkernel", "Enable user kernel(by default vadd kernel)", cmd, false);
    TCLAP::ValueArg<std::string> xclbin_arg(
        "x", "xclbin", "xclbin of accl driver if hardware mode is used", false,
        "accl.xclbin", "file");
    cmd.add(xclbin_arg);
    TCLAP::ValueArg<std::string> fpgaIP_arg(
        "l", "ipList", "ip list of FPGAs if hardware mode is used", false,
        "fpga", "file");
    cmd.add(fpgaIP_arg);
    TCLAP::ValueArg<uint16_t> device_index_arg(
        "i", "device-index", "device index of FPGA if hardware mode is used",
        false, 0, "positive integer");
    cmd.add(device_index_arg);
    cmd.parse(argc, argv);
    if (hardware_arg.getValue()) {
			if (axis3_arg.getValue()){
				if (udp_arg.getValue() || tcp_arg.getValue()){
					throw std::runtime_error("When using hardware axis3 mode, tcp or udp can not be used.");
				}
				std::cout << "Hardware axis3 mode" << std::endl;
			}
			if (udp_arg.getValue())
			{
				if (axis3_arg.getValue() || tcp_arg.getValue()){
					throw std::runtime_error("When using hardware udp mode, tcp or axis3 can not be used.");
				}
				std::cout << "Hardware udp mode" << std::endl;
			}
			if (tcp_arg.getValue())
			{
				if (axis3_arg.getValue() || udp_arg.getValue()){
					throw std::runtime_error("When using hardware tcp mode, udp or axis3 can not be used.");
				}
				std::cout << "Hardware tcp mode" << std::endl;
			}
      if ((axis3_arg.getValue() || udp_arg.getValue() || tcp_arg.getValue()) == false) {
        throw std::runtime_error("When using hardware, specify either axis3 or tcp or"
                                 "udp mode.");
      }
      if (hwbench_arg.getValue() && hardware_arg.getValue()==false)
      {
        throw std::runtime_error("Hardware bench mode should be set with hardware mode.");
      }
    }

    options_t opts;
    opts.start_port = start_port_arg.getValue();
    opts.count = count_arg.getValue();
    opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
    opts.seg_size = seg_arg.getValue() * 1024; // convert to bytes
    opts.num_rxbufmem = num_rxbufmem_arg.getValue();
    opts.nruns = nruns_arg.getValue();
    opts.debug = debug_arg.getValue();
    opts.hardware = hardware_arg.getValue();
    opts.axis3 = axis3_arg.getValue();
    opts.udp = udp_arg.getValue();
    opts.tcp = tcp_arg.getValue();
    opts.test_mode = test_mode_arg.getValue();
    opts.hw_bench = hwbench_arg.getValue();
    opts.enableUserKernel = userkernel_arg.getValue();
    opts.device_index = device_index_arg.getValue();
    opts.xclbin = xclbin_arg.getValue();
    opts.fpgaIP = fpgaIP_arg.getValue();
    opts.test_xrt_simulator = xrt_simulator_ready(opts);

    std::cout<<"count:"<<opts.count<<" rxbuf_size:"<<opts.rxbuf_size<<" seg_size:"<<opts.seg_size<<" num_rxbufmem:"<<opts.num_rxbufmem<<std::endl;
    return opts;

  } catch (std::exception &e) {
    if (rank == 0) {
      std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
  }


}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  options_t options = parse_options(argc, argv);

  int len;
  char name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(name, &len);

  std::ostringstream stream;
  stream << prepend_process() << "rank " << rank << " size " << size <<" "<< name
         << std::endl;
  std::cout << stream.str();

  int errors = start_test(options);

  MPI_Finalize();
  return errors;
}
