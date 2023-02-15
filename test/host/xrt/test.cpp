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
#include <accl_network_utils.hpp>
#include <cstdlib>
#include <experimental/xrt_ip.h>
#include <fstream>
#include <functional>
#include <json/json.h>
#include <mpi.h>
#include <random>
#include <iostream>
#include <sstream>
#include <tclap/CmdLine.h>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>
#include <csignal>
#include <chrono>
#include <thread>

using namespace ACCL;
using namespace accl_network_utils;

// Set the tolerance for compressed datatypes high enough, since we do currently
// not replicate the float32 -> float16 conversion for our reference results
#define FLOAT16RTOL 0.005
#define FLOAT16ATOL 0.05

struct options_t {
  int start_port;
  unsigned int rxbuf_size;
  unsigned int rxbuf_count;
  unsigned int segment_size;
  unsigned int count;
  unsigned int device_index;
  bool test_xrt_simulator;
  bool debug;
  bool hardware;
  bool axis3;
  bool udp;
  bool tcp;
  bool roce;
  bool return_error;
  bool rsfec;
  std::string xclbin;
  std::string config_file;
  bool startemu;
};

int rank, size;
pid_t emulator_pid;
options_t options;
xrt::device dev;
std::unique_ptr<ACCL::ACCL> accl;

pid_t start_emulator(options_t opts) {
  // Start emulator subprocess
  pid_t pid = fork();
  if (pid == 0) {
    // we're in a child process, start emulated rank
    std::stringstream outss, errss;
    outss << "emu_rank_" << rank << "_stdout.log";
    errss << "emu_rank_" << rank << "_stderr.log";
    int outfd = open(outss.str().c_str(), O_WRONLY | O_CREAT, 0666);
    int errfd = open(errss.str().c_str(), O_WRONLY | O_CREAT, 0666);

    dup2(outfd, STDOUT_FILENO);
    dup2(errfd, STDERR_FILENO);

    char* emu_argv[] = {(char*)"../../model/emulator/cclo_emu",
                        (char*)"-s", (char*)(std::to_string(size).c_str()),
                        (char*)"-r", (char*)(std::to_string(rank).c_str()),
                        (char*)"-p", (char*)(std::to_string(opts.start_port).c_str()),
                        (char*)"-b",
                        opts.udp ? (char*)"-u" : NULL,
                        NULL};
    execvp("../../model/emulator/cclo_emu", emu_argv);
    //guard against failed execution of emulator (child will exit)
    exit(0);
  }
  return pid;
}

bool emulator_is_running(pid_t pid){
  return (kill(pid, 0) == 0);
}

void kill_emulator(pid_t pid){
  std::cout << "Stopping emulator processes" << std::endl;
  kill(pid, SIGINT);
}

void sigint_handler(int signum) {
    std::cout << "Received SIGINT signal, sending to child processes..." << std::endl;

    // Send SIGINT to all child processes.
    kill(0, SIGINT);

    // exit main process
    exit(signum);
}

void test_debug(std::string message, options_t &options) {
  if (options.debug) {
    std::cerr << message << std::endl;
  }
}

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

class TestEnvironment : public ::testing::Environment {
  public:
    // Initialise the ACCL instance.
    virtual void SetUp() {
      std::vector<rank_t> ranks;

      if (options.config_file == "") {
        ranks = generate_ranks(!options.hardware || options.axis3, rank, size,
                              options.start_port, options.rxbuf_size);
      } else {
        ranks = generate_ranks(options.config_file, rank, options.start_port,
                              options.rxbuf_size);
      }

      acclDesign design;
      if (options.axis3) {
        design = acclDesign::AXIS3x;
      } else if (options.udp) {
        design = acclDesign::UDP;
      } else if (options.tcp) {
        design = acclDesign::TCP;
      } else if (options.roce) {
        design = acclDesign::ROCE;
      }

      if (options.hardware || options.test_xrt_simulator) {
        dev = xrt::device(options.device_index);
      }

      accl = initialize_accl(
          ranks, rank, !options.hardware, design, dev, options.xclbin, options.rxbuf_count,
          options.rxbuf_size, options.segment_size, options.rsfec);
      std::cout << "Setting up TestEnvironment" << std::endl;
      accl->set_timeout(1e6);
    }

    virtual void TearDown(){
      accl.reset();
    }
};

class ACCLTest : public ::testing::Test {
protected:
    virtual void SetUp() {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    virtual void TearDown() {
      MPI_Barrier(MPI_COMM_WORLD);
    }
};

class ACCLRootTest : public ACCLTest, public testing::WithParamInterface<int> {};
class ACCLFuncTest : public ACCLTest, public testing::WithParamInterface<reduceFunction> {};
class ACCLRootFuncTest : public ACCLTest, public testing::WithParamInterface<std::tuple<int, reduceFunction>> {};

TEST_F(ACCLTest, test_copy){
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  accl->copy(*op_buf, *res_buf, count);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ((*op_buf)[i], (*res_buf)[i]);
  }
}

TEST_F(ACCLTest, test_copy_stream) {
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  accl->copy_to_stream(*op_buf, count, false);
  accl->copy_from_stream(*res_buf, count, false);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ((*op_buf)[i], (*res_buf)[i]);
  }
}

TEST_F(ACCLTest, test_copy_p2p) {
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> p2p_buf;
  try {
    p2p_buf = accl->create_buffer_p2p<float>(count, dataType::float32);
  } catch (const std::bad_alloc &e) {
    std::cout << "Can't allocate p2p buffer (" << e.what() << "). "
              << "This probably means p2p is disabled on the FPGA.\n"
              << "Skipping p2p test..." << std::endl;
    return;
  }
  random_array(op_buf->buffer(), count);

  accl->copy(*op_buf, *p2p_buf, count);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ((*op_buf)[i], (*p2p_buf)[i]);
  }
}

TEST_P(ACCLFuncTest, test_combine) {
  if((GetParam() != reduceFunction::SUM) && (GetParam() != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  auto op_buf1 = accl->create_buffer<float>(count, dataType::float32);
  auto op_buf2 = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf1->buffer(), count);
  random_array(op_buf2->buffer(), count);

  accl->combine(count, GetParam(), *op_buf1, *op_buf2, *res_buf);

  float ref, res;
  for (unsigned int i = 0; i < count; ++i) {
    if(GetParam() == reduceFunction::SUM){
      ref = (*op_buf1)[i] + (*op_buf2)[i];
      res = (*res_buf)[i];
    } else if(GetParam() == reduceFunction::MAX){
      ref = ((*op_buf1)[i] > (*op_buf2)[i]) ? (*op_buf1)[i] : (*op_buf2)[i];
      res = (*res_buf)[i];
    }
    EXPECT_EQ(ref, res);
  }
}

TEST_F(ACCLTest, test_sendrcv_bo) {

  if(!options.test_xrt_simulator) {
    GTEST_SKIP() << "Skipping xrt::bo test. We are not running on hardware and "
                 "XCL emulation is disabled. Make sure XILINX_VITIS and "
                 "XCL_EMULATION_MODE are set.";
  }
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Send recv currently doesn't support segmentation. ";
  }

  // Initialize bo
  float *data =
      static_cast<float *>(std::aligned_alloc(4096, count * sizeof(float)));
  float *validation_data =
      static_cast<float *>(std::aligned_alloc(4096, count * sizeof(float)));
  random_array(data, count);

  xrt::bo send_bo(dev, data, count * sizeof(float), accl->devicemem());
  xrt::bo recv_bo(dev, validation_data, count * sizeof(float),
                  accl->devicemem());
  auto op_buf = accl->create_buffer<float>(send_bo, count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(recv_bo, count, dataType::float32);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Syncing buffers...", options);
  send_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl->send(*op_buf, count, next_rank, 0, GLOBAL_COMM, true);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->recv(*op_buf, count, prev_rank, 0, GLOBAL_COMM, true);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->send(*op_buf, count, prev_rank, 1, GLOBAL_COMM, true);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl->recv(*op_buf, count, next_rank, 1, GLOBAL_COMM, true);

  accl->copy(*op_buf, *res_buf, count, true, true);

  recv_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ(validation_data[i], data[i]);
  }

  std::free(data);
  std::free(validation_data);
}

TEST_F(ACCLTest, test_sendrcv) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Send recv currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl->send(*op_buf, count, next_rank, 0);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->recv(*res_buf, count, prev_rank, 0);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->send(*res_buf, count, prev_rank, 1);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl->recv(*res_buf, count, next_rank, 1);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ((*res_buf)[i], (*op_buf)[i]);
  }

}

TEST_F(ACCLTest, test_sendrcv_stream) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Send recv currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Sending data on " + std::to_string(rank) + " to " +
             std::to_string(next_rank) + "...", options);
  accl->send(*op_buf, count, next_rank, 0);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
             std::to_string(prev_rank) + "...", options);
  accl->recv(dataType::float32, count, prev_rank, 0, GLOBAL_COMM);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
             std::to_string(prev_rank) + "...", options);
  accl->send(dataType::float32, count, prev_rank, 1, GLOBAL_COMM);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
             std::to_string(next_rank) + "...", options);
  accl->recv(*res_buf, count, next_rank, 1);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ((*res_buf)[i], (*op_buf)[i]);
  }

}

TEST_F(ACCLTest, test_stream_put) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Send recv currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Sending data on " + std::to_string(rank) + " to stream 0 on " +
             std::to_string(next_rank) + "...", options);
  accl->stream_put(*op_buf, count, next_rank, 9);

  test_debug("Sending data on " + std::to_string(rank) + " from stream to " +
             std::to_string(prev_rank) + "...", options);
  accl->send(dataType::float32, count, prev_rank, 1, GLOBAL_COMM);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
             std::to_string(next_rank) + "...", options);
  accl->recv(*res_buf, count, next_rank, 1);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ((*res_buf)[i], (*op_buf)[i]);
  }

}

TEST_F(ACCLTest, test_sendrcv_compressed) {

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Send recv currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl->send(*op_buf, count, next_rank, 0, GLOBAL_COMM, false,
            dataType::float16);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->recv(*res_buf, count, prev_rank, 0, GLOBAL_COMM, false,
            dataType::float16);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_TRUE(is_close((*res_buf)[i], (*op_buf)[i], FLOAT16RTOL, FLOAT16ATOL));
  }

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->send(*op_buf, count, prev_rank, 1, GLOBAL_COMM, false,
            dataType::float16);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl->recv(*res_buf, count, next_rank, 1, GLOBAL_COMM, false,
            dataType::float16);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_TRUE(is_close((*res_buf)[i], (*op_buf)[i], FLOAT16RTOL, FLOAT16ATOL));
  }

}

TEST_P(ACCLRootTest, test_bcast) {
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  int root = GetParam();
  if (rank == root) {
    test_debug("Broadcasting data from " + std::to_string(rank) + "...", options);
    accl->bcast(*op_buf, count, root);
  } else {
    test_debug("Getting broadcast data from " + std::to_string(root) + "...", options);
    accl->bcast(*res_buf, count, root);
  }

  if (rank != root) {
    for (unsigned int i = 0; i < count; ++i) {
      EXPECT_EQ((*res_buf)[i], (*op_buf)[i]);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootTest, test_bcast_compressed) {
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  int root = GetParam();
  if (rank == root) {
    test_debug("Broadcasting data from " + std::to_string(rank) + "...", options);
    accl->bcast(*op_buf, count, root, GLOBAL_COMM, false, false, dataType::float16);
  } else {
    test_debug("Getting broadcast data from " + std::to_string(root) + "...", options);
    accl->bcast(*res_buf, count, root, GLOBAL_COMM, false, false, dataType::float16);
  }

  if (rank != root) {
    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i];
      float ref = (*op_buf)[i];
      EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootTest, test_scatter) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Scatter currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count * size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * size);
  int root = GetParam();
  test_debug("Scatter data from " + std::to_string(rank) + "...", options);
  accl->scatter(*op_buf, *res_buf, count, root);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ((*res_buf)[i], (*op_buf)[i + rank * count]);
  }
}

TEST_P(ACCLRootTest, test_scatter_compressed) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Scatter currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count * size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * size);
  int root = GetParam();
  test_debug("Scatter data from " + std::to_string(rank) + "...", options);
  accl->scatter(*op_buf, *res_buf, count, root, GLOBAL_COMM, false, false, dataType::float16);

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i + rank * count];
    EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
  }
}

TEST_P(ACCLRootTest, test_gather) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Gather currently doesn't support segmentation. ";
  }
  int root = GetParam();
  std::unique_ptr<float> host_op_buf = random_array<float>(count * size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count * rank, count, dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> res_buf;
  if (rank == root) {
    res_buf = accl->create_buffer<float>(count * size, dataType::float32);
  } else {
    res_buf = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }

  test_debug("Gather data from " + std::to_string(rank) + "...", options);
  accl->gather(*op_buf, *res_buf, count, root);

  if (rank == root) {
    for (unsigned int i = 0; i < count * size; ++i) {
      EXPECT_EQ((*res_buf)[i], host_op_buf.get()[i]);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootTest, test_gather_compressed) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Gather currently doesn't support segmentation. ";
  }
  int root = GetParam();
  std::unique_ptr<float> host_op_buf = random_array<float>(count * size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count * rank, count, dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> res_buf;
  if (rank == root) {
    res_buf = accl->create_buffer<float>(count * size, dataType::float32);
  } else {
    res_buf = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }

  test_debug("Gather data from " + std::to_string(rank) + "...", options);
  accl->gather(*op_buf, *res_buf, count, root, GLOBAL_COMM, false, false, dataType::float16);

  if (rank == root) {
    for (unsigned int i = 0; i < count * size; ++i) {
      float res = (*res_buf)[i];
      float ref = host_op_buf.get()[i];
      EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_F(ACCLTest, test_allgather) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Allgather currently doesn't support segmentation. ";
  }

  std::unique_ptr<float> host_op_buf = random_array<float>(count * size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count * rank, count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count * size, dataType::float32);

  test_debug("Gathering data...", options);
  accl->allgather(*op_buf, *res_buf, count);

  for (unsigned int i = 0; i < count * size; ++i) {
    EXPECT_EQ((*res_buf)[i], host_op_buf.get()[i]);
  }
}

TEST_F(ACCLTest, test_allgather_compressed) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Allgather currently doesn't support segmentation. ";
  }

  std::unique_ptr<float> host_op_buf = random_array<float>(count * size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count * rank, count,
                                   dataType::float32);
  auto res_buf = accl->create_buffer<float>(count * size, dataType::float32);

  test_debug("Gathering data...", options);
  accl->allgather(*op_buf, *res_buf, count, GLOBAL_COMM, false, false,
                 dataType::float16);

  for (unsigned int i = 0; i < count * size; ++i) {
    EXPECT_TRUE(is_close((*res_buf)[i], host_op_buf.get()[i], FLOAT16RTOL, FLOAT16ATOL));
  }
}

TEST_F(ACCLTest, test_allgather_comms) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Allgather currently doesn't support segmentation. ";
  }

  std::unique_ptr<float> host_op_buf(new float[count * size]);
  auto op_buf = accl->create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count * size, dataType::float32);

  for (unsigned int i = 0; i < count * size; i++) {
    host_op_buf.get()[i] = rank + i;
  }
  std::fill(res_buf->buffer(), res_buf->buffer() + count * size, 0);

  test_debug("Setting up communicators...", options);
  auto group = accl->get_comm_group(GLOBAL_COMM);
  unsigned int own_rank = accl->get_comm_rank(GLOBAL_COMM);
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
  communicatorId new_comm = accl->create_communicator(new_group, new_rank);
  test_debug(accl->dump_communicator(), options);
  test_debug("Gathering data... count=" + std::to_string(count) +
                 ", comm=" + std::to_string(new_comm),
             options);
  accl->allgather(*op_buf, *res_buf, count, new_comm);
  test_debug("Validate data...", options);

  unsigned int data_split =
      is_in_lower_part ? count * split : count * size - count * split;

  for (unsigned int i = 0; i < count * size; ++i) {
    float res = (*res_buf)[i];
    float ref;
    if (i < data_split) {
      ref = is_in_lower_part ? 0 : split;
      ref += (i / count) + (i % count);
    } else {
      ref = 0.0;
    }
    EXPECT_EQ(res, ref);
  }
}

TEST_F(ACCLTest, test_multicomm) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Send currently doesn't support segmentation. ";
  }

  auto group = accl->get_comm_group(GLOBAL_COMM);
  unsigned int own_rank = accl->get_comm_rank(GLOBAL_COMM);
  int errors = 0;
  if (group.size() < 4) {
    GTEST_SKIP() << "Too few ranks. ";
  }
  if (own_rank == 1 || own_rank > 3) {
    EXPECT_TRUE(true);
  }
  std::vector<rank_t> new_group;
  new_group.push_back(group[0]);
  new_group.push_back(group[2]);
  new_group.push_back(group[3]);
  unsigned int new_rank = (own_rank == 0) ? 0 : own_rank - 1;
  communicatorId new_comm = accl->create_communicator(new_group, new_rank);
  test_debug(accl->dump_communicator(), options);
  std::unique_ptr<float> host_op_buf = random_array<float>(count);
  auto op_buf = accl->create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  // start with a send/recv between ranks 0 and 2 (0 and 1 in the new
  // communicator)
  if (new_rank == 0) {
    accl->send(*op_buf, count, 1, 0, new_comm);
    accl->recv(*res_buf, count, 1, 1, new_comm);
    test_debug("Second recv completed", options);
    for (unsigned int i = 0; i < count; ++i) {
      EXPECT_EQ((*res_buf)[i], host_op_buf.get()[i]);
    }
  } else if (new_rank == 1) {
    accl->recv(*res_buf, count, 0, 0, new_comm);
    test_debug("First recv completed", options);
    accl->send(*op_buf, count, 0, 1, new_comm);
  }

  // do an all-reduce on the new communicator
  for (unsigned int i = 0; i < count; ++i) {
    host_op_buf.get()[i] = i;
  }
  accl->allreduce(*op_buf, *res_buf, count, ACCL::reduceFunction::SUM, new_comm);
  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_EQ((*res_buf)[i], 3*host_op_buf.get()[i]);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Reduce currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(*op_buf, *res_buf, count, root, function);

  float res, ref;
  if (rank == root) {
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
      EXPECT_EQ(res, ref);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce_compressed) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Reduce currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(*op_buf, *res_buf, count, root, function, GLOBAL_COMM, false,
              false, dataType::float16);

  float res, ref;
  if (rank == root) {
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
      EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce_stream2mem) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Reduce currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Loading stream on rank" + std::to_string(rank) + "...", options);
  accl->copy_to_stream(*op_buf, count, false);
  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(dataType::float32, *res_buf, count, root, function);

  float res, ref;
  if (rank == root) {
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
      EXPECT_EQ(res, ref);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce_mem2stream) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Reduce currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  random_array(res_buf->buffer(), count);
  res_buf->sync_to_device();

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(*op_buf, dataType::float32, count, root, function);

  float res, ref;
  if (rank == root) {
    test_debug("Unloading stream on rank" + std::to_string(rank) + "...", options);
    accl->copy_from_stream(*res_buf, count, false);
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
      EXPECT_EQ(res, ref);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce_stream2stream) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Reduce currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  random_array(res_buf->buffer(), count);
  res_buf->sync_to_device();

  test_debug("Loading stream on rank" + std::to_string(rank) + "...", options);
  accl->copy_to_stream(*op_buf, count, false);
  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(dataType::float32, dataType::float32, count, root, function);

  float res, ref;
  if (rank == root) {
    test_debug("Unloading stream on rank" + std::to_string(rank) + "...", options);
    accl->copy_from_stream(*res_buf, count, false);
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
      EXPECT_EQ(res, ref);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLFuncTest, test_reduce_scatter) {
  reduceFunction function = GetParam();
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Reduce scatter currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count * size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * size);

  test_debug("Reducing data...", options);
  accl->reduce_scatter(*op_buf, *res_buf, count, function);

  float res, ref;
  for (unsigned int i = 0; i < count; ++i) {
    res = (*res_buf)[i];
    ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
    EXPECT_EQ(res, ref);
  }
}

TEST_P(ACCLFuncTest, test_reduce_scatter_compressed) {
  reduceFunction function = GetParam();
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32);
  if (count_bytes > options.segment_size) {
    GTEST_SKIP() << "Reduce scatter currently doesn't support segmentation. ";
  }

  auto op_buf = accl->create_buffer<float>(count * size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * size);

  test_debug("Reducing data...", options);
  accl->reduce_scatter(*op_buf, *res_buf, count, function, GLOBAL_COMM, false, false, dataType::float16);

  float res, ref;
  for (unsigned int i = 0; i < count; ++i) {
    res = (*res_buf)[i];
    ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
    EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
  }
}

TEST_P(ACCLFuncTest, test_allreduce) {
  reduceFunction function = GetParam();
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reducing data...", options);
  accl->allreduce(*op_buf, *res_buf, count, function);

  float res, ref;
  for (unsigned int i = 0; i < count; ++i) {
    res = (*res_buf)[i];
    ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
    EXPECT_EQ(res, ref);
  }
}

TEST_P(ACCLFuncTest, test_allreduce_compressed) {
  reduceFunction function = GetParam();
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reducing data...", options);
  accl->allreduce(*op_buf, *res_buf, count, function, GLOBAL_COMM, false, false, dataType::float16);

  float res, ref;
  for (unsigned int i = 0; i < count; ++i) {
    res = (*res_buf)[i];
    ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] * size;
    EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
  }
}

TEST_F(ACCLTest, test_barrier) {
  accl->barrier();
}

INSTANTIATE_TEST_SUITE_P(reduction_tests, ACCLFuncTest, testing::Values(reduceFunction::SUM, reduceFunction::MAX));
INSTANTIATE_TEST_SUITE_P(rooted_tests, ACCLRootTest, testing::Range(0, size));
INSTANTIATE_TEST_SUITE_P(rooted_reduction_tests, ACCLRootFuncTest, 
  testing::Combine(testing::Range(0, size), testing::Values(reduceFunction::SUM, reduceFunction::MAX))
);

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
  TCLAP::CmdLine cmd("Test ACCL C++ driver");
  TCLAP::ValueArg<uint16_t> start_port_arg(
      "p", "start-port", "Start of range of ports usable for sim", false, 5500,
      "positive integer");
  cmd.add(start_port_arg);
  TCLAP::ValueArg<unsigned int> count_arg(
      "s", "count", "How many items per test", false, 16, "positive integer");
  cmd.add(count_arg);
  TCLAP::ValueArg<unsigned int> bufsize_arg("", "rxbuf-size",
                                            "How many KB per RX buffer", false,
                                            1, "positive integer");
  cmd.add(bufsize_arg);
  TCLAP::ValueArg<unsigned int> bufcount_arg("", "rxbuf-count",
                                            "How RX buffers", false,
                                            16, "positive integer");
  cmd.add(bufcount_arg);
  TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
  TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd,
                                false);
  TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd,
                             false);
  TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
  TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
  TCLAP::SwitchArg roce_arg("r", "roce", "Use RoCE hardware setup", cmd, false);
  TCLAP::ValueArg<std::string> xclbin_arg(
      "x", "xclbin", "xclbin of accl driver if hardware mode is used", false,
      "accl.xclbin", "file");
  cmd.add(xclbin_arg);
  TCLAP::ValueArg<uint16_t> device_index_arg(
      "i", "device-index", "device index of FPGA if hardware mode is used",
      false, 0, "positive integer");
  cmd.add(device_index_arg);
  TCLAP::ValueArg<std::string> config_arg("c", "config",
                                          "Config file containing IP mapping",
                                          false, "", "JSON file");
  cmd.add(config_arg);
  TCLAP::SwitchArg rsfec_arg("", "rsfec", "Enables RS-FEC in CMAC.", cmd, false);
  TCLAP::SwitchArg startemu_arg("", "startemu", "Start emulator processes locally.", cmd, false);
  try {
    cmd.parse(argc, argv);
    if (hardware_arg.getValue()) {
      if (axis3_arg.getValue() + udp_arg.getValue() + tcp_arg.getValue() +
              roce_arg.getValue() != 1) {
        throw std::runtime_error("When using hardware, specify one of axis3, "
                                 "tcp, udp, or roce mode, but not both.");
      }
    }
  } catch (std::exception &e) {
    if (rank == 0) {
      std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
  }

  options_t opts;
  opts.start_port = start_port_arg.getValue();
  opts.count = count_arg.getValue();
  opts.rxbuf_count = bufcount_arg.getValue();
  opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
  opts.segment_size = opts.rxbuf_size;
  opts.debug = debug_arg.getValue();
  opts.hardware = hardware_arg.getValue();
  opts.axis3 = axis3_arg.getValue();
  opts.udp = udp_arg.getValue();
  opts.tcp = tcp_arg.getValue();
  opts.roce = roce_arg.getValue();
  opts.device_index = device_index_arg.getValue();
  opts.xclbin = xclbin_arg.getValue();
  opts.test_xrt_simulator = xrt_simulator_ready(opts);
  opts.config_file = config_arg.getValue();
  opts.rsfec = rsfec_arg.getValue();
  opts.startemu = startemu_arg.getValue();
  return opts;
}

int main(int argc, char *argv[]) {

  signal(SIGINT, sigint_handler);

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //init google test with any arguments specific to it
  ::testing::InitGoogleTest(&argc, argv);

  //gather ACCL options for the test
  //NOTE: this has to come before the gtest environment is initialized
  options = parse_options(argc, argv);


  if(options.startemu){
    emulator_pid = start_emulator(options);
    if(!emulator_is_running(emulator_pid)){
      std::cout << "Could not start emulator" << std::endl;
      return -1;
    }
  }

  // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
  ::testing::AddGlobalTestEnvironment(new TestEnvironment);

  std::this_thread::sleep_for(std::chrono::milliseconds(5000));

  bool fail = RUN_ALL_TESTS();
  std::cout << (fail ? "Some tests failed" : "All tests successful") << std::endl;

  if(options.startemu){
    kill_emulator(emulator_pid);
  }

  MPI_Finalize();
  return 0;
}
