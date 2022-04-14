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
#include <cxxopts.hpp>
#include <functional>
#include <mpi.h>
#include <random>
#include <sstream>
#include <vector>

using namespace ACCL;

int rank, size;

struct options_t {
  int start_port;
  unsigned int rxbuf_size;
  unsigned int count;
  unsigned int nruns;
  bool debug;
};

void test_debug(std::string message, options_t &options) {
  if (options.debug) {
    std::cerr << message << std::endl;
  }
}

void check_usage(int argc, char *argv[]) {}

std::string prepend_process() {
  return "[process " + std::to_string(rank) + "] ";
}

static std::unique_ptr<float> random_array(size_t count) {
  static std::uniform_real_distribution<float> distribution(-1000, 1000);
  static std::mt19937 engine;
  static auto generator = std::bind(distribution, engine);
  std::unique_ptr<float> data(new float[count]);
  for (size_t i = 0; i < count; ++i) {
    data.get()[i] = generator();
  }

  return data;
}

void test_copy(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start copy test..." << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array(count);
  std::unique_ptr<float> res_op_buf(new float[count]);
  auto op_buf = accl.create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl.create_buffer(res_op_buf.get(), count, dataType::float32);
  (*op_buf).sync_to_device();
  (*res_buf).sync_to_device();
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

void test_combine(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start combine test..." << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf1 = random_array(count);
  std::unique_ptr<float> host_op_buf2 = random_array(count);
  std::unique_ptr<float> res_op_buf(new float[count]);
  auto op_buf1 =
      accl.create_buffer(host_op_buf1.get(), count, dataType::float32);
  auto op_buf2 =
      accl.create_buffer(host_op_buf2.get(), count, dataType::float32);
  auto res_buf = accl.create_buffer(res_op_buf.get(), count, dataType::float32);
  (*op_buf1).sync_to_device();
  (*op_buf2).sync_to_device();
  (*res_buf).sync_to_device();
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

void test_sendrcv(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start send recv test..." << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array(count);
  std::unique_ptr<float> res_op_buf(new float[count]);
  auto op_buf = accl.create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl.create_buffer(res_op_buf.get(), count, dataType::float32);
  int next_rank = (rank + 1) % size;
  int prev_rank = (rank + size - 1) % size;

  test_debug("Syncing buffers...", options);
  (*op_buf).sync_to_device();
  (*res_buf).sync_to_device();

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl.send(0, *op_buf, count, next_rank, 0);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl.recv(0, *res_buf, count, prev_rank, 0);

  test_debug("Sending data on " + std::to_string(rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl.send(0, *res_buf, count, prev_rank, 1);

  test_debug("Receiving data on " + std::to_string(rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl.recv(0, *res_buf, count, next_rank, 1);

  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float ref = (*res_buf)[i];
    float res = (*op_buf)[i];
    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_bcast(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start bcast test with root " + std::to_string(root) + " ..."
            << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array(count);
  std::unique_ptr<float> res_op_buf(new float[count]);
  auto op_buf = accl.create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl.create_buffer(res_op_buf.get(), count, dataType::float32);

  test_debug("Syncing buffers...", options);
  (*op_buf).sync_to_device();
  (*res_buf).sync_to_device();

  if (rank == root) {
    test_debug("Broadcasting data from " + std::to_string(rank) + "...",
               options);
    accl.bcast(0, *op_buf, count, root);
  } else {
    test_debug("Getting broadcast data from " + std::to_string(root) + "...",
               options);
    accl.bcast(0, *res_buf, count, root);
  }

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
      MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }
}

void test_scatter(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start scatter test with root " + std::to_string(root) + " ..."
            << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array(count * size);
  std::unique_ptr<float> res_op_buf(new float[count]);
  auto op_buf =
      accl.create_buffer(host_op_buf.get(), count * size, dataType::float32);
  auto res_buf = accl.create_buffer(res_op_buf.get(), count, dataType::float32);

  test_debug("Syncing buffers...", options);
  (*op_buf).sync_to_device();
  (*res_buf).sync_to_device();

  test_debug("Scatter data from " + std::to_string(rank) + "...", options);
  accl.scatter(0, *op_buf, *res_buf, count, root);

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
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_gather(ACCL::ACCL &accl, options_t &options, int root) {
  std::cout << "Start gather test with root " + std::to_string(root) + "..."
            << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array(count * size);
  auto op_buf = accl.create_buffer(host_op_buf.get() + count * rank, count,
                                   dataType::float32);
  std::unique_ptr<float> res_op_buf;
  std::unique_ptr<ACCL::Buffer<float>> res_buf;
  if (rank == root) {
    res_op_buf = std::unique_ptr<float>(new float[count * size]);
    res_buf =
        accl.create_buffer(res_op_buf.get(), count * size, dataType::float32);
  } else {
    res_buf = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }

  test_debug("Syncing buffers...", options);
  (*op_buf).sync_to_device();

  if (rank == root) {
    (*res_buf).sync_to_device();
  }

  test_debug("Gather data from " + std::to_string(rank) + "...", options);
  accl.gather(0, *op_buf, *res_buf, count, root);

  if (rank == root) {
    int errors = 0;
    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i + rank * count];
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
      MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }
}

void test_allgather(ACCL::ACCL &accl, options_t &options) {
  std::cout << "Start allgather test..." << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array(count * size);
  auto op_buf = accl.create_buffer(host_op_buf.get() + count * rank, count,
                                   dataType::float32);
  std::unique_ptr<float> res_op_buf =
      std::unique_ptr<float>(new float[count * size]);
  std::unique_ptr<ACCL::Buffer<float>> res_buf =
      accl.create_buffer(res_op_buf.get(), count * size, dataType::float32);
  ;

  test_debug("Syncing buffers...", options);
  (*op_buf).sync_to_device();
  (*res_buf).sync_to_device();

  test_debug("Gathering data...", options);
  accl.allgather(0, *op_buf, *res_buf, count);

  int errors = 0;
  for (unsigned int i = 0; i < count; ++i) {
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
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_reduce(ACCL::ACCL &accl, options_t &options, int root,
                 reduceFunction function) {
  std::cout << "Start reduce test with root " + std::to_string(root) +
                   " and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array(count);
  std::unique_ptr<float> res_op_buf(new float[count]);
  auto op_buf = accl.create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl.create_buffer(res_op_buf.get(), count, dataType::float32);

  test_debug("Syncing buffers...", options);
  (*op_buf).sync_to_device();
  (*res_buf).sync_to_device();

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl.reduce(0, *op_buf, *res_buf, count, root, function);

  if (rank == root) {
    int errors = 0;

    for (unsigned int i = 0; i < count; ++i) {
      int res = (*res_buf)[i];
      int ref = (*op_buf)[i] * size;

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
      MPI_Abort(MPI_COMM_WORLD, 1);
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
  std::unique_ptr<float> host_op_buf = random_array(count * size);
  std::unique_ptr<float> res_op_buf(new float[count]);
  auto op_buf =
      accl.create_buffer(host_op_buf.get(), count * size, dataType::float32);
  auto res_buf = accl.create_buffer(res_op_buf.get(), count, dataType::float32);

  test_debug("Syncing buffers...", options);
  (*op_buf).sync_to_device();
  (*res_buf).sync_to_device();

  test_debug("Reducing data...", options);
  accl.reduce_scatter(0, *op_buf, *res_buf, count, function);

  int errors = 0;
  int rank_prod = 0;

  for (int i = 0; i < rank; ++i) {
    rank_prod += i;
  }

  for (unsigned int i = 0; i < count; ++i) {
    int res = (*res_buf)[i];
    int ref = (*op_buf)[i + rank * count] * size;

    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_allreduce(ACCL::ACCL &accl, options_t &options,
                    reduceFunction function) {
  std::cout << "Start allreduce test and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  std::unique_ptr<float> host_op_buf = random_array(count);
  std::unique_ptr<float> res_op_buf(new float[count]);
  auto op_buf = accl.create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl.create_buffer(res_op_buf.get(), count, dataType::float32);

  test_debug("Syncing buffers...", options);
  (*op_buf).sync_to_device();
  (*res_buf).sync_to_device();

  test_debug("Reducing data...", options);
  accl.allreduce(0, *op_buf, *res_buf, count, function);

  int errors = 0;

  for (unsigned int i = 0; i < count; ++i) {
    int res = (*res_buf)[i];
    int ref = (*op_buf)[i] * size;

    if (res != ref) {
      std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                       std::to_string(res) + " != " + std::to_string(ref) + ")"
                << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void start_test(options_t options) {
  std::vector<rank_t> ranks = {};
  for (int i = 0; i < size; ++i) {
    rank_t new_rank = {"127.0.0.1", options.start_port + i, i,
                       options.rxbuf_size};
    ranks.emplace_back(new_rank);
  }

  ACCL::ACCL accl(ranks, rank, options.start_port,
                  networkProtocol::TCP, 16, options.rxbuf_size);
  accl.set_timeout(1e8);

  // barrier here to make sure all the devices are configured before testing
  MPI_Barrier(MPI_COMM_WORLD);
  accl.nop();
  MPI_Barrier(MPI_COMM_WORLD);
  test_copy(accl, options);
  MPI_Barrier(MPI_COMM_WORLD);
  test_combine(accl, options);
  MPI_Barrier(MPI_COMM_WORLD);
  test_sendrcv(accl, options);
  MPI_Barrier(MPI_COMM_WORLD);
  test_allgather(accl, options);
  MPI_Barrier(MPI_COMM_WORLD);
  test_allreduce(accl, options, reduceFunction::SUM);
  MPI_Barrier(MPI_COMM_WORLD);
  test_reduce_scatter(accl, options, reduceFunction::SUM);
  MPI_Barrier(MPI_COMM_WORLD);

  for (int root = 0; root < size; ++root) {
    test_bcast(accl, options, root);
    MPI_Barrier(MPI_COMM_WORLD);
    test_scatter(accl, options, root);
    MPI_Barrier(MPI_COMM_WORLD);
    test_gather(accl, options, root);
    MPI_Barrier(MPI_COMM_WORLD);
    test_reduce(accl, options, root, reduceFunction::SUM);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

options_t parse_options(int argc, char *argv[]) {
  cxxopts::Options options("test", "Test ACCL C++ driver");
  options.add_options()("n,nruns", "How many times to run each test",
                        cxxopts::value<unsigned int>()->default_value("1"))(
      "s,start-port", "Start of range of ports usable for sim",
      cxxopts::value<uint16_t>()->default_value("5500"))(
      "c,count", "how many bytes per buffer",
      cxxopts::value<unsigned int>()->default_value("16"))(
      "rxbuf-size", "How many KB per RX buffer",
      cxxopts::value<unsigned int>()->default_value("1"))(
      "d,debug", "enable debug mode")("h,help", "Print usage");
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (cxxopts::OptionException e) {
    if (rank == 0) {
      std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
  }

  if (result.count("help")) {
    if (rank == 0) {
      std::cout << options.help() << std::endl;
    }
    MPI_Finalize();
    exit(0);
  }

  options_t opts;
  opts.start_port = result["start-port"].as<uint16_t>();
  opts.count = result["count"].as<unsigned int>();
  opts.rxbuf_size =
      result["rxbuf-size"].as<unsigned int>() * 1024; // convert to bytes
  opts.nruns = result["nruns"].as<unsigned int>();
  opts.debug = result.count("debug") > 0;

  return opts;
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  options_t options = parse_options(argc, argv);

  std::ostringstream stream;
  stream << prepend_process() << "rank " << rank << " size " << size
         << std::endl;
  std::cout << stream.str();

  start_test(options);

  MPI_Finalize();
  return 0;
}
