/*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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
#include <mpi.h>
#include <sstream>
#include <vector>

int rank, size;

struct options_t {
  int start_port;
  unsigned int rxbuf_size;
  unsigned int count;
  unsigned int nruns;
  bool debug;
};

void check_usage(int argc, char *argv[]) {}

std::string prepend_process() {
  return "[process " + std::to_string(rank) + "] ";
}

void start_test(options_t options) {
  std::vector<ACCL::rank_t> ranks = {};
  for (int i = 0; i < size; ++i) {
    ACCL::rank_t new_rank = {"127.0.0.1", options.start_port + i, i,
                             options.rxbuf_size};
    ranks.emplace_back(new_rank);
  }

  ACCL::ACCL accl(ranks, rank,
                  "tcp://localhost:" +
                      std::to_string(options.start_port + rank));
  accl.set_timeout(1e8);

  // barrier here to make sure all the devices are configured before testing
  MPI_Barrier(MPI_COMM_WORLD);

  accl.nop();
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
  opts.rxbuf_size = result["rxbuf-size"].as<unsigned int>();
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
