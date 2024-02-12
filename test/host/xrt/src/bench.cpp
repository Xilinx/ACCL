/*******************************************************************************
#  Copyright (C) 2022 Advanced Micro Devices, Inc
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

#include <fixture.hpp>
#include <utility.hpp>
#include <iostream>
#include <tclap/CmdLine.h>


TEST_P(ACCLSweepBenchmark, benchmark_sendrecv) {
  if(::rank == 0){
    duration = accl->get_duration(accl->send(*buf_0, std::pow(2,GetParam()), 1, 0, GLOBAL_COMM, true));
  } else if(::rank == 1) {
    duration = accl->get_duration(accl->recv(*buf_0, std::pow(2,GetParam()), 0, 0, GLOBAL_COMM, true));
  }
}

TEST_P(ACCLSweepBenchmark, benchmark_broadcast) {
  duration = accl->get_duration(accl->bcast(*buf_0, std::pow(2,GetParam()), 0, GLOBAL_COMM, true, true));
}

TEST_P(ACCLSweepBenchmark, benchmark_scatter) {
  duration = accl->get_duration(accl->scatter(*buf_0, *buf_1, std::pow(2,GetParam()), 0, GLOBAL_COMM, true, true));
}

TEST_P(ACCLSweepBenchmark, benchmark_allreduce) {
  duration = accl->get_duration(accl->allreduce(*buf_0, *buf_1, std::pow(2,GetParam()), reduceFunction::SUM, GLOBAL_COMM, true, true));
}

TEST_P(ACCLSweepBenchmark, benchmark_reduce) {
  duration = accl->get_duration(accl->reduce(*buf_0, *buf_1, std::pow(2,GetParam()), 0, reduceFunction::SUM, GLOBAL_COMM, true, true));
}

TEST_P(ACCLSweepBenchmark, benchmark_reduce_scatter) {
  duration = accl->get_duration(accl->reduce_scatter(*buf_0, *buf_1, std::pow(2,GetParam()), reduceFunction::SUM, GLOBAL_COMM, true, true));
}

TEST_P(ACCLSweepBenchmark, benchmark_allgather) {
  duration = accl->get_duration(accl->allgather(*buf_0, *buf_1, std::pow(2,GetParam()), GLOBAL_COMM, true, true));
}

TEST_P(ACCLSweepBenchmark, benchmark_gather) {
  duration = accl->get_duration(accl->gather(*buf_0, *buf_1, std::pow(2,GetParam()), 0, GLOBAL_COMM, true, true));
}

INSTANTIATE_TEST_SUITE_P(sweep_benchmarks, ACCLSweepBenchmark, testing::Range(4, 20));

options_t parse_options(int argc, char *argv[]) {
  TCLAP::CmdLine cmd("ACCL benchmark");
  TCLAP::ValueArg<uint16_t> start_port_arg(
      "p", "start-port", "Start of range of ports usable for sim", false, 5500,
      "positive integer");
  cmd.add(start_port_arg);
  TCLAP::ValueArg<unsigned int> count_arg(
      "s", "count", "How many items per test", false, 16, "positive integer");
  cmd.add(count_arg);
  TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
  TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd,
                             false);
  TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
  TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
  TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd, false);
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
  TCLAP::SwitchArg rsfec_arg("", "rsfec", "Enables RS-FEC in CMAC.", cmd,
                             false);

  try {
    cmd.parse(argc, argv);
    if (axis3_arg.getValue() + udp_arg.getValue() + tcp_arg.getValue() != 1) {
      throw std::runtime_error("When using hardware, specify exactly one of axis3, "
                                "tcp, or udp modes.");
    }
    if (axis3_arg.getValue() && (::size > 3)) {
      throw std::runtime_error("When using axis3x, use up to 3 ranks.");
    }
  } catch (std::exception &e) {
    if (::rank == 0) {
      std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
  }
  options_t opts;
  opts.hardware = hardware_arg.getValue();
  opts.start_port = start_port_arg.getValue();
  opts.count = count_arg.getValue();
  opts.rxbuf_size = 4*opts.count; // 4 MB by default
  opts.rxbuf_count = 16;
  opts.segment_size = opts.rxbuf_size;
  opts.axis3 = axis3_arg.getValue();

  opts.benchmark = true;
  opts.udp = udp_arg.getValue();
  opts.tcp = tcp_arg.getValue();
  opts.rsfec = rsfec_arg.getValue();
  opts.device_index = device_index_arg.getValue();
  opts.xclbin = xclbin_arg.getValue();
  opts.csvfile = "profile_"+std::to_string(::rank)+".csv";
  opts.test_xrt_simulator = xrt_simulator_ready(opts);

  return opts;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &::rank);
  MPI_Comm_size(MPI_COMM_WORLD, &::size);

  //init google test with any arguments specific to it
  ::testing::InitGoogleTest(&argc, argv);

  options = parse_options(argc, argv);

  // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
  ::testing::AddGlobalTestEnvironment(new TestEnvironment);

  bool fail = RUN_ALL_TESTS();

  MPI_Finalize();
  return fail;
}
