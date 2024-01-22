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
  if(::rank == 0)
    accl->send(*buf_0, std::pow(2,GetParam()), 1, 0, GLOBAL_COMM, true);
  else if(::rank == 1)
    accl->recv(*buf_0, std::pow(2,GetParam()), 0, 0, GLOBAL_COMM, true);
}

TEST_P(ACCLSweepBenchmark, benchmark_broadcast) {
  accl->bcast(*buf_0, std::pow(2,GetParam()), 0, GLOBAL_COMM, true, true);
}

TEST_P(ACCLSweepBenchmark, benchmark_scatter) {
  accl->scatter(*buf_0, *buf_1, std::pow(2,GetParam()), 0, GLOBAL_COMM, true, true);
}

TEST_P(ACCLSweepBenchmark, benchmark_allreduce) {
  accl->allreduce(*buf_0, *buf_1, std::pow(2,GetParam()), reduceFunction::SUM, GLOBAL_COMM, true, true);
}

TEST_P(ACCLSweepBenchmark, benchmark_reduce) {
  accl->reduce(*buf_0, *buf_1, std::pow(2,GetParam()), 0, reduceFunction::SUM, GLOBAL_COMM, true, true);
}

TEST_P(ACCLSweepBenchmark, benchmark_reduce_scatter) {
  accl->reduce_scatter(*buf_0, *buf_1, std::pow(2,GetParam()), reduceFunction::SUM, GLOBAL_COMM, true, true);
}

TEST_P(ACCLSweepBenchmark, benchmark_allgather) {
  accl->allgather(*buf_0, *buf_1, std::pow(2,GetParam()), GLOBAL_COMM, true, true);
}

TEST_P(ACCLSweepBenchmark, benchmark_gather) {
  accl->gather(*buf_0, *buf_1, std::pow(2,GetParam()), 0, GLOBAL_COMM, true, true);
}

INSTANTIATE_TEST_SUITE_P(sweep_benchmarks, ACCLSweepBenchmark, testing::Range(4, 20));

options_t parse_options(int argc, char *argv[]) {
  TCLAP::CmdLine cmd("ACCL benchmark");
  TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
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
  TCLAP::SwitchArg rsfec_arg("", "rsfec", "Enables RS-FEC in CMAC.", cmd,
                             false);

  try {
    cmd.parse(argc, argv);
    if (axis3_arg.getValue() + udp_arg.getValue() + tcp_arg.getValue() +
        roce_arg.getValue() != 1) {
      throw std::runtime_error("When using hardware, specify one of axis3, "
                                "tcp, udp, or roce mode, but not both.");
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
  opts.count = 1024*1024; // 1 Melements by default
  opts.rxbuf_size = 4*opts.count; // 4 MB by default
  opts.rxbuf_count = 16;
  opts.segment_size = opts.rxbuf_size;
  opts.axis3 = axis3_arg.getValue();
  opts.hardware = true; //benchmark only runs on hardware
  opts.benchmark = true;
  opts.udp = udp_arg.getValue();
  opts.tcp = tcp_arg.getValue();
  opts.roce = roce_arg.getValue();
  opts.rsfec = rsfec_arg.getValue();
  opts.device_index = device_index_arg.getValue();
  opts.xclbin = xclbin_arg.getValue();
  opts.csvfile = "results_"+std::to_string(::rank)+".csv";

  return opts;
}

void accl_sa_handler(int)
{
	static bool once = true;
	if(once) {
		accl.reset();
		std::cout << "Error! Signal received. Finalizing MPI..." << std::endl;
		MPI_Finalize();
		std::cout << "Done. Terminating..." << std::endl;
		once = false;
	}
	exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  struct sigaction sa;
  memset( &sa, 0, sizeof(sa) );
  sa.sa_handler = accl_sa_handler;
  sigfillset(&sa.sa_mask);
  sigaction(SIGINT,&sa,NULL);
	sigaction(SIGSEGV, &sa, NULL);

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
