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

#include <fixture.hpp>
#include <utility.hpp>
#include <iostream>
#include <tclap/CmdLine.h>

TEST_F(ACCLTest, test_stress_sndrcv) {
  unsigned int count = options.count;
  auto buf = accl->create_buffer<float>(count, dataType::float32);
  int next_rank = (::rank + 1) % ::size;
  int prev_rank = (::rank + ::size - 1) % ::size;
  for(unsigned int i=0; i<2000; i++) {
    accl->send(*buf, count, next_rank, 0, 0, true);
    accl->recv(*buf, count, prev_rank, 0, 0, true);
  }
  EXPECT_TRUE(true);
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
  try {
    cmd.parse(argc, argv);
    if (hardware_arg.getValue()) {
      if (axis3_arg.getValue() + udp_arg.getValue() + tcp_arg.getValue() != 1) {
        throw std::runtime_error("When using hardware, specify exactly one of axis3, "
                                 "tcp, or udp modes.");
      }
    }
  } catch (std::exception &e) {
    if (::rank == 0) {
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
  opts.device_index = device_index_arg.getValue();
  opts.xclbin = xclbin_arg.getValue();
  opts.test_xrt_simulator = xrt_simulator_ready(opts);
  opts.config_file = config_arg.getValue();
  opts.rsfec = rsfec_arg.getValue();
  return opts;
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &::rank);
  MPI_Comm_size(MPI_COMM_WORLD, &::size);

  //init google test with any arguments specific to it
  ::testing::InitGoogleTest(&argc, argv);

  //gather ACCL options for the test
  //NOTE: this has to come before the gtest environment is initialized
  options = parse_options(argc, argv);

  // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
  ::testing::AddGlobalTestEnvironment(new TestEnvironment);

  bool fail = RUN_ALL_TESTS();
  std::cout << (fail ? "Some tests failed" : "All tests successful") << std::endl;

  return 0;
}
