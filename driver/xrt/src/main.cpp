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
# *******************************************************************************/

#include "timing.hpp"
#include "xlnx-dac.hpp"

#include <mpi.h>
#include <vector>

int check_usage(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <bitstream> <device_idx> <bank_id> <mode>" << std::endl;
    exit(-1);
  }
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  check_usage(argc, argv);

  const std::string bitstream_f = argv[1];
  const auto device_idx = atoi(argv[2]);
  const auto bank_idx = atoi(argv[3]);
  const auto mode = atoi(argv[4]);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "Rank " << rank << std::endl;
  std::cout << "Bitstream " << bitstream_f << std::endl;
  std::cout << "Bank Idx " << bank_idx << std::endl;
  std::cout << "Mode " << mode << std::endl;

  // Setup
  Timer t_construct, t_bitstream, t_read_reg, t_write_reg, t_execute_kernel,
      t_preprxbuffers, t_dump_rx_buffers, t_config_comm;
  accl_operation_t op = nop;

  const int nbuf = size;
  constexpr int buffer_size = 16 * 1024;

  t_construct.start();
  ACCL f(nbuf, buffer_size, device_idx, DUAL);
  t_construct.end();

  t_bitstream.start();
  f.load_bitstream(bitstream_f);
  t_bitstream.end();

  t_config_comm.start();
  //f.config_comm(nbuf);
  t_config_comm.end();

  t_preprxbuffers.start();
  f.prep_rx_buffers(bank_idx);
  t_preprxbuffers.end();

  t_dump_rx_buffers.start();
  f.dump_rx_buffers();
  t_dump_rx_buffers.end();

  t_execute_kernel.start();
  f.nop_op();
  t_execute_kernel.end();

  std::cout << "t_construct: " << t_construct.elapsed() << " usecs"
            << std::endl;
  std::cout << "t_bitstream: " << t_bitstream.elapsed() << " usecs"
            << std::endl;
  std::cout << "t_config_comm: " << t_config_comm.elapsed() << " usecs"
            << std::endl;
  std::cout << "t_preprxbuffers: " << t_preprxbuffers.elapsed() << " usecs"
            << std::endl;
  std::cout << "t_dump_rx_buffers: " << t_dump_rx_buffers.elapsed() << " usecs"
            << std::endl;
  std::cout << "t_execute_kernel: " << t_execute_kernel.elapsed() << " usecs"
            << std::endl;

  std::cout << "HWID:" << f.get_hwid() << std::dec << std::endl;
  MPI_Finalize();
  return 0;
}
