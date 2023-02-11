/*******************************************************************************
#  Copyright (C) 2023 Xilinx, Inc
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
#include <mpi.h>
#include <random>
#include <sstream>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

using namespace ACCL;
using namespace accl_network_utils;

int main(int argc, char *argv[]) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  xrt::device dev; //dummy XRT device, will not be used

  //ACCL set-up
  std::vector<rank_t> ranks = generate_ranks(true, rank, size);
  std::unique_ptr<ACCL::ACCL >accl = initialize_accl(ranks, rank, true, acclDesign::UDP);
  accl->set_timeout(1e6); //increase time-out for emulation

  //application set-up
  unsigned int i, datasize = 8;
  auto op_buf = accl->create_buffer<float>(datasize * size, dataType::float32);
  for (i=0; i<datasize*size; i++) op_buf->buffer()[i] = 0.0;
  auto scatter_buf = accl->create_buffer<float>(datasize, dataType::float32);
  auto res_buf = accl->create_buffer<float>(datasize, dataType::float32);
  auto gather_buf = accl->create_buffer<float>(datasize * size, dataType::float32);
  MPI_Barrier(MPI_COMM_WORLD);

  //application compute
  accl->scatter(*op_buf, *scatter_buf, datasize, 0); //scatter inputs from rank 0
  for (i=0; i<datasize; i++) res_buf->buffer()[i] = scatter_buf->buffer()[i] + (i + rank);
  accl->gather(*res_buf, *gather_buf, datasize, 0); //gather results to rank 0

  //print results
  if(rank == 0){
    for (i=0; i<datasize*size; ++i) std::cout << "gather_buf[" << i << "] = " << gather_buf->buffer()[i] << std::endl;
  }

  //destroy ACCL unique_ptr
  MPI_Barrier(MPI_COMM_WORLD);
  accl.reset();

  MPI_Finalize();
  return 0;
}
