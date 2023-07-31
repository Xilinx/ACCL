/*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
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

#pragma once

#include <accl.hpp>
#include <accl_network_utils.hpp>
#include <cstdlib>
#include <experimental/xrt_ip.h>
#include <fstream>
#include <json/json.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <sys/types.h>
#include <probe.hpp>
#include <utility.hpp>

using namespace ACCL;
using namespace accl_network_utils;

inline int rank;
inline int size;
inline pid_t emulator_pid;
inline options_t options;
inline xrt::device dev;
inline std::unique_ptr<ACCL::ACCL> accl;
inline ACCLProbe *probe;
inline std::ofstream csvstream;

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

      // Set up for benchmarking
      if (options.benchmark && options.hardware && !(options.axis3 && rank > 0)) {
        // Create a probe object or disable benchmark
        std::cout << "Initializing call probe on rank " << rank << std::endl;
        auto xclbin_uuid = dev.load_xclbin(options.xclbin);
        try{
          auto probe_krnl = xrt::kernel(dev, xclbin_uuid, "call_probe:{probe_0}");
          probe = new ACCLProbe(dev, probe_krnl, options.csvfile);
        } catch(...) {
          std::cout << "Failed to initialize call probe, disabling benchmarking" << std::endl;
          options.benchmark = false;
        }
      } else {
        // Clear any erroneous setting of benchmark flag
        options.benchmark = false;
      }

      if(options.benchmark){
        // Arm probe with enough iterations to allow ACCL initialization
        if (options.tcp) {
          probe->arm(7);
        } else {
          probe->arm(5);
        }
      }

      accl = initialize_accl(
          ranks, rank, !options.hardware, design, dev, options.xclbin, options.rxbuf_count,
          options.rxbuf_size, options.segment_size, options.rsfec);
      std::cout << "Setting up TestEnvironment" << std::endl;
      accl->set_timeout(1e6);

      if(options.benchmark){
        probe->read();
      }

    }

    virtual void TearDown(){
      if(options.benchmark){
        probe->flush();//flush profiling data to CSV file
        delete probe;
      }
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

class ACCLBenchmark : public ::testing::Test {
protected:
  inline static std::unique_ptr<ACCL::Buffer<float>> buf_0, buf_1, buf_2;
  virtual void SetUp() {
    if(options.benchmark) probe->arm(1);//arm probe for one ACCL call
    if(!buf_0) buf_0 = accl->create_buffer<float>(options.count, dataType::float32);
    if(!buf_1) buf_1 = accl->create_buffer<float>(options.count, dataType::float32);
    if(!buf_2) buf_2 = accl->create_buffer<float>(options.count, dataType::float32);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  virtual void TearDown() {
    MPI_Barrier(MPI_COMM_WORLD);
    if(options.benchmark) probe->read(true);//readback profiling data and append to existing
  }
};

class ACCLRootTest : public ACCLTest, public testing::WithParamInterface<int> {};
class ACCLFuncTest : public ACCLTest, public testing::WithParamInterface<reduceFunction> {};
class ACCLRootFuncTest : public ACCLTest, public testing::WithParamInterface<std::tuple<int, reduceFunction>> {};
class ACCLSegmentationTest : public ACCLTest, public testing::WithParamInterface<std::tuple<unsigned int, int>> {};
class ACCLSweepBenchmark : public ACCLBenchmark, public testing::WithParamInterface<int> {};
