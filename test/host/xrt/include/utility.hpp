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

#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <string>
#include <random>
#include <functional>
#include <memory>

struct options_t {
  int start_port;
  unsigned int rxbuf_size;
  unsigned int rxbuf_count;
  unsigned int segment_size;
  unsigned int max_eager_count;
  unsigned int count;
  unsigned int device_index;
  bool test_xrt_simulator;
  bool debug;
  bool hardware;
  bool axis3;
  bool udp;
  bool tcp;
  bool cyt_tcp;
  bool cyt_rdma;
  bool return_error;
  bool rsfec;
  std::string xclbin;
  std::string config_file;
  bool startemu;
  bool benchmark;
  std::string csvfile;
};

pid_t start_emulator(options_t opts, unsigned size, unsigned rank);
bool emulator_is_running(pid_t pid);
void kill_emulator(pid_t pid);

void sigint_handler(int signum);

void test_debug(std::string message, options_t &options);
bool xrt_simulator_ready(const options_t &opts);

template <typename T>
bool is_close(T a, T b, double rtol = 1e-5, double atol = 1e-8) {
  // std::cout << abs(a - b) << " <= " << (atol + rtol * abs(b)) << "? " <<
  // (abs(a - b) <= (atol + rtol * abs(b))) << std::endl;
  return abs(a - b) <= (atol + rtol * abs(b));
}

template <typename T> static void random_array(T *data, size_t count) {
  std::uniform_real_distribution<T> distribution(-1, 1);
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
