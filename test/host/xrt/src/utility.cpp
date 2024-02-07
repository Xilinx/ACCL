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

#include <utility.hpp>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <signal.h>

pid_t start_emulator(options_t opts, unsigned size, unsigned rank) {
  // Start emulator subprocess
  pid_t pid = fork();
  if (pid == 0) {
    // we're in a child process, start emulated rank
    char const* emu_path_envvar = getenv("ACCL_EMULATOR_PATH");
    std::string emu_path;
    if(emu_path_envvar != NULL){
      emu_path = std::string(emu_path_envvar);
      std::cout << "Using emulator at " << emu_path << std::endl;
    } else {
      std::cout << "Using emulator at default path" << std::endl;
      emu_path = "../../model/emulator/cclo_emu";
    }

    std::stringstream outss, errss;
    outss << "emu_rank_" << rank << "_stdout.log";
    errss << "emu_rank_" << rank << "_stderr.log";
    int outfd = open(outss.str().c_str(), O_WRONLY | O_CREAT, 0666);
    int errfd = open(errss.str().c_str(), O_WRONLY | O_CREAT, 0666);

    dup2(outfd, STDOUT_FILENO);
    dup2(errfd, STDERR_FILENO);

    std::string comm_backend;
    if(opts.udp){
      comm_backend = "udp";
    } else if(opts.tcp || opts.cyt_tcp || opts.axis3){
      comm_backend = "tcp";
    } else if(opts.cyt_rdma){
      comm_backend = "cyt_rdma";
    }

    char* emu_argv[] = {(char*)(emu_path.c_str()),
                        (char*)"-s", (char*)(std::to_string(size).c_str()),
                        (char*)"-r", (char*)(std::to_string(rank).c_str()),
                        (char*)"-p", (char*)(std::to_string(opts.start_port).c_str()),
                        (char*)"-b",
                        (char*)"--comms", (char*)(comm_backend.c_str()),
                        NULL};
    execvp(emu_path.c_str(), emu_argv);
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
