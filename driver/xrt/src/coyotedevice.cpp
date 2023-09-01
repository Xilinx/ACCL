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

#include "accl/coyotedevice.hpp"
#include "accl/common.hpp"
#include "cProcess.hpp"

#include <iomanip>

namespace ACCL {

CoyoteDevice::CoyoteDevice(): num_qp(0) {
  this->coyote_proc = new fpga::cProcess(targetRegion, getpid());
	std::cerr << "ACLL DEBUG: aquiring cProc: targetRegion: " << targetRegion << ", cPid: " << coyote_proc->getCpid() << std::endl;
}

CoyoteDevice::CoyoteDevice(unsigned int num_qp): num_qp(num_qp) {

  for (unsigned int i=0; i<(num_qp+1); i++)
  {
    fpga::cProcess* cproc = new fpga::cProcess(targetRegion, getpid());
    coyote_qProc_vec.push_back(cproc);
  }

  for (unsigned int i=0; i<coyote_qProc_vec.size(); i++){
    if(coyote_qProc_vec[i]->getCpid() == 0){
      this->coyote_proc = coyote_qProc_vec[i];
      std::cerr << "ACLL DEBUG: aquiring cProc: targetRegion: " << targetRegion << ", cPid: " << coyote_proc->getCpid() << std::endl;
      coyote_qProc_vec.erase(coyote_qProc_vec.begin() + i);
      break;
    }
  }

  if(coyote_proc == NULL || coyote_proc->getCpid() != 0){
    std::cerr << "cProc initialization error!"<<std::endl;

  }

  for (unsigned int i=0; i<coyote_qProc_vec.size(); i++){
    std::cerr << "ACLL DEBUG: aquiring qProc: targetRegion: " << targetRegion << ", cPid: " << coyote_qProc_vec[i]->getCpid() << std::endl;
  }

}

void CoyoteDevice::start(const Options &options) {

  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();

  if (coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2) && (0x02 == 0)) { // read AP_CTRL and check bit 2 (the done bit)
    throw std::runtime_error(
        "Error, collective is already running, wait for previous to complete!");
  }
  int function;
  if (options.scenario == operation::config) {
    function = static_cast<int>(options.cfg_function);
  } else {
    function = static_cast<int>(options.reduce_function);
  }

  uint32_t flags = static_cast<uint32_t>(options.host_flags) << 8 | static_cast<uint32_t>(options.stream_flags);

  // std::cerr << "start: COMPRESSION_FLAGS:" << std::setbase(16) << static_cast<uint32_t>(options.compression_flags) << ", HOST_STREAM_FLAGS:"<<flags<<std::setbase(10) << std::endl;

  coyote_proc->setCSR(static_cast<uint32_t>(options.scenario), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::SCEN)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.count), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::LEN)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.comm), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMM)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.root_src_dst), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ROOT_SRC_DST)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(function), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::FUNCTION_R)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.tag), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::MSG_TAG)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.arithcfg_addr), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::DATAPATH_CFG)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.compression_flags), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMPRESSION_FLAGS)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(flags), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::STREAM_FLAGS)>>2);
  addr_t addr_a = options.addr_0->address();
  coyote_proc->setCSR(static_cast<uint32_t>(addr_a), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_0)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(addr_a >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_1)>>2);
  addr_t addr_b = options.addr_1->address();
  coyote_proc->setCSR(static_cast<uint32_t>(addr_b), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_0)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(addr_b >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_1)>>2);
  addr_t addr_c = options.addr_2->address();
  coyote_proc->setCSR(static_cast<uint32_t>(addr_c), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_0)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(addr_c >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_1)>>2);

  // start the kernel
  coyote_proc->setCSR(0x1U, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);

  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  std::cerr<<"CoyoteDevice invocation latency:"<<durationUs<<"[us]"<<std::endl;
}

void CoyoteDevice::wait() {
  uint32_t is_done = 0;
  uint32_t last = 0xffffffff;
  while (!is_done) {
    uint32_t regi = coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
    if (last != regi) {
      // std::cerr << "Read from AP_CTRL: " << std::setbase(16) << regi << std::setbase(10) << std::endl;
      last = regi;
    }
    is_done = (regi >> 1) & 0x1;
  }

}

CCLO::timeoutStatus CoyoteDevice::wait(std::chrono::milliseconds timeout) {

  uint32_t is_done = 0;
  uint32_t last = 0xffffffff;
  auto start = std::chrono::high_resolution_clock::now();
  while (!is_done) {
    uint32_t regi = coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
    if (last != regi) {
      // std::cerr << "Read from AP_CTRL: " << std::setbase(16) << regi << std::setbase(10) << std::endl;
      last = regi;
    }
    is_done = (regi >> 1) & 0x1;
    auto end = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(end-start) > timeout){
        std::cout<<"CoyoteDevice Wait Time Out"<<std::endl;
        break;
    } 
  }

  return CCLO::no_timeout;
}

CCLO::deviceType CoyoteDevice::get_device_type()
{
  std::cerr<<"get_device_type: coyote_device"<<std::endl;
  return CCLO::coyote_device;
}

void CoyoteDevice::printDebug(){
  coyote_proc->printDebug();

  std::ifstream inputFile("/sys/kernel/coyote_cnfg/cyt_attr_nstats_q0");  

  if (!inputFile.is_open()) {
      std::cerr << "Failed to open net sts file." << std::endl;
  }

  // Read and print the file line by line
  std::string line;
  while (std::getline(inputFile, line)) {
      std::cout << line << std::endl;
  }

  // Close the file
  inputFile.close();
}

void CoyoteDevice::call(const Options &options) {
  start(options);
  wait();
}

val_t CoyoteDevice::read(addr_t offset) {
	// std::cerr << "CoyoteDevice read address: " << ((OFFSET_CCLO + offset)>>2) << std::endl;
  return coyote_proc->getCSR((OFFSET_CCLO + offset)>>2);
}

void CoyoteDevice::write(addr_t offset, val_t val) {
	// std::cerr << "CoyoteDevice write address: " << ((OFFSET_CCLO + offset)>>2) << std::endl;
  coyote_proc->setCSR(val, (OFFSET_CCLO + offset)>>2);
}
} // namespace ACCL
