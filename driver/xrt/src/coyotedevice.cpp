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
#include <future>
#include <iomanip>

static void finish_coyote_request(ACCL::CoyoteRequest *req) {
  req->wait_kernel();
  ACCL::CoyoteDevice *cclo = reinterpret_cast<ACCL::CoyoteDevice *>(req->cclo());
  // get ret code before notifying waiting threads
  req->set_retcode(cclo->read(ACCL::CCLO_ADDR::RETCODE_OFFSET));
  req->set_status(ACCL::operationStatus::COMPLETED);
  req->notify();
  cclo->complete_request(req);
}

namespace ACCL {

void CoyoteRequest::start() {
  assert(this->get_status() ==  operationStatus::EXECUTING);

  int function, arg_id = 0;

  if (options.scenario == operation::config) {
    function = static_cast<int>(options.cfg_function);
  } else {
    function = static_cast<int>(options.reduce_function);
  }
  uint32_t flags = static_cast<uint32_t>(options.host_flags) << 8 | static_cast<uint32_t>(options.stream_flags);

  auto coyote_proc = reinterpret_cast<ACCL::CoyoteDevice *>(cclo())->get_device();

  if (coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2) && (0x02 == 0)) { // read AP_CTRL and check bit 2 (the done bit)
    throw std::runtime_error(
        "Error, collective is already running, wait for previous to complete!");
  }

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

  auto f = std::async(std::launch::async, finish_coyote_request, this);

  // start the kernel
  coyote_proc->setCSR(0x1U, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);

}

void CoyoteRequest::wait_kernel() {
  auto coyote_proc = reinterpret_cast<ACCL::CoyoteDevice *>(cclo())->get_device();
  uint32_t is_done = 0;
  while (!is_done) {
    uint32_t regi = coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
    is_done = (regi >> 1) & 0x1; // get bit 1 of AP_CTRL register
  }
}

CoyoteDevice::CoyoteDevice(): coyote_proc(targetRegion, getpid()), num_qp(0) {
	std::cerr << "ACLL DEBUG: aquiring cProc: targetRegion: " << targetRegion << ", cPid: " << coyote_proc.getCpid() << std::endl;
}

CoyoteDevice::CoyoteDevice(unsigned int num_qp): coyote_proc(targetRegion, getpid()), num_qp(num_qp) {
	std::cerr << "ACLL DEBUG: aquiring cProc: targetRegion: " << targetRegion << ", cPid: " << coyote_proc.getCpid() << std::endl;

  for (unsigned int i=0; i<num_qp; i++)
  {
    
    fpga::cProcess* cproc = new fpga::cProcess(targetRegion, getpid());
    coyote_qProc_vec.push_back(cproc);
    std::cerr << "ACLL DEBUG: aquiring qProc: targetRegion: " << targetRegion << ", cPid: " <<  cproc->getCpid() << std::endl;
  }

}

ACCLRequest *CoyoteDevice::start(const Options &options) {
  ACCLRequest *request = new ACCLRequest;

  if (options.waitfor.size() != 0) {
    throw std::runtime_error("CoyoteDevice does not support chaining");
  }

  CoyoteRequest *fpga_handle =
      new CoyoteRequest(reinterpret_cast<void *>(this), options);

  *request = queue.push(fpga_handle);
  fpga_handle->set_status(operationStatus::QUEUED);

  request_map.emplace(std::make_pair(*request, fpga_handle));

  launch_request();

  return request;

}

void CoyoteDevice::wait(ACCLRequest *request) { 
  auto fpga_handle = request_map.find(*request);
  if (fpga_handle != request_map.end())
    fpga_handle->second->wait(); 
}

timeoutStatus CoyoteDevice::wait(ACCLRequest *request,
                               std::chrono::milliseconds timeout) {
  auto fpga_handle = request_map.find(*request);

  if (fpga_handle == request_map.end() || fpga_handle->second->wait(timeout))
    return timeoutStatus::no_timeout;

  return timeoutStatus::timeout;
}

CCLO::deviceType CoyoteDevice::get_device_type()
{
  std::cerr<<"get_device_type: coyote_device"<<std::endl;
  return CCLO::coyote_device;
}

void CoyoteDevice::printDebug(){
  coyote_proc.printDebug();
}

ACCLRequest *CoyoteDevice::call(const Options &options) {
  ACCLRequest *req = start(options);
  wait(req);
  
  // internal use only
  return req;
}

val_t CoyoteDevice::read(addr_t offset) {
	std::cerr << "CoyoteDevice read address: " << ((OFFSET_CCLO + offset)>>2) << std::endl;
  return coyote_proc.getCSR((OFFSET_CCLO + offset)>>2);
}

void CoyoteDevice::write(addr_t offset, val_t val) {
	std::cerr << "CoyoteDevice write address: " << ((OFFSET_CCLO + offset)>>2) << std::endl;
  coyote_proc.setCSR(val, (OFFSET_CCLO + offset)>>2);
}

void CoyoteDevice::launch_request() {
  // This guarantees permission to only one thread trying to start an operation
  if (queue.run()) {
    CoyoteRequest *req = queue.front();
    assert(req->get_status() == operationStatus::QUEUED);
    req->set_status(operationStatus::EXECUTING);
    req->start();
  }
}

void CoyoteDevice::complete_request(CoyoteRequest *request) {
  if (request->get_status() == operationStatus::COMPLETED) {
    queue.pop();
    launch_request();
  }
}

} // namespace ACCL
