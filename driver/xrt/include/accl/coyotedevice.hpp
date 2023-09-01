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

#pragma once
#include "cclo.hpp"
#include "constants.hpp"
#include "cProcess.hpp"
#include "ibvQpConn.hpp"
#include "ibvStructs.hpp"
#include <string>
#include <iostream>
#include <fstream>

constexpr int targetRegion = 0;

/** @file coyotedevice.hpp */

namespace ACCL {
	
/**
 * Address offsets
 */
const size_t OFFSET_HOSTCTRL = 0x2000; 
const size_t OFFSET_CCLO     = 0x0000; 

namespace HOSTCTRL_ADDR {
// address maps for the hostcontrol kernel
// (copied from hw/hdl/operators/examples/accl/rtl/hostctrl_control_s_axi.v)
//------------------------Address Info-------------------
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
constexpr auto const AP_CTRL                = 0x00;
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
constexpr auto const GIE                    = 0x04;
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
constexpr auto const IER                    = 0x08;
// 0x0c : IP Interrupt Status Register (Read/COR)
//        bit 0 - ap_done (Read/COR)
//        bit 1 - ap_ready (Read/COR)
//        others - reserved
constexpr auto const ISR                    = 0x0c;
// 0x10 : Data signal of scenario
//        bit 31~0 - scenario[31:0] (Read/Write)
constexpr auto const SCEN                   = 0x10;
// 0x14 : reserved
// 0x18 : Data signal of len
//        bit 31~0 - len[31:0] (Read/Write)
constexpr auto const LEN                    = 0x18;
// 0x1c : reserved
// 0x20 : Data signal of comm
//        bit 31~0 - comm[31:0] (Read/Write)
constexpr auto const COMM                   = 0x20;
// 0x24 : reserved
// 0x28 : Data signal of root_src_dst
//        bit 31~0 - root_src_dst[31:0] (Read/Write)
constexpr auto const ROOT_SRC_DST           = 0x28;
// 0x2c : reserved
// 0x30 : Data signal of function_r
//        bit 31~0 - function_r[31:0] (Read/Write)
constexpr auto const FUNCTION_R             = 0x30;
// 0x34 : reserved
// 0x38 : Data signal of msg_tag
//        bit 31~0 - msg_tag[31:0] (Read/Write)
constexpr auto const MSG_TAG               = 0x38;
// 0x3c : reserved
// 0x40 : Data signal of datapath_cfg
//        bit 31~0 - datapath_cfg[31:0] (Read/Write)
constexpr auto const DATAPATH_CFG           = 0x40;
// 0x44 : reserved
// 0x48 : Data signal of compression_flags
//        bit 31~0 - compression_flags[31:0] (Read/Write)
constexpr auto const COMPRESSION_FLAGS           = 0x48;
// 0x4c : reserved
// 0x50 : Data signal of stream_flags
//        bit 31~0 - stream_flags[31:0] (Read/Write)
constexpr auto const STREAM_FLAGS           = 0x50;
// 0x54 : reserved
// 0x58 : Data signal of addra
//        bit 31~0 - addra[31:0] (Read/Write)
constexpr auto const ADDRA_0           = 0x58;
// 0x5c : Data signal of addra
//        bit 31~0 - addra[63:32] (Read/Write)
constexpr auto const ADDRA_1           = 0x5c;
// 0x60 : reserved
// 0x64 : Data signal of addrb
//        bit 31~0 - addrb[31:0] (Read/Write)
constexpr auto const ADDRB_0           = 0x64;
// 0x68 : Data signal of addrb
//        bit 31~0 - addrb[63:32] (Read/Write)
constexpr auto const ADDRB_1           = 0x68;
// 0x6c : reserved
// 0x70 : Data signal of addrc
//        bit 31~0 - addrc[31:0] (Read/Write)
constexpr auto const ADDRC_0           = 0x70;
// 0x74 : Data signal of addrc
//        bit 31~0 - addrc[63:32] (Read/Write)
constexpr auto const ADDRC_1           = 0x74;
// 0x78 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)
// control signals
}

namespace CCLO_ADDR {
// address maps for the cclo kernel
// TODO
constexpr auto const GIE                    = 0x04;
constexpr auto const IER                    = 0x08;
constexpr auto const ISR                    = 0x0c;

}

/**
 * Implementation of CCLO that uses a CCLO kernel on a FPGA.
 *
 */
class CoyoteDevice : public CCLO {
public:
  CoyoteDevice();
  CoyoteDevice(unsigned int num_qp);
  /**
   * Destroy the CoyoteDevice object
   *
   */
  virtual ~CoyoteDevice() {}

  void call(const Options &options) override;

  void start(const Options &options) override;

  val_t read(addr_t offset) override;

  void write(addr_t offset, val_t val) override;

  void wait() override;

  timeoutStatus wait(std::chrono::milliseconds timeout) override;

  addr_t get_base_addr() override {
    // TODO: Find way to retrieve CCLO base address on FPGA
    return 0x0;
  }

  deviceType get_device_type() override;

  void printDebug() override;

  fpga::cProcess* get_device(){
    return coyote_proc;
  }

  fpga::cProcess* coyote_proc;

  // RDMA related 
  // RDMA requires multiple processes to establish queue pairs
  // The CCLO kernel is still managed by coyote_proc
  unsigned int num_qp;
  std::vector<fpga::cProcess*> coyote_qProc_vec;

private:
  
};
} // namespace ACCL
