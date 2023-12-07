#pragma once
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
#include <bits/stdint-uintn.h>
#include <map>
#include <stddef.h>

/** @file constants.hpp */

/** ACCL namespace */
namespace ACCL {
/** Device address type */
typedef uint64_t addr_t;
/** Device value type */
typedef uint32_t val_t;
/** Communicator IDs used to refer to a certain communicator */
typedef unsigned int communicatorId;

/** Tag any */
const unsigned int TAG_ANY = 0xFFFFFFFF;
/** Exchange mem offset address */
const addr_t EXCHANGE_MEM_OFFSET_ADDRESS = 0x0;
/** Exchange mem address range */
const addr_t EXCHANGE_MEM_ADDRESS_RANGE = 0x2000;
/** Global Communicator */
const communicatorId GLOBAL_COMM = 0x0;

/**
 * Address offsets inside the HOSTCTRL internal memory
 * 
 */
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

/**
 * Address offsets inside the CCLO internal memory
 * 
 */
namespace CCLO_ADDR {
  constexpr auto const RETCODE_OFFSET         = 0x1FFC;
  constexpr auto const IDCODE_OFFSET          = 0x1FF8;
  constexpr auto const CFGRDY_OFFSET          = 0x1FF4;
  constexpr auto const PERFCNT_OFFSET         = 0x1FF0;
  constexpr auto const SPARE3_OFFSET          = 0x1FE8;
  constexpr auto const SPARE2_OFFSET          = 0x1FE0;
  constexpr auto const SPARE1_OFFSET          = 0x1FD8;
  constexpr auto const REDUCE_FLAT_TREE_MAX_COUNT_OFFSET = 0x1FD4;
  constexpr auto const REDUCE_FLAT_TREE_MAX_RANKS_OFFSET = 0x1FD0;
  constexpr auto const BCAST_FLAT_TREE_MAX_RANKS_OFFSET = 0x1FCC;
  constexpr auto const GATHER_FLAT_TREE_MAX_COUNT_OFFSET = 0x1FC8;
  constexpr auto const GATHER_FLAT_TREE_MAX_FANIN_OFFSET = 0x1FC4;
  constexpr auto const EGR_RX_BUF_SIZE_OFFSET = 0x4;
  constexpr auto const NUM_EGR_RX_BUFS_OFFSET = 0x0;
}

/**
 * IDs for XRT argument passing
*/
namespace XRT_ARG_ID {
  constexpr auto const SCENARIO_ID = 0;
  constexpr auto const COUNT_ID = 1;
  constexpr auto const COMM_ID = 2;
  constexpr auto const ROOT_SRC_DST_ID = 3;
  constexpr auto const FUNCTION_ID = 4;
  constexpr auto const TAG_ID = 5;
  constexpr auto const ARITHCFG_ADDR_ID = 6;
  constexpr auto const COMPRESSION_FLAGS_ID = 7;
  constexpr auto const STREAM_FLAGS_ID = 8;
  constexpr auto const ADDR_0_ID = 9;
  constexpr auto const ADDR_1_ID = 10;
  constexpr auto const ADDR_2_ID = 11;
}


/**
 * Configuration functions
 *
 */
enum class cfgFunc {
  reset_periph = 0,
  enable_pkt = 1,
  set_timeout = 2,
  set_max_eager_msg_size = 3,
  set_max_rendezvous_msg_size = 4
};

/**
 * Supported ACCL operations
 *
 */
enum class operation : int {
  config = 0,           /**< Set CCLO config */
  copy = 1,             /**< Copy FPGA buffer */
  combine = 2,          /**< Perform reduce operator on FPGA buffers */
  send = 3,             /**< Send FPGA buffer to rank */
  recv = 4,             /**< Recieve FPGA buffer from rank */
  bcast = 5,            /**< Broadcast FPGA buffer from root */
  scatter = 6,          /**< Scatter FPGA buffer from root */
  gather = 7,           /**< Gather FPGA buffers on root */
  reduce = 8,           /**< Perform reduce operator on remote FPGA buffers
                             on root */
  allgather = 9,        /**< Gather FPGA buffers on all ranks */
  allreduce = 10,       /**< Perform reduce operator on remote FPGA buffers
                             on all ranks */
  reduce_scatter = 11,  /**< Perform reduce operator on remote FPGA buffers
                             and scatter to all ranks */
  barrier = 12,         /**< barrier kernel */
  alltoall = 13,        /**< All-to-all kernel */
  nop = 255             /**< NOP operation */
};

/**
 * ACCL reduce functions
 *
 * Used by operation::combine, operation::reduce,
 * operation::allreduce and operation::reduce_scatter.
 */
enum class reduceFunction {
  SUM = 0, /**< Elementwise sum */
  MAX = 1  /**< Elementwise max */
};

/**
 * Status of ACCL operations
 *
 */
enum class operationStatus : int {
  QUEUED = 0,    /**< Operation waiting in hardware queue */
  EXECUTING = 1, /**< Operation currently executing */
  COMPLETED = 2  /**< Operation completed */
};

/**
 * Request handle passe to the user
 */  
typedef long long ACCLRequest;

/**
 * Status of the FPGA queue
 * 
 */
enum class queueStatus {
  IDLE = 0,
  BUSY = 1
};

enum timeoutStatus {
  no_timeout,
  timeout
};

/**
 * ACCL supported data types
 *
 */
enum class dataType {
  none,    /**< No datatype */
  int8,    /**< 8-bit integer; unsupported datatype, only used internally. */
  float16, /**< 16-bit floating-point number */
  float32, /**< 32-bit floating-point number */
  float64, /**< 64-bit floating-point number */
  int32,   /**< 32-bit integer */
  int64    /**< 64-bit integer */
};

/**
 * Size of the datatypes in bits.
 *
 */
const std::map<dataType, unsigned int> dataTypeSize = {
    {dataType::none, 0},     {dataType::int8, 8},     {dataType::float16, 16},
    {dataType::float32, 32}, {dataType::float64, 64}, {dataType::int32, 32},
    {dataType::int64, 64}};

/**
 * ACCL stream flags to specify streamed buffers.
 *
 */
enum class streamFlags {
  NO_STREAM = 0,  /**< No buffers are streamed */
  OP0_STREAM = 1, /**< The first operand is streamed */
  RES_STREAM = 2  /**< The result is streamed */
};

inline streamFlags operator|(streamFlags lhs, streamFlags rhs) {
  return static_cast<streamFlags>(static_cast<int>(lhs) |
                                  static_cast<int>(rhs));
}

inline streamFlags &operator|=(streamFlags &lhs, streamFlags rhs) {
  lhs = lhs | rhs;
  return lhs;
}

/**
 * ACCL host flags to specify host-only buffers.
 *
 */
enum class hostFlags {
  NO_HOST = 0,  /**< No buffers are host-only */
  OP0_HOST = 1, /**< The first operand is host-only */
  OP1_HOST = 2, /**< The second operand is host-only */
  RES_HOST = 4  /**< The result is host-only */
};

inline hostFlags operator|(hostFlags lhs, hostFlags rhs) {
  return static_cast<hostFlags>(static_cast<int>(lhs) |
                                  static_cast<int>(rhs));
}

inline hostFlags &operator|=(hostFlags &lhs, hostFlags rhs) {
  lhs = lhs | rhs;
  return lhs;
}

/**
 * ACCL compression flags to specify compression configuration.
 *
 */
enum class compressionFlags {
  NO_COMPRESSION = 0, /**< No compression should be used */
  OP0_COMPRESSED = 1, /**< Operand 0 is already compressed */
  OP1_COMPRESSED = 2, /**< Operand 1 is already compressed */
  RES_COMPRESSED = 4, /**< Result should be compressed */
  ETH_COMPRESSED = 8  /**< Ethernet compression should be used */
};

/**
 * ACCL supported network protocols.
 *
 * Should match the protocol used in the ACCL kernel.
 *
 */
enum class networkProtocol {
  TCP,  /**< The TCP protocol */
  UDP,  /**< The UDP protocol */
  RDMA  /**< Use RDMA for data transfers */
};

inline compressionFlags operator|(compressionFlags lhs, compressionFlags rhs) {
  return static_cast<compressionFlags>(static_cast<int>(lhs) |
                                       static_cast<int>(rhs));
}

inline compressionFlags &operator|=(compressionFlags &lhs,
                                    compressionFlags rhs) {
  lhs = lhs | rhs;
  return lhs;
}

/**
 * ACCL error codes used internally.
 *
 */
enum class errorCode {
  COLLECTIVE_OP_SUCCESS = 0,
  DMA_MISMATCH_ERROR = 1 << 0,
  DMA_INTERNAL_ERROR = 1 << 1,
  DMA_DECODE_ERROR = 1 << 2,
  DMA_SLAVE_ERROR = 1 << 3,
  DMA_NOT_OKAY_ERROR = 1 << 4,
  DMA_NOT_END_OF_PACKET_ERROR = 1 << 5,
  DMA_NOT_EXPECTED_BTT_ERROR = 1 << 6,
  DMA_TIMEOUT_ERROR = 1 << 7,
  CONFIG_SWITCH_ERROR = 1 << 8,
  DEQUEUE_BUFFER_TIMEOUT_ERROR = 1 << 9,
  DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR = 1 << 10,
  RECEIVE_TIMEOUT_ERROR = 1 << 11,
  DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH = 1 << 12,
  DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR = 1 << 13,
  COLLECTIVE_NOT_IMPLEMENTED = 1 << 14,
  RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID = 1 << 15,
  EAGER_THRESHOLD_INVALID = 1 << 16,
  RENDEZVOUS_THRESHOLD_INVALID = 1 << 17,
  DMA_SIZE_ERROR = 1 << 18,
  ARITH_ERROR = 1 << 19,
  PACK_TIMEOUT_STS_ERROR = 1 << 20,
  PACK_SEQ_NUMBER_ERROR = 1 << 21,
  COMPRESSION_ERROR = 1 << 22,
  KRNL_TIMEOUT_STS_ERROR = 1 << 23,
  KRNL_STS_COUNT_ERROR = 1 << 24,
  SEGMENTER_EXPECTED_BTT_ERROR = 1 << 25,
  DMA_TAG_MISMATCH_ERROR = 1 << 26
};

/** Amount of bits used for error codes. */
const size_t error_code_bits = 26;

inline errorCode operator|(errorCode lhs, errorCode rhs) {
  return static_cast<errorCode>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline errorCode &operator|=(errorCode &lhs, errorCode rhs) {
  lhs = lhs | rhs;
  return lhs;
}

/**
 * Convert an ACCL error code to a string.
 *
 * @param code          Error code to convert to string.
 * @return const char*  Matching error code string.
 */
const char *error_code_to_string(errorCode code);
} // namespace ACCL
