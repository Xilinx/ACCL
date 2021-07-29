#pragma once
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
#
*******************************************************************************/


const auto TAG_ANY = 0xFFFFFFFF;
const auto EXCHANGE_MEM_OFFSET_ADDRESS = 0x1000;
const auto EXCHANGE_MEM_ADDRESS_RANGE = 0x1000;
const auto HOST_CTRL_ADDRESS_RANGE = 0x800;

enum accl_fgFunc {
  enable_irq = 0,
  disable_irq = 1,
  reset_periph = 2,
  enable_pkt = 3,
  set_timeout = 4,
  init_connection = 5,
  open_port = 6,
  open_con = 7,
  use_tcp_stack = 8,
  use_udp_stack = 9,
  start_profiling = 10,
  end_profiling = 11,
  set_dma_transaction_size = 12
};

enum accl_operation_t {
  config = 0,
  sendop = 1,
  recvop = 2,
  bcast = 3,
  scatter = 4,
  gather = 5,
  reduce = 6,
  allgather = 7,
  allreduce = 8,
  accumulate = 9,
  copy = 10,
  reduce_ring = 11,
  allreduce_fused_ring = 12,
  gather_ring = 13,
  allgather_ring = 14,
  ext_stream_krnl = 15,
  ext_reduce = 16,
  bcast_rr = 17,
  scatter_rr = 18,
  allreduce_share_ring = 19,
  nop = 255
};

typedef enum { fp = 0, dp = 1, i32 = 2, i64 = 3 } accl_reduce_func;

enum accl_error_code {
  COLLECTIVE_OP_SUCCESS = 0,
  DMA_MISMATCH_ERROR = 1,
  DMA_INTERNAL_ERROR = 2,
  DMA_DECODE_ERROR = 3,
  DMA_SLAVE_ERROR = 4,
  DMA_NOT_OKAY_ERROR = 5,
  DMA_NOT_END_OF_PACKET_ERROR = 6,
  DMA_NOT_EXPECTED_BTT_ERROR = 7,
  DMA_TIMEOUT_ERROR = 8,
  CONFIG_SWITCH_ERROR = 9,
  DEQUEUE_BUFFER_TIMEOUT_ERROR = 10,
  RECEIVE_TIMEOUT_ERROR = 12,
  DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR = 11,
  DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH = 13,
  DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR = 14,
  COLLECTIVE_NOT_IMPLEMENTED = 15,
  RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID = 16,
  OPEN_PORT_NOT_SUCCEEDED = 17,
  OPEN_COM_NOT_SUCCEEDED = 18,
  DMA_SIZE_ERROR = 19,
  ARITH_ERROR = 20,
  PACK_TIMEOUT_STS_ERROR = 21,
  PACK_SEQ_NUMBER_ERROR = 22
};
