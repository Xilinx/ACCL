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

#include "accl/constants.hpp"

namespace ACCL {
const char *error_code_to_string(errorCode code) {
  switch (code) {
  case errorCode::DMA_MISMATCH_ERROR:
    return "DMA MISMATCH ERROR";
  case errorCode::DMA_INTERNAL_ERROR:
    return "DMA INTERNAL ERROR";
  case errorCode::DMA_DECODE_ERROR:
    return "DMA DECODE ERROR";
  case errorCode::DMA_SLAVE_ERROR:
    return "DMA SLAVE ERROR";
  case errorCode::DMA_NOT_OKAY_ERROR:
    return "DMA NOT OKAY ERROR";
  case errorCode::DMA_NOT_END_OF_PACKET_ERROR:
    return "DMA NOT END OF PACKET ERROR";
  case errorCode::DMA_NOT_EXPECTED_BTT_ERROR:
    return "DMA NOT EXPECTED BTT ERROR";
  case errorCode::DMA_TIMEOUT_ERROR:
    return "DMA TIMEOUT ERROR";
  case errorCode::CONFIG_SWITCH_ERROR:
    return "CONFIG SWITCH ERROR";
  case errorCode::DEQUEUE_BUFFER_TIMEOUT_ERROR:
    return "DEQUEUE BUFFER ERROR";
  case errorCode::DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR:
    return "DEQUEUE BUFFER SPARE BUFFER STATUS ERROR";
  case errorCode::RECEIVE_TIMEOUT_ERROR:
    return "RECEIVE TIMEOUT ERROR";
  case errorCode::DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH:
    return "DEQUEUE BUFFER SPARE BUFFER DMATAG MISMATCH";
  case errorCode::DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR:
    return "DEQUEUE BUFFER SPARE BUFFER INDEX ERROR";
  case errorCode::COLLECTIVE_NOT_IMPLEMENTED:
    return "COLLECTIVE NOT IMPLEMENTED ERROR";
  case errorCode::RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID:
    return "RECEIVE OFFCHIP SPARE BUFF ID NOT VALID";
  case errorCode::EAGER_THRESHOLD_INVALID:
    return "EAGER THRESHOLD VALUE INVALID";
  case errorCode::RENDEZVOUS_THRESHOLD_INVALID:
    return "RENDEZVOUS THRESHOLD VALUE INVALID";
  case errorCode::DMA_SIZE_ERROR:
    return "DMA SIZE ERROR";
  case errorCode::ARITH_ERROR:
    return "ARITHMETIC ERROR";
  case errorCode::PACK_TIMEOUT_STS_ERROR:
    return "PACK TIMEOUT STS ERROR";
  case errorCode::PACK_SEQ_NUMBER_ERROR:
    return "PACK SEQUENCE NUMBER ERROR";
  case errorCode::COMPRESSION_ERROR:
    return "COMPRESSION ERROR";
  case errorCode::KRNL_TIMEOUT_STS_ERROR:
    return "KERNEL TIMEOUT STS ERROR";
  case errorCode::KRNL_STS_COUNT_ERROR:
    return "KERNEL STS COUNT ERROR";
  case errorCode::SEGMENTER_EXPECTED_BTT_ERROR:
    return "SEGMENTER EXPECTED BTT ERROR";
  case errorCode::DMA_TAG_MISMATCH_ERROR:
    return "DMA TAG MISMATCH ERROR";
  default:
    return "UNKNOWN ERROR";
  }
}
} // namespace ACCL
