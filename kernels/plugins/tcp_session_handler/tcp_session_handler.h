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
# *******************************************************************************/
#include "accl_hls.h"
#include "stdint.h"
#include "constants.hpp"

void tcp_session_handler(
    uint32_t ip,
    uint16_t port_nr,
    volatile uint16_t *session_id,
    volatile uint8_t *return_code,
    ACCL::tcpSessionHandlerOperation operation,
    STREAM<ap_axiu<16, 0, 0, 0>>& listen_port, 
    STREAM<ap_axiu<8, 0, 0, 0>>& port_status,
	STREAM<ap_axiu<64, 0, 0, 0>>& open_connection,
    STREAM<ap_axiu<16, 0, 0, 0>>& close_connection,
    STREAM<ap_axiu<128, 0, 0, 0>>& open_status
);
