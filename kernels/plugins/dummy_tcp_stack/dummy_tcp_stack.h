# /*******************************************************************************
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
# *******************************************************************************/

#pragma once

#include "eth_intf.h"
#include "accl_hls.h"

void network_krnl(
    STREAM<pkt128>& m_axis_tcp_notification, 
    STREAM<pkt32>& s_axis_tcp_read_pkg,
    STREAM<pkt16>& m_axis_tcp_rx_meta, 
    STREAM<stream_word>& m_axis_tcp_rx_data,
    STREAM<pkt32>& s_axis_tcp_tx_meta, 
    STREAM<stream_word>& s_axis_tcp_tx_data, 
    STREAM<pkt64>& m_axis_tcp_tx_status,
    STREAM<stream_word>& net_rx,
    STREAM<stream_word>& net_tx
);