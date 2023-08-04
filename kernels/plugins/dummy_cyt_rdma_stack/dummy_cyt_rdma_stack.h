# /*******************************************************************************
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

#pragma once

#include "eth_intf.h"
#include "accl_hls.h"

void cyt_rdma(
	STREAM<rdma_req_t> & rdma_sq,
    STREAM<eth_notification> & notif,
    STREAM<stream_word> & send_data,
    STREAM<stream_word> & recv_data,
    STREAM<stream_word> & wr_data,
    STREAM<ap_axiu<104,0,0,DEST_WIDTH>> & wr_cmd,
    STREAM<ap_uint<32> > & wr_sts,
    STREAM<stream_word> & rx,
    STREAM<stream_word> & tx
);