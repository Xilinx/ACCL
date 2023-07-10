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
# *******************************************************************************/
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "eth_intf.h"
#include "rxbuf_offload.h"

using namespace std;



void rdma_metatable(
	STREAM<rdma_meta_upd_t> & rdma_meta_upd,
    STREAM<rdma_meta_req_t> & rdma_meta_req,
	STREAM<rdma_meta_rsp_t> & rdma_meta_rsp
) {
#pragma HLS INTERFACE axis register both port=rdma_meta_upd
#pragma HLS INTERFACE axis register both port=rdma_meta_req
#pragma HLS INTERFACE axis register both port=rdma_meta_rsp
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

	static rdma_meta_upd_t meta_upd;
	static rdma_meta_req_t meta_req;
	static rdma_meta_rsp_t meta_rsp;


	// dummy code
	if (!STREAM_IS_EMPTY(rdma_meta_upd)){
		meta_upd = STREAM_READ(rdma_meta_upd);
	} else if (!STREAM_IS_EMPTY(rdma_meta_req)) {
		meta_req = STREAM_READ(rdma_meta_req);
		STREAM_WRITE(rdma_meta_rsp, meta_rsp);
	}
}