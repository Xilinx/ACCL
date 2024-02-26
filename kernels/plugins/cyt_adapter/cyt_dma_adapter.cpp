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

#include "cyt.h"

using namespace std;

template <int DMA_CHANNEL>
void dm_byp_cmd_converter(hls::stream<ap_axiu<104,0,0,DEST_WIDTH>>& dm_cmd,
							hls::stream<cyt_req_t>& byp_cmd,
							hls::stream<ap_uint<1+4+23>>& dm_meta
							)
{
#pragma HLS inline off
#pragma HLS pipeline II=1
	
	if (!STREAM_IS_EMPTY(dm_cmd))
	{
		ap_axiu<104,0,0,DEST_WIDTH> dm_cmd_with_dest = STREAM_READ(dm_cmd);
		ap_uint<104> dm_cmd_word = dm_cmd_with_dest.data;

		ap_uint<23> btt = dm_cmd_word(22,0);
		ap_uint<64> saddr = dm_cmd_word(95,32);
		ap_uint<4> tag = dm_cmd_word(99,96);
		ap_uint<1> strm = dm_cmd_with_dest.dest(2,0); // 1 if targeting host memory, 0 if targeting card memory
		ap_uint<1> ctl = dm_cmd_word(30,30); // ctl field determines if a TLAST must be asserted at the end of the data stream

		cyt_req_t req(0, 0, 0, DMA_CHANNEL, 0, ctl, 0, strm, btt, saddr);
		STREAM_WRITE(byp_cmd, req);

		ap_uint<1+4+23> dm_meta_word;
		dm_meta_word(22,0) = btt;
		dm_meta_word(26,23) = tag;
		dm_meta_word(27,27) = ctl;
		STREAM_WRITE(dm_meta, dm_meta_word);
	}
}

template <int DMA_CHANNEL>
void rdma_req_byp_cmd_converter(
						hls::stream<cyt_req_t>& rdma_req,
						hls::stream<cyt_req_t>& byp_cmd
)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	if(!STREAM_IS_EMPTY(rdma_req)){
		cyt_req_t req = STREAM_READ(rdma_req);
		// TODO:
		// Better mechanism of buffer & proc mapping 
		// Currently has to set the pid to 0, corresponding to coyote_proc instead of any coyote_qproc
		// Every coyote_qproc has a unique physical address in device
		cyt_req_t cmd(req.rsrvd, req.vfid, 0 /*req.pid*/, DMA_CHANNEL, 0, 1, 0, req.stream, req.len, req.vaddr);
		STREAM_WRITE(byp_cmd, cmd);
	}

}

void multiplexor(hls::stream<cyt_req_t>& in0,
				hls::stream<cyt_req_t>& in1,
				hls::stream<cyt_req_t>& out)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	cyt_req_t currWord;

	if (!STREAM_IS_EMPTY(in0))
	{
		currWord = STREAM_READ(in0);
		STREAM_WRITE(out, currWord);
	}
	else if(!STREAM_IS_EMPTY(in1))
	{
		currWord = STREAM_READ(in1);
		STREAM_WRITE(out, currWord);
	}
}

void multiplexor(hls::stream<cyt_req_t>& in0,
				hls::stream<cyt_req_t>& in1,
				hls::stream<cyt_req_t>& in2,
				hls::stream<cyt_req_t>& out)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	cyt_req_t currWord;

	if (!STREAM_IS_EMPTY(in0))
	{
		currWord = STREAM_READ(in0);
		STREAM_WRITE(out, currWord);
	}
	else if(!STREAM_IS_EMPTY(in1))
	{
		currWord = STREAM_READ(in1);
		STREAM_WRITE(out, currWord);
	} 
	else if(!STREAM_IS_EMPTY(in2))
	{
		currWord = STREAM_READ(in2);
		STREAM_WRITE(out, currWord);
	}

}


void byp_dm_sts_converter(hls::stream<ap_uint<16>> & byp_sts, 
						hls::stream<ap_axiu<32,0,0,0>> & dm0_sts,
						hls::stream<ap_axiu<32,0,0,0>> & dm1_sts,
						hls::stream<ap_uint<1+4+23>>& dm0_meta,
						hls::stream<ap_uint<1+4+23>>& dm1_meta)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	if (!STREAM_IS_EMPTY(byp_sts))
	{
		ap_uint<16> byp_sts_word = STREAM_READ(byp_sts);
		// PID in LSB according to Coyote dma_rsp_t:
		ap_uint<CYT_PID_BITS> pid = byp_sts_word(CYT_PID_BITS-1,0);
		ap_uint<CYT_DEST_BITS> dest = byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS-1,CYT_PID_BITS);
		ap_uint<1> strm = byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS,CYT_DEST_BITS+CYT_PID_BITS);
		ap_uint<1> host = byp_sts_word(CYT_DEST_BITS+CYT_PID_BITS+1,CYT_DEST_BITS+CYT_PID_BITS+1);
		
		ap_axiu<32,0,0,0> dm_sts_word;
		ap_uint<1+4+23> dm_meta_word;

		// only send back ack when the byp_sts stems from kernel issued bypass commands
		// if dest == 2, this comes from wr_req/rd_req, no need to forward to data mover
		if(host == 0)
		{
			do{
				if(dest == 0){
					dm_meta_word = STREAM_READ(dm0_meta);
				} else if (dest == 1){
					dm_meta_word = STREAM_READ(dm1_meta);
				}
				dm_sts_word.data.range(3,0) = dm_meta_word(26,23); //tag
				dm_sts_word.data.range(4,4) = 0; // internal error
				dm_sts_word.data.range(5,5) = 0; // decode error
				dm_sts_word.data.range(6,6) = 0; // slave error
				dm_sts_word.data.range(7,7) = 1; // OK
				dm_sts_word.data.range(30,8) = dm_meta_word(22,0); // bytes received
				dm_sts_word.data.range(31,31) = dm_meta_word(27,27); // EOP
				dm_sts_word.last = 1;
				if(dest == 0){
					STREAM_WRITE(dm0_sts, dm_sts_word);
				} else if (dest == 1){
					STREAM_WRITE(dm1_sts, dm_sts_word);
				}
			} while(dm_meta_word(27,27) == 0);
		}
	}

}

// The cyt bypass commands have 3 sources if RDMA is enabled
// 2 DMA channels from the CCLO and the rdma req interface
void cyt_dma_adapter(
	//DM command streams
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma0_s2mm_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma1_s2mm_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma0_mm2s_cmd,
	hls::stream<ap_axiu<104,0,0,DEST_WIDTH>> &dma1_mm2s_cmd,
	//DM status streams
	hls::stream<ap_axiu<32,0,0,0>> &dma0_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma0_mm2s_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_mm2s_sts,
#ifdef ACCL_RDMA
	//RDMA rd_req and wr_req
	hls::stream<cyt_req_t> & rdma_wr_req,
	hls::stream<cyt_req_t> & rdma_rd_req,
#endif
	//Coyote Bypass interface command and status
	hls::stream<cyt_req_t> &cyt_byp_wr_cmd,
	hls::stream<ap_uint<16>> &cyt_byp_wr_sts,
	hls::stream<cyt_req_t> &cyt_byp_rd_cmd,
	hls::stream<ap_uint<16>> &cyt_byp_rd_sts
) {
#pragma HLS INTERFACE axis port=dma0_s2mm_cmd
#pragma HLS INTERFACE axis port=dma1_s2mm_cmd
#pragma HLS INTERFACE axis port=dma0_mm2s_cmd
#pragma HLS INTERFACE axis port=dma1_mm2s_cmd
#pragma HLS INTERFACE axis port=dma0_s2mm_sts
#pragma HLS INTERFACE axis port=dma1_s2mm_sts
#pragma HLS INTERFACE axis port=dma0_mm2s_sts
#pragma HLS INTERFACE axis port=dma1_mm2s_sts
#pragma HLS INTERFACE axis port=cyt_byp_rd_cmd
#pragma HLS INTERFACE axis port=cyt_byp_rd_sts
#pragma HLS INTERFACE axis port=cyt_byp_wr_cmd
#pragma HLS INTERFACE axis port=cyt_byp_wr_sts
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW disable_start_propagation

#pragma HLS aggregate variable=cyt_byp_wr_cmd compact=bit
#pragma HLS aggregate variable=cyt_byp_rd_cmd compact=bit

#ifdef ACCL_RDMA
#pragma HLS INTERFACE axis port=rdma_wr_req
#pragma HLS INTERFACE axis port=rdma_rd_req
#pragma HLS aggregate variable=rdma_wr_req compact=bit
#pragma HLS aggregate variable=rdma_rd_req compact=bit
#endif
	

	static hls::stream<cyt_req_t > byp_wr_cmd_0;
    #pragma HLS stream variable=byp_wr_cmd_0 depth=16
	static hls::stream<cyt_req_t > byp_wr_cmd_1;
    #pragma HLS stream variable=byp_wr_cmd_1 depth=16
	static hls::stream<cyt_req_t > byp_rd_cmd_0;
    #pragma HLS stream variable=byp_rd_cmd_0 depth=16
	static hls::stream<cyt_req_t > byp_rd_cmd_1;
    #pragma HLS stream variable=byp_rd_cmd_1 depth=16

	static hls::stream<ap_uint<1+4+23>> dma0_mm2s_meta;
    #pragma HLS stream variable=dma0_mm2s_meta depth=16
	static hls::stream<ap_uint<1+4+23>> dma1_mm2s_meta;
    #pragma HLS stream variable=dma1_mm2s_meta depth=16
	static hls::stream<ap_uint<1+4+23>> dma0_s2mm_meta;
    #pragma HLS stream variable=dma0_s2mm_meta depth=16
	static hls::stream<ap_uint<1+4+23>> dma1_s2mm_meta;
    #pragma HLS stream variable=dma1_s2mm_meta depth=16

#ifdef ACCL_RDMA
	static hls::stream<cyt_req_t > byp_wr_cmd_2;
    #pragma HLS stream variable=byp_wr_cmd_2 depth=16
	static hls::stream<cyt_req_t > byp_rd_cmd_2;
    #pragma HLS stream variable=byp_rd_cmd_2 depth=16
#endif

	dm_byp_cmd_converter<0>(dma0_s2mm_cmd, byp_wr_cmd_0, dma0_s2mm_meta);
	dm_byp_cmd_converter<1>(dma1_s2mm_cmd, byp_wr_cmd_1, dma1_s2mm_meta);
#ifdef ACCL_RDMA
	rdma_req_byp_cmd_converter<2>(rdma_wr_req, byp_wr_cmd_2);
#endif

#ifdef ACCL_RDMA
	multiplexor(byp_wr_cmd_0,byp_wr_cmd_1,byp_wr_cmd_2,cyt_byp_wr_cmd);
#else
	multiplexor(byp_wr_cmd_0,byp_wr_cmd_1,cyt_byp_wr_cmd);
#endif

	dm_byp_cmd_converter<0>(dma0_mm2s_cmd,byp_rd_cmd_0, dma0_mm2s_meta);
	dm_byp_cmd_converter<1>(dma1_mm2s_cmd,byp_rd_cmd_1, dma1_mm2s_meta);
#ifdef ACCL_RDMA
	rdma_req_byp_cmd_converter<2>(rdma_rd_req, byp_rd_cmd_2);
#endif

#ifdef ACCL_RDMA
	multiplexor(byp_rd_cmd_0,byp_rd_cmd_1,byp_rd_cmd_2,cyt_byp_rd_cmd);
#else
	multiplexor(byp_rd_cmd_0,byp_rd_cmd_1,cyt_byp_rd_cmd);
#endif

	byp_dm_sts_converter(cyt_byp_wr_sts, dma0_s2mm_sts, dma1_s2mm_sts, dma0_s2mm_meta, dma1_s2mm_meta);
	byp_dm_sts_converter(cyt_byp_rd_sts, dma0_mm2s_sts, dma1_mm2s_sts, dma0_mm2s_meta, dma1_mm2s_meta);


}
