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

// convert the data mover command (dma) to the cyt_req_t (dma)
// currently all the memory accesses initialized by the CCLO is associated with pid 0 (coyote_proc)
// also we assume a vfid 0 for single cyt region
// the dest field of the dm cmd indicates the host/card accesses
// the dest field is converted to strm flag in the cyt_sq_cmd
// DMA Channel is used to select axis streams, channel 0 and 1 are reserved 
template <int DMA_CHANNEL>
void dm_sq_cmd_converter(hls::stream<ap_axiu<104,0,0,DEST_WIDTH>>& dm_cmd,
							hls::stream<cyt_req_t>& cyt_sq_cmd,
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

		cyt_req_t req(0/*rsrvd_arg*/, 0 /*offs_arg*/, 0 /*host_arg*/, 0 /*actv_arg*/,
              btt/*len_arg*/, saddr /*vaddr_arg*/, ctl /*last_arg*/,
              DMA_CHANNEL /*dest_arg*/, 0 /*pid_arg*/, 0 /*vfid_arg*/,
              0 /*remote_arg*/, 0 /*rdma_arg*/, 0 /*mode_arg*/, strm /*strm_arg*/, 0 /*opcode_arg*/);

		STREAM_WRITE(cyt_sq_cmd, req);

		ap_uint<1+4+23> dm_meta_word;
		dm_meta_word(22,0) = btt;
		dm_meta_word(26,23) = tag;
		dm_meta_word(27,27) = ctl;
		STREAM_WRITE(dm_meta, dm_meta_word);
	}
}

// convert the cyt_rq (rdma) to cyt_sq (dma)
// Channel 2 of the host/card axis stream is reserved for cyt_rq command
// the rq dest field is used to indicate whether this is host/device access, it should be converted to strm field here
// the sq opcode is not relevant as it is targeting dma
template <int DMA_CHANNEL>
void cyt_rq_sq_cmd_converter(
						hls::stream<cyt_req_t>& cyt_rq_cmd,
						hls::stream<cyt_req_t>& cyt_sq_cmd
)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	if(!STREAM_IS_EMPTY(cyt_rq_cmd)){
		cyt_req_t req = STREAM_READ(cyt_rq_cmd);
		
		// Currently has to set the pid to 0, corresponding to coyote_proc instead of any coyote_qproc
		// Because all the buffer allocation within the ACCL driver is associated with the coyote_proc
		// And every coyote_qproc has a unique physical address in device which is different than the coyote_proc
		// Also mark the host flag in the new output command to 0 to indicate the command is issued from the kernel instead of host
		// However, the cq of this command is not processed in the cq_dm_sts_converter as the dest channel is 2
		cyt_req_t cmd(req.rsrvd/*rsrvd_arg*/, req.offs /*offs_arg*/, 0/*host_arg*/, req.actv /*actv_arg*/,
              req.len/*len_arg*/, req.vaddr /*vaddr_arg*/, req.last /*last_arg*/,
              DMA_CHANNEL /*dest_arg*/, 0 /*pid_arg*/, req.vfid /*vfid_arg*/,
              req.remote /*remote_arg*/, req.rdma /*rdma_arg*/, req.mode /*mode_arg*/, req.dest /*strm_arg*/, req.opcode /*opcode_arg*/);

		STREAM_WRITE(cyt_sq_cmd, cmd);
	}

}

void multiplexor(hls::stream<cyt_req_t>& in0,
				hls::stream<cyt_req_t>& in1,
				hls::stream<cyt_req_t>& in2,
				hls::stream<cyt_req_t>& in3,
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
	else if(!STREAM_IS_EMPTY(in3))
	{
		currWord = STREAM_READ(in3);
		STREAM_WRITE(out, currWord);
	}

}


void cq_dm_sts_converter(hls::stream<cyt_ack_t> & cq_sts, 
						hls::stream<ap_axiu<32,0,0,0>> & dm0_sts,
						hls::stream<ap_axiu<32,0,0,0>> & dm1_sts,
						hls::stream<ap_uint<1+4+23>>& dm0_meta,
						hls::stream<ap_uint<1+4+23>>& dm1_meta)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	if (!STREAM_IS_EMPTY(cq_sts))
	{
		cyt_ack_t cq_sts_word = STREAM_READ(cq_sts);

		ap_axiu<32,0,0,0> dm_sts_word;
		ap_uint<1+4+23> dm_meta_word;

		// only process status if it is local memory completion status
		if(cq_sts_word.opcode == CYT_STRM_CARD || cq_sts_word.opcode == CYT_STRM_HOST)
		{
			// only send back ack when the cq_sts stems from kernel issued bypass commands
			// if dest == 2, this comes from wr_req/rd_req, no need to forward to data mover
			if(cq_sts_word.host == 0)
			{
				do{
					if(cq_sts_word.dest == 0){
						dm_meta_word = STREAM_READ(dm0_meta);
					} else if (cq_sts_word.dest == 1){
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
					if(cq_sts_word.dest == 0){
						STREAM_WRITE(dm0_sts, dm_sts_word);
					} else if (cq_sts_word.dest == 1){
						STREAM_WRITE(dm1_sts, dm_sts_word);
					}
				} while(dm_meta_word(27,27) == 0);
			}
		}
	}

}

// The cyt sq commands have 4 sources if RDMA is enabled
// 2 DMA channels from the CCLO, CCLO sq command, and the Cyt rq interface
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

	//Coyote rq rd_req and wr_req
	hls::stream<cyt_req_t> & cyt_rq_wr_cmd,
	hls::stream<cyt_req_t> & cyt_rq_rd_cmd,

	//CCLO sq command
	hls::stream<cyt_req_t >& cclo_sq_wr_cmd,
	hls::stream<cyt_req_t >& cclo_sq_rd_cmd,

	//Coyote sq interface command and cq status
	hls::stream<cyt_req_t> &cyt_sq_wr_cmd,
	hls::stream<cyt_req_t> &cyt_sq_rd_cmd,

	hls::stream<cyt_ack_t> &cyt_cq_wr_sts,
	hls::stream<cyt_ack_t> &cyt_cq_rd_sts
) {
#pragma HLS INTERFACE axis port=dma0_s2mm_cmd
#pragma HLS INTERFACE axis port=dma1_s2mm_cmd
#pragma HLS INTERFACE axis port=dma0_mm2s_cmd
#pragma HLS INTERFACE axis port=dma1_mm2s_cmd
#pragma HLS INTERFACE axis port=dma0_s2mm_sts
#pragma HLS INTERFACE axis port=dma1_s2mm_sts
#pragma HLS INTERFACE axis port=dma0_mm2s_sts
#pragma HLS INTERFACE axis port=dma1_mm2s_sts
#pragma HLS INTERFACE axis port=cyt_sq_rd_cmd
#pragma HLS INTERFACE axis port=cyt_cq_rd_sts
#pragma HLS INTERFACE axis port=cyt_sq_wr_cmd
#pragma HLS INTERFACE axis port=cyt_cq_wr_sts
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW disable_start_propagation

#pragma HLS aggregate variable=cyt_sq_wr_cmd compact=bit
#pragma HLS aggregate variable=cyt_sq_rd_cmd compact=bit
#pragma HLS aggregate variable=cyt_cq_wr_sts compact=bit
#pragma HLS aggregate variable=cyt_cq_rd_sts compact=bit

#pragma HLS INTERFACE axis port=cyt_rq_wr_cmd
#pragma HLS INTERFACE axis port=cyt_rq_rd_cmd
#pragma HLS aggregate variable=cyt_rq_wr_cmd compact=bit
#pragma HLS aggregate variable=cyt_rq_rd_cmd compact=bit

#pragma HLS INTERFACE axis port=cclo_sq_wr_cmd
#pragma HLS aggregate variable=cclo_sq_wr_cmd compact=bit
#pragma HLS INTERFACE axis port=cclo_sq_rd_cmd
#pragma HLS aggregate variable=cclo_sq_rd_cmd compact=bit

	static hls::stream<cyt_req_t > sq_wr_cmd_0;
    #pragma HLS stream variable=sq_wr_cmd_0 depth=16
	static hls::stream<cyt_req_t > sq_wr_cmd_1;
    #pragma HLS stream variable=sq_wr_cmd_1 depth=16
	static hls::stream<cyt_req_t > sq_rd_cmd_0;
    #pragma HLS stream variable=sq_rd_cmd_0 depth=16
	static hls::stream<cyt_req_t > sq_rd_cmd_1;
    #pragma HLS stream variable=sq_rd_cmd_1 depth=16

	static hls::stream<ap_uint<1+4+23>> dma0_mm2s_meta;
    #pragma HLS stream variable=dma0_mm2s_meta depth=16
	static hls::stream<ap_uint<1+4+23>> dma1_mm2s_meta;
    #pragma HLS stream variable=dma1_mm2s_meta depth=16
	static hls::stream<ap_uint<1+4+23>> dma0_s2mm_meta;
    #pragma HLS stream variable=dma0_s2mm_meta depth=16
	static hls::stream<ap_uint<1+4+23>> dma1_s2mm_meta;
    #pragma HLS stream variable=dma1_s2mm_meta depth=16

	static hls::stream<cyt_req_t > sq_wr_cmd_2;
    #pragma HLS stream variable=sq_wr_cmd_2 depth=16
	static hls::stream<cyt_req_t > sq_rd_cmd_2;
    #pragma HLS stream variable=sq_rd_cmd_2 depth=16

	dm_sq_cmd_converter<0>(dma0_s2mm_cmd, sq_wr_cmd_0, dma0_s2mm_meta);
	dm_sq_cmd_converter<1>(dma1_s2mm_cmd, sq_wr_cmd_1, dma1_s2mm_meta);
	cyt_rq_sq_cmd_converter<2>(cyt_rq_wr_cmd, sq_wr_cmd_2);
	multiplexor(cclo_sq_wr_cmd, sq_wr_cmd_0,sq_wr_cmd_1,sq_wr_cmd_2, cyt_sq_wr_cmd);


	dm_sq_cmd_converter<0>(dma0_mm2s_cmd,sq_rd_cmd_0, dma0_mm2s_meta);
	dm_sq_cmd_converter<1>(dma1_mm2s_cmd,sq_rd_cmd_1, dma1_mm2s_meta);
	cyt_rq_sq_cmd_converter<2>(cyt_rq_rd_cmd, sq_rd_cmd_2);
	multiplexor(cclo_sq_rd_cmd, sq_rd_cmd_0,sq_rd_cmd_1,sq_rd_cmd_2, cyt_sq_rd_cmd);

	// handle the completion queue conversion to dm sts
	cq_dm_sts_converter(cyt_cq_wr_sts, dma0_s2mm_sts, dma1_s2mm_sts, dma0_s2mm_meta, dma1_s2mm_meta);
	cq_dm_sts_converter(cyt_cq_rd_sts, dma0_mm2s_sts, dma1_mm2s_sts, dma0_mm2s_meta, dma1_mm2s_meta);


}
