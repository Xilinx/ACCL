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

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

using namespace std;

#define VADDR_BITS 48
#define LEN_BITS 28
#define DEST_BITS 4
#define PID_BITS 6

template <int DMA_CHANNEL>
void dm_byp_cmd_converter(hls::stream<ap_uint<104>>& dm_cmd,
							hls::stream<ap_uint<96>>& byp_cmd
							)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	if (!dm_cmd.empty())
	{
		ap_uint<96> byp_cmd_word;
		ap_uint<104> dm_cmd_word = dm_cmd.read();

		ap_uint<23> btt = dm_cmd_word(22,0);
		ap_uint<64> saddr = dm_cmd_word(95,32);
		ap_uint<3> tag = dm_cmd_word(99,96); // assuming tag field encodes the host stream/fpga stream information

		byp_cmd_word.range(VADDR_BITS-1,0) = saddr; //vaddr
		byp_cmd_word.range(VADDR_BITS+LEN_BITS-1,VADDR_BITS) = btt; //len
		byp_cmd_word.range(VADDR_BITS+LEN_BITS,VADDR_BITS+LEN_BITS) = tag; //strm
		byp_cmd_word.range(VADDR_BITS+LEN_BITS+1,VADDR_BITS+LEN_BITS+1) = 0; //sync
		byp_cmd_word.range(VADDR_BITS+LEN_BITS+2,VADDR_BITS+LEN_BITS+2) = 1; //ctl
		byp_cmd_word.range(VADDR_BITS+LEN_BITS+3,VADDR_BITS+LEN_BITS+3) = 1; //host
		byp_cmd_word.range(VADDR_BITS+LEN_BITS+4+DEST_BITS-1,VADDR_BITS+LEN_BITS+4) = DMA_CHANNEL; //dest
		byp_cmd_word.range(VADDR_BITS+LEN_BITS+4+DEST_BITS+PID_BITS-1,VADDR_BITS+LEN_BITS+4+DEST_BITS) = 0; //pid
		byp_cmd_word.range(95,VADDR_BITS+LEN_BITS+4+DEST_BITS+PID_BITS) = 0; //vfid+rsvd, disregard
		byp_cmd.write(byp_cmd_word);
	}
}

template <int WIDTH>
void multiplexor(hls::stream<ap_uint<WIDTH>>& in0,
				hls::stream<ap_uint<WIDTH>>& in1,
				hls::stream<ap_uint<WIDTH>>& out)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	ap_uint<WIDTH> currWord;

	if (!in0.empty())
	{
		in0.read(currWord);
		out.write(currWord);
	}
	else if(!in1.empty())
	{
		in1.read(currWord);
		out.write(currWord);
	}

}


void byp_dm_sts_converter(hls::stream<ap_uint<16>> & byp_sts, 
						hls::stream<ap_axiu<32,0,0,0>> & dm0_sts,
						hls::stream<ap_axiu<32,0,0,0>> & dm1_sts)
{
#pragma HLS inline off
#pragma HLS pipeline II=1

	if (!byp_sts.empty())
	{
		ap_uint<16> byp_sts_word = byp_sts.read();
		ap_uint<PID_BITS> pid = byp_sts_word(PID_BITS-1,0);
		ap_uint<1> strm = byp_sts_word(PID_BITS,PID_BITS);
		ap_uint<DEST_BITS> dest = byp_sts_word(DEST_BITS+PID_BITS,PID_BITS+1);

		
		ap_axiu<32,0,0,0> dm_sts_word;
		dm_sts_word.data.range(3,0) = dest; //tag
		dm_sts_word.data.range(4,4) = 0; // internal error
		dm_sts_word.data.range(5,5) = 0; // decode error
		dm_sts_word.data.range(6,6) = 0; // slave error
		dm_sts_word.data.range(7,7) = 1; // OK
		dm_sts_word.data.range(30,8) = 0; // bytes received; this field is not examined in the DMP
		dm_sts_word.data.range(31,31) = 1; // EOP; not examined in the DMP
		dm_sts_word.last = 1;

		if(dest == 0)
		{
			dm0_sts.write(dm_sts_word);
		} 
		else if (dest == 1){
			dm1_sts.write(dm_sts_word);
		}

	}

}


void cyt_dma_adapter(
	//DM command streams
	hls::stream<ap_uint<104>> &dma0_s2mm_cmd,
	hls::stream<ap_uint<104>> &dma1_s2mm_cmd,
	hls::stream<ap_uint<104>> &dma0_mm2s_cmd,
	hls::stream<ap_uint<104>> &dma1_mm2s_cmd,
	//DM status streams
	hls::stream<ap_axiu<32,0,0,0>> &dma0_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_s2mm_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma0_mm2s_sts,
	hls::stream<ap_axiu<32,0,0,0>> &dma1_mm2s_sts,
	//Coyote Bypass interface command and status
	hls::stream<ap_uint<96>> &cyt_byp_wr_cmd,
	hls::stream<ap_uint<16>> &cyt_byp_wr_sts,
	hls::stream<ap_uint<96>> &cyt_byp_rd_cmd,
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
	

	static hls::stream<ap_uint<96> > byp_wr_cmd_0;
    #pragma HLS stream variable=byp_wr_cmd_0 depth=16
	static hls::stream<ap_uint<96> > byp_wr_cmd_1;
    #pragma HLS stream variable=byp_wr_cmd_1 depth=16
	static hls::stream<ap_uint<96> > byp_rd_cmd_0;
    #pragma HLS stream variable=byp_rd_cmd_0 depth=16
	static hls::stream<ap_uint<96> > byp_rd_cmd_1;
    #pragma HLS stream variable=byp_rd_cmd_1 depth=16

	dm_byp_cmd_converter<0>(dma0_s2mm_cmd,byp_wr_cmd_0);
	dm_byp_cmd_converter<1>(dma1_s2mm_cmd,byp_wr_cmd_1);
	multiplexor<96>(byp_wr_cmd_0,byp_wr_cmd_1,cyt_byp_wr_cmd);

	dm_byp_cmd_converter<0>(dma0_mm2s_cmd,byp_rd_cmd_0);
	dm_byp_cmd_converter<1>(dma1_mm2s_cmd,byp_rd_cmd_1);
	multiplexor<96>(byp_rd_cmd_0,byp_rd_cmd_1,cyt_byp_rd_cmd);

	byp_dm_sts_converter(cyt_byp_wr_sts, dma0_s2mm_sts, dma1_s2mm_sts);
	byp_dm_sts_converter(cyt_byp_rd_sts, dma0_mm2s_sts, dma1_mm2s_sts);


}
