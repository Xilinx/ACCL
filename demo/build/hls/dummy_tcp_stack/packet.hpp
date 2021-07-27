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
#ifndef PACKET_HPP
#define PACKET_HPP

#include "stdint.h"
#include "axi_utils.hpp"

using namespace hls;

template <int W, int HEADER_SIZE>
class packetHeader {
public:
	bool ready;
	uint16_t idx;
	ap_uint<HEADER_SIZE> header;

public:
	packetHeader()
		:ready(false), idx(0) {}
	packetHeader& operator=(const packetHeader& other)
	{
		ready = other.ready;
		idx = other.idx;
		header = other.header;
		return *this;
	}

	void parseWord(ap_uint<W>& w)
	{
		if (ready)
			return;

		if (idx*W+W < HEADER_SIZE)
		{
			header(idx*W+W-1, idx*W) = w;
		}
		else //(idx*W+W >= HEADER_SIZE)
		{
			header(HEADER_SIZE-1, idx*W) = w;
			ready = true;
		}
		idx++;
		/*(header(idx*W+W-1, idx*W) = w;
		if (idx*W+W >= HEADER_SIZE)
		{
			ready = true;
		}*/
	}
	ap_uint<8> consumeWord(ap_uint<W>& w)
	{
		if ((idx+1)*W <= HEADER_SIZE)
		{
			w = header(((idx+1)*W)-1, idx*W);
			idx++;
			return ((HEADER_SIZE - (idx*W)) / 8);
		}
		else if (idx*W < HEADER_SIZE)
		{
			w((HEADER_SIZE%W)-1, 0) = header(HEADER_SIZE-1, idx*W);
			idx++;
			return 0;//(HEADER_SIZE - (idx*W));
		}
		return 0;
	}
	/*bool consumeWord(ap_uint<W>& w)
	{
		if ((idx+2)*W <= HEADER_SIZE)
		{
			w = header(((idx+1)*W)-1, idx*W);
			idx++;
			return false;
			/*if ((idx+1)*W > HEADER_SIZE)
			{
				return true;
			}
			else
			{
				return false;
			}*//*
		}
		else if ((idx+1)*W <= HEADER_SIZE)
		{
			w = header(((idx+1)*W)-1, idx*W);
			idx++;
			return true;
		}
		return true;
	}*/
	/*void consumePartialWord(ap_uint<W>& w)
	{
		if (idx*W < HEADER_SIZE)
		{
			w((HEADER_SIZE%AXI_WIDTH)-1, 0) = header(HEADER_SIZE-1, idx*W);
			idx++;
		}
		//return true;
	}
	/*bool consumeWord(ap_uint<W>& w)
	{
		if ((idx+1)*W <= HEADER_SIZE)
		{
			w = header(((idx+1)*W)-1, idx*W);
			idx++;
			return true;
		}
		return false;
	}
	bool consumePartialWord(ap_uint<W>& w)
	{
		if (idx*W < HEADER_SIZE)
		{
			w((HEADER_SIZE%AXI_WIDTH)-1, 0) = header(HEADER_SIZE-1, idx*W);
			idx++;
			return true;
		}
		return false;
	}*/
	void setRawHeader(ap_uint<HEADER_SIZE> h)
	{
		header = h;
	}
	ap_uint<HEADER_SIZE> getRawHeader()
	{
		return header;
	}
	bool isReady()
	{
		return ready;
	}

	void clear()
	{
#pragma HLS pipeline II=1
		//header = 0;
		ready = false;
		idx = 0;
	}
};

#endif
