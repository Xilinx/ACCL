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
#include "hls_stream.h"
#include "ap_int.h"
#include <stdint.h>

using namespace hls;
using namespace std;

#ifndef DATA_WIDTH
#define DATA_WIDTH 512
#endif

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

template<unsigned int data_width, typename T>
void stream_add(stream<ap_axiu<2*data_width,0,0,0> > & in, stream<ap_axiu<data_width,0,0,0> > & out);

void reduce_sum_float(stream<ap_axiu<2*DATA_WIDTH,0,0,0> > & in, stream<ap_axiu<DATA_WIDTH,0,0,0> > & out);
void reduce_sum_int32_t(stream<ap_axiu<2*DATA_WIDTH,0,0,0> > & in, stream<ap_axiu<DATA_WIDTH,0,0,0> > & out);
void reduce_sum_double(stream<ap_axiu<2*DATA_WIDTH,0,0,0> > & in, stream<ap_axiu<DATA_WIDTH,0,0,0> > & out);
void reduce_sum_int64_t(stream<ap_axiu<2*DATA_WIDTH,0,0,0> > & in, stream<ap_axiu<DATA_WIDTH,0,0,0> > & out);
#ifdef REDUCE_HALF_PRECISION
void reduce_sum_half(stream<ap_axiu<2*DATA_WIDTH,0,0,0> > & in, stream<ap_axiu<DATA_WIDTH,0,0,0> > & out) ;
#endif
