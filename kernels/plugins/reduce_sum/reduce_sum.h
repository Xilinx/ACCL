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

#include "streamdefines.h"
#include "ap_int.h"
#include <stdint.h>


// template<unsigned int data_width, unsigned int dest_width, typename T>
// void stream_add(STREAM<ap_axiu<2*data_width,0,0,dest_width> > & in, STREAM<ap_axiu<data_width,0,0,dest_width> > & out);

void reduce_sum(STREAM<stream_word> & in0, STREAM<stream_word> & in1, STREAM<stream_word> & out);
