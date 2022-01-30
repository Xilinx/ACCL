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

#pragma once

#include "ap_axi_sdata.h"

#define DATA_WIDTH 512
#define DEST_WIDTH 8

typedef ap_axiu<DATA_WIDTH, 0, 0, DEST_WIDTH> stream_word;

//define correspondence of
//central datapath switch master/slave
//ports to block design components
#define SWITCH_M_DMA0_WRITE 0
#define SWITCH_M_DMA1_WRITE 1
#define SWITCH_M_ETH_TX     2
#define SWITCH_M_ARITH_OP0  3
#define SWITCH_M_ARITH_OP1  4
#define SWITCH_M_EXT_KRNL   5
#define SWITCH_M_CLANE0     6
#define SWITCH_M_CLANE1     7
#define SWITCH_M_CLANE2     8
#define SWITCH_M_BYPASS     9

#define SWITCH_S_DMA0_READ  0
#define SWITCH_S_DMA1_READ  1
#define SWITCH_S_ETH_RX     2
#define SWITCH_S_ARITH_RES  3
#define SWITCH_S_EXT_KRNL   4
#define SWITCH_S_CLANE0     5
#define SWITCH_S_CLANE1     6
#define SWITCH_S_CLANE2     7

//this is a work-around for hlslib streams not synthesizing
//with vitis hls. Instead, we test a macro definition to select
//between simulation behaviour (use hlslib streams) and
//synthesis behaviour (use hls streams)
//all code using these macros should make sure it doesnt use features
//that aren't supported by both types of streams. For example,
//hlslib stream depths and storage types can't be defined on declaration
#ifdef ACCL_SYNTHESIS
//use hls streams
#include "hls_stream.h"
#define STREAM hls::stream 
#define STREAM_IS_EMPTY(s) s.empty()
#define STREAM_IS_FULL(s) s.full()
#define STREAM_READ(s) s.read()
#define STREAM_WRITE(s, val) s.write(val)
#else
//use hlslib streams
#include "Stream.h"
#include <sstream>
#include <iostream>
#define STREAM hlslib::Stream 
#define STREAM_IS_EMPTY(s) s.IsEmpty()
#define STREAM_IS_FULL(s) s.IsFull()
#define STREAM_READ(s) s.Pop()
#define STREAM_WRITE(s, val) s.Push(val)
#endif