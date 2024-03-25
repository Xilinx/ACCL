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
#include "accl_hls.h"
#include "Stream.h"
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "log.hpp"
#include <vector>
#include "zmq_common.h"

#ifndef NUM_CTRL_STREAMS
#define NUM_CTRL_STREAMS 1
#endif

#ifndef ACCL_SIM_NUM_BANKS
#define ACCL_SIM_NUM_BANKS 1
#endif

#ifndef ACCL_SIM_MEM_SIZE_KB
#define ACCL_SIM_MEM_SIZE_KB 256
#endif

// Destination ranks are padded in the ZMQ messages with DEST_PADDING digists.
// ZMQ comares the first characters of a message with the subscription string.
// In consequence, without padding, rank 1 will also receive all messages targetted to 
// rank 11 to 19 etc...
#ifndef DEST_PADDING
#define DEST_PADDING 4
#endif

/**
 * @brief Create a server-side interface to the CCLO simulator/emulator, via ZMQ
 * 
 * @param starting_port The first in a sequence of ports that provide access to the simulator/emulator
 * @param local_rank The local rank of this process. This will determine actual ports used by ZMQ
 * @param world_size The total number of ranks
 * @param use_krnl_sockets Flag indicating whether or not to expose kernel data interfaces
 * @param log Logger
 * @return zmq_intf_context A set of ZMQ sockets through which clients can talk to the simulator/emulator
 */
zmq_intf_context zmq_server_intf(unsigned int starting_port, unsigned int local_rank, unsigned int world_size, bool use_krnl_sockets, Log &log);

/**
 * @brief Serve configuration memory, device memory, and call requests
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param cfgmem Pointer to emulated configuration memory
 * @param devicemem Pointer to emulated device memory
 * @param hostmem Pointer to emulated host memory
 * @param cmd Command stream going to emulated CCLO
 * @param sts Status stream coming from emulated CCLO
 */
void serve_zmq(zmq_intf_context *ctx, uint32_t *cfgmem, std::vector<char> &devicemem, std::vector<char> &hostmem, hlslib::Stream<ap_axiu<32,0,0,0>> cmd[NUM_CTRL_STREAMS], hlslib::Stream<ap_axiu<32,0,0,0>> sts[NUM_CTRL_STREAMS]);

/**
 * @brief Serve an input Ethernet port
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param out Stream carrying data received over ZMQ to the emulated CCLO Ethernet input
 */
void eth_endpoint_ingress_port(zmq_intf_context *ctx, hlslib::Stream<stream_word > &out);

/**
 * @brief Serve an output Ethernet port
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param in Stream carrying data from the emulated CCLO Ethernet output, to be send over ZMQ
 * @param local_rank The local rank of this process.
 */
void eth_endpoint_egress_port(zmq_intf_context *ctx, hlslib::Stream<stream_word > &in, unsigned int local_rank);

/**
 * @brief Serve an input streaming data port
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param out Stream carrying data received over ZMQ to the emulated CCLO stream data input
 */
void krnl_endpoint_ingress_port(zmq_intf_context *ctx, hlslib::Stream<stream_word > &out);

/**
 * @brief Serve an output streaming data port
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param in Stream carrying data from the emulated CCLO streaming data output, to be send over ZMQ
 */
void krnl_endpoint_egress_port(zmq_intf_context *ctx, hlslib::Stream<stream_word > &in);

/**
 * @brief Serve commands to a simulator
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param axilite_rd_addr 
 * @param axilite_rd_data 
 * @param axilite_wr_addr 
 * @param axilite_wr_data 
 * @param aximm_rd_addr
 * @param aximm_rd_len 
 * @param aximm_rd_data 
 * @param aximm_wr_addr
 * @param aximm_wr_len 
 * @param aximm_wr_data 
 * @param aximm_wr_strb 
 * @param callreq 
 * @param callack 
 */
void zmq_cmd_server(zmq_intf_context *ctx,
                hlslib::Stream<unsigned int> &axilite_rd_addr, hlslib::Stream<unsigned int> &axilite_rd_data,
                hlslib::Stream<unsigned int> &axilite_wr_addr, hlslib::Stream<unsigned int> &axilite_wr_data,
                hlslib::Stream<ap_uint<64> > &aximm_rd_addr, hlslib::Stream<ap_uint<32> > &aximm_rd_len, hlslib::Stream<ap_uint<512> > &aximm_rd_data,
                hlslib::Stream<ap_uint<64> > &aximm_wr_addr, hlslib::Stream<ap_uint<32> > &aximm_wr_len, hlslib::Stream<ap_uint<512> > &aximm_wr_data, hlslib::Stream<ap_uint<64> > &aximm_wr_strb,
                hlslib::Stream<unsigned int> &callreq, hlslib::Stream<unsigned int> &callack);

/**
 * @brief Run eth_endpoint_ingress_port repeatedly
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param out Stream carrying data received over ZMQ to the emulated CCLO Ethernet input
 */
void zmq_eth_ingress_server(zmq_intf_context *ctx, hlslib::Stream<stream_word > &out);

/**
 * @brief Run eth_endpoint_egress_port repeatedly
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param in Stream carrying data from the emulated CCLO Ethernet output, to be send over ZMQ
 * @param local_rank The local rank of this process.
 * @param remap_dest Activate destination remapping. Set to True when using TCP.
 */
void zmq_eth_egress_server(zmq_intf_context *ctx, hlslib::Stream<stream_word > &in, unsigned int local_rank);

/**
 * @brief Run zmq_krnl_egress_port repeatedly
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param in Stream carrying data from the emulated CCLO streaming data output, to be send over ZMQ
 */
void zmq_krnl_egress_server(zmq_intf_context *ctx, hlslib::Stream<stream_word > &in);

/**
 * @brief Run zmq_krnl_ingress_port repeatedly
 * 
 * @param ctx Pointer to existing ZMQ context
 * @param out Stream carrying data received over ZMQ to the emulated CCLO stream data input
 */
void zmq_krnl_ingress_server(zmq_intf_context *ctx, hlslib::Stream<stream_word > &out);

