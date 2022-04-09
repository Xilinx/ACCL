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
#include <zmqpp/zmqpp.hpp>
#include "streamdefines.h"
#include "Stream.h"
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "log.hpp"
#include <vector>

struct zmq_intf_context{
    zmqpp::context context;
    zmqpp::socket *cmd_socket;
    zmqpp::socket *eth_tx_socket;
    zmqpp::socket *eth_rx_socket;
    zmqpp::socket *krnl_tx_socket;
    zmqpp::socket *krnl_rx_socket;
    bool stop = false;
    zmq_intf_context() : context() {}
};

zmq_intf_context zmq_intf(unsigned int starting_port, unsigned int local_rank, unsigned int world_size, bool use_krnl_sockets, Log &log);
void serve_zmq(zmq_intf_context *ctx, uint32_t *cfgmem, std::vector<char> &devicemem, hlslib::Stream<ap_axiu<32,0,0,0> > &cmd, hlslib::Stream<ap_axiu<32,0,0,0> > &sts);
void eth_endpoint_ingress_port(zmq_intf_context *ctx, hlslib::Stream<stream_word > &out);
void eth_endpoint_egress_port(zmq_intf_context *ctx, hlslib::Stream<stream_word > &in, unsigned int local_rank, bool remap_dest);
void krnl_endpoint_ingress_port(zmq_intf_context *ctx, hlslib::Stream<stream_word > &out);
void krnl_endpoint_egress_port(zmq_intf_context *ctx, hlslib::Stream<stream_word > &in);

void zmq_cmd_server(zmq_intf_context *ctx,
                hlslib::Stream<unsigned int> &axilite_rd_addr, hlslib::Stream<unsigned int> &axilite_rd_data,
                hlslib::Stream<unsigned int> &axilite_wr_addr, hlslib::Stream<unsigned int> &axilite_wr_data,
                hlslib::Stream<ap_uint<64> > &aximm_rd_addr, hlslib::Stream<ap_uint<512> > &aximm_rd_data,
                hlslib::Stream<ap_uint<64> > &aximm_wr_addr, hlslib::Stream<ap_uint<512> > &aximm_wr_data, hlslib::Stream<ap_uint<64> > &aximm_wr_strb,
                hlslib::Stream<unsigned int> &callreq, hlslib::Stream<unsigned int> &callack);
void zmq_eth_ingress_server(zmq_intf_context *ctx, hlslib::Stream<stream_word > &out);
void zmq_eth_egress_server(zmq_intf_context *ctx, hlslib::Stream<stream_word > &in, unsigned int local_rank, bool remap_dest);
void zmq_krnl_egress_server(zmq_intf_context *ctx, hlslib::Stream<stream_word > &in);
void zmq_krnl_ingress_server(zmq_intf_context *ctx, hlslib::Stream<stream_word > &out);
