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
#include "zmq_common.h"

zmq_intf_context zmq_client_cmd_intf(unsigned int starting_port, unsigned int local_rank);
zmq_intf_context zmq_client_krnl_intf(  unsigned int starting_port, unsigned int local_rank, 
                                        unsigned int world_size, unsigned int krnl_dest);
zmq_intf_context zmq_client_intf(   unsigned int starting_port, unsigned int local_rank, 
                                    unsigned int world_size, unsigned int krnl_dest);

void zmq_client_startcall(zmq_intf_context *ctx, unsigned int scenario, unsigned int tag, unsigned int count,
                        unsigned int comm, unsigned int root_src_dst, unsigned int function,
                        unsigned int arithcfg_addr, unsigned int compression_flags, unsigned int stream_flags,
                        uint64_t addr_0, uint64_t addr_1, uint64_t addr_2);
void zmq_client_retcall(zmq_intf_context *ctx);
unsigned int zmq_client_cfgread(zmq_intf_context *ctx, unsigned int offset);
void zmq_client_cfgwrite(zmq_intf_context *ctx, unsigned int offset, unsigned int val);
void zmq_client_memread(zmq_intf_context *ctx, uint64_t adr, unsigned int size, uint8_t *data);
void zmq_client_memwrite(zmq_intf_context *ctx, uint64_t adr, unsigned int size, uint8_t *data);