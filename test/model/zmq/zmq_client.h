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
#include <vector>

/**
 * @brief Create a command and optionally data interface to the CCLO simulator/emulator, via ZMQ
 *
 * @param starting_port The first in a sequence of ports that provide access to the simulator/emulator
 * @param local_rank The local rank of this process. This will determine actual ports used by ZMQ
 * @param krnl_dest A destination field to attach to data, for routing purposes
 * @param world_size The total number of ranks
 * @return zmq_intf_context A set of ZMQ sockets through which we can talk to the simulator/emulator
 */
zmq_intf_context zmq_client_intf(  unsigned int starting_port, unsigned int local_rank,
                                        const std::vector<unsigned int>& krnl_dest=std::vector<unsigned int>(), unsigned int world_size=0);

/**
 * @brief Initiate an ACCL call via the ZMQ connection to the emulator/simulator. Parameters correspond to CCLO::Options
 *
 * @param ctx Pointer to existing ZMQ context
 *
 */
void zmq_client_startcall(zmq_intf_context *ctx, unsigned int scenario, unsigned int tag, unsigned int count,
                        unsigned int comm, unsigned int root_src_dst, unsigned int function,
                        unsigned int arithcfg_addr, unsigned int compression_flags, unsigned int stream_flags,
                        uint64_t addr_0, uint64_t addr_1, uint64_t addr_2, unsigned int ctrl_id=0);

/**
 * @brief Wait for completion of a previously-initiated ACCL call via the ZMQ connection to the emulator/simulator
 *
 * @param ctx Pointer to existing ZMQ context
 */
void zmq_client_retcall(zmq_intf_context *ctx, unsigned int ctrl_id=0);

/**
 * @brief Read from emulated CCLO local memory
 *
 * @param ctx Pointer to existing ZMQ context
 * @param offset Address in CCLO local memory
 * @return unsigned int Value at specified address in CCLO local memory
 */
unsigned int zmq_client_cfgread(zmq_intf_context *ctx, unsigned int offset);

/**
 * @brief Write to emulated CCLO local memory
 *
 * @param ctx Pointer to existing ZMQ context
 * @param offset Address in CCLO local memory
 * @param val Value to write at specified address in CCLO local memory
 */
void zmq_client_cfgwrite(zmq_intf_context *ctx, unsigned int offset, unsigned int val);

/**
 * @brief Read from emulated device memory
 *
 * @param ctx Pointer to existing ZMQ context
 * @param adr Address in emulated device memory
 * @param size Number of bytes to read
 * @param data Pointer to data
 * @param host Flag indicating if targeting a "host" buffer
 */
void zmq_client_memread(zmq_intf_context *ctx, uint64_t adr, unsigned int size, uint8_t *data, bool host=false);

/**
 * @brief Write to emulated device memory
 *
 * @param ctx Pointer to existing ZMQ context
 * @param adr Address in emulated device memory
 * @param size Number of bytes to read
 * @param data Pointer to data
 * @param host Flag indicating if targeting a "host" buffer
 */
void zmq_client_memwrite(zmq_intf_context *ctx, uint64_t adr, unsigned int size, uint8_t *data, bool host=false);

/**
 * @brief Allocate memory on emulated device
 *
 * @param ctx Pointer to existing ZMQ context
 * @param adr Address in emulated device memory
 * @param size Number of bytes to allocate
 * @param host Flag indicating if targeting a "host" buffer
 */
void zmq_client_memalloc(zmq_intf_context *ctx, uint64_t adr, unsigned int size, bool host=false);

/**
 * @brief Read from CCLO output data stream
 *
 * @param ctx Pointer to existing ZMQ context
 * @return std::vector<uint8_t> A data message
 */
std::vector<uint8_t> zmq_client_strmread(zmq_intf_context *ctx, bool dont_block=false);

/**
 * @brief Write to CCLO input data stream
 *
 * @param ctx Pointer to existing ZMQ context
 * @param val A data message
 * @param dest An identifier for the destination, to enable routing
 */
void zmq_client_strmwrite(zmq_intf_context *ctx, std::vector<uint8_t> val, unsigned int dest=0);
