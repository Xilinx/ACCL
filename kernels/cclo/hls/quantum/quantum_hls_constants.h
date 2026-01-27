/*******************************************************************************
#  Copyright (C) 2026 ACCL-Q Project Contributors
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
*******************************************************************************/

#pragma once

#include "accl_hls.h"
#include "ap_int.h"

/**
 * ACCL-Q HLS Constants
 *
 * Hardware-specific constants for quantum-optimized FPGA implementation.
 * These are used in the HLS synthesis of Aurora-direct and clock sync modules.
 */

// ============================================================================
// Clock and Timing
// ============================================================================

#define QUANTUM_CLOCK_PERIOD_NS     2       // 500 MHz operation
#define QUANTUM_CLOCK_FREQ_MHZ      500
#define QUANTUM_MAX_RANKS           16
#define QUANTUM_DATA_WIDTH          512
#define QUANTUM_BYTES_PER_WORD      (QUANTUM_DATA_WIDTH / 8)

// ============================================================================
// Pipeline Configuration
// ============================================================================

#define QUANTUM_CCLO_PIPE_STAGES    4
#define QUANTUM_TREE_REDUCE_STAGES  4       // log2(MAX_RANKS)
#define QUANTUM_SCHEDULED_CYCLES    16

// ============================================================================
// Counter and Sync Configuration
// ============================================================================

#define QUANTUM_COUNTER_WIDTH       48
#define QUANTUM_SYNC_MARKER         0xAA
#define QUANTUM_MSG_COUNTER_REQ     0x01
#define QUANTUM_MSG_COUNTER_RESP    0x02
#define QUANTUM_MSG_PHASE_ADJ       0x03
#define QUANTUM_MSG_SYNC_COMPLETE   0x04

// ============================================================================
// Aurora Configuration
// ============================================================================

#define AURORA_LANE_WIDTH           64
#define AURORA_LANES                8       // 8 lanes for 512-bit width
#define AURORA_USER_WIDTH           512

// ============================================================================
// Latency Targets (in clock cycles at 500 MHz)
// ============================================================================

#define QUANTUM_P2P_LATENCY_CYCLES      100     // 200 ns
#define QUANTUM_BCAST_LATENCY_CYCLES    150     // 300 ns
#define QUANTUM_REDUCE_LATENCY_CYCLES   200     // 400 ns
#define QUANTUM_BARRIER_TIMEOUT_CYCLES  5000    // 10 us

// ============================================================================
// Reduce Operations
// ============================================================================

#define QUANTUM_REDUCE_XOR          0
#define QUANTUM_REDUCE_ADD          1
#define QUANTUM_REDUCE_MAX          2
#define QUANTUM_REDUCE_MIN          3

// ============================================================================
// Collective Operations
// ============================================================================

#define QUANTUM_OP_BROADCAST        0
#define QUANTUM_OP_REDUCE           1
#define QUANTUM_OP_ALLREDUCE        2
#define QUANTUM_OP_ALLGATHER        3
#define QUANTUM_OP_SCATTER          4
#define QUANTUM_OP_BARRIER          5

// ============================================================================
// Message Types
// ============================================================================

#define QUANTUM_MSG_MEASUREMENT     0x10
#define QUANTUM_MSG_SYNDROME        0x11
#define QUANTUM_MSG_TRIGGER         0x12
#define QUANTUM_MSG_PHASE_CORR      0x13
#define QUANTUM_MSG_CONDITIONAL     0x14

// ============================================================================
// Sync Header Format (64 bits)
// ============================================================================
// [63:56] = Sync marker (0xAA)
// [55:48] = Message type
// [47:0]  = Counter value or payload

#define SYNC_HDR_MARKER_START       56
#define SYNC_HDR_MARKER_END         63
#define SYNC_HDR_TYPE_START         48
#define SYNC_HDR_TYPE_END           55
#define SYNC_HDR_PAYLOAD_START      0
#define SYNC_HDR_PAYLOAD_END        47

// ============================================================================
// Type Definitions
// ============================================================================

typedef ap_uint<QUANTUM_COUNTER_WIDTH> quantum_counter_t;
typedef ap_uint<QUANTUM_DATA_WIDTH> quantum_data_t;
typedef ap_uint<4> quantum_op_t;
typedef ap_uint<4> quantum_rank_t;
typedef ap_uint<8> quantum_msg_type_t;

// ============================================================================
// Sync Message Structure
// ============================================================================

struct quantum_sync_msg_t {
    ap_uint<8> marker;
    ap_uint<8> msg_type;
    ap_uint<QUANTUM_COUNTER_WIDTH> payload;

    quantum_sync_msg_t() : marker(0), msg_type(0), payload(0) {}

    quantum_sync_msg_t(ap_uint<64> in) {
        marker = in(SYNC_HDR_MARKER_END, SYNC_HDR_MARKER_START);
        msg_type = in(SYNC_HDR_TYPE_END, SYNC_HDR_TYPE_START);
        payload = in(SYNC_HDR_PAYLOAD_END, SYNC_HDR_PAYLOAD_START);
    }

    operator ap_uint<64>() {
        ap_uint<64> ret;
        ret(SYNC_HDR_MARKER_END, SYNC_HDR_MARKER_START) = marker;
        ret(SYNC_HDR_TYPE_END, SYNC_HDR_TYPE_START) = msg_type;
        ret(SYNC_HDR_PAYLOAD_END, SYNC_HDR_PAYLOAD_START) = payload;
        return ret;
    }

    bool is_valid() {
        return marker == QUANTUM_SYNC_MARKER;
    }
};

// ============================================================================
// Measurement Data Structure
// ============================================================================

struct quantum_meas_t {
    ap_uint<32> qubit_id;
    ap_uint<32> timestamp;
    ap_uint<8> outcome;      // 0 or 1
    ap_uint<8> confidence;   // 0-255 confidence level
    ap_uint<16> reserved;

    quantum_meas_t() : qubit_id(0), timestamp(0), outcome(0), confidence(0), reserved(0) {}
};

// ============================================================================
// Collective Operation Request Structure
// ============================================================================

struct quantum_collective_req_t {
    ap_uint<4> op_type;           // Collective operation type
    ap_uint<4> reduce_op;         // Reduce operation (for reduce/allreduce)
    ap_uint<4> root_rank;         // Root rank for rooted operations
    ap_uint<4> local_rank;        // This node's rank
    ap_uint<16> count;            // Element count
    ap_uint<32> flags;            // Operation flags

    quantum_collective_req_t() :
        op_type(0), reduce_op(0), root_rank(0),
        local_rank(0), count(0), flags(0) {}
};
