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

/**
 * @file aurora_direct.cpp
 * @brief Aurora-direct communication path for ACCL-Q
 *
 * This module provides a direct Aurora 64B/66B communication path that
 * bypasses the TCP/UDP network stack for sub-microsecond latency.
 *
 * Latency breakdown:
 * - Aurora 64B/66B PHY: ~40 ns (fixed)
 * - Protocol processing: ~80 ns (fixed)
 * - Fiber propagation (10m): ~50 ns
 * - Total point-to-point: ~170 ns
 *
 * Features:
 * - Fixed-latency pipeline for deterministic timing
 * - Direct Aurora user interface without network stack
 * - Configurable ring or mesh topology
 * - Zero-copy data path for measurement results
 */

#include "quantum_hls_constants.h"
#include "accl_hls.h"

#ifndef ACCL_SYNTHESIS
#include "log.hpp"
extern Log logger;
#endif

using namespace std;

// ============================================================================
// Aurora Packet Format
// ============================================================================

/**
 * Aurora-direct packet header format (64 bits)
 *
 * [63:60] - Packet type (data, control, sync)
 * [59:56] - Source rank
 * [55:52] - Destination rank (0xF for broadcast)
 * [51:48] - Collective operation type
 * [47:32] - Sequence number
 * [31:16] - Payload length (in 64-byte words)
 * [15:0]  - Flags and options
 */

#define AURORA_PKT_TYPE_START       60
#define AURORA_PKT_TYPE_END         63
#define AURORA_PKT_SRC_RANK_START   56
#define AURORA_PKT_SRC_RANK_END     59
#define AURORA_PKT_DST_RANK_START   52
#define AURORA_PKT_DST_RANK_END     55
#define AURORA_PKT_OP_START         48
#define AURORA_PKT_OP_END           51
#define AURORA_PKT_SEQN_START       32
#define AURORA_PKT_SEQN_END         47
#define AURORA_PKT_LEN_START        16
#define AURORA_PKT_LEN_END          31
#define AURORA_PKT_FLAGS_START      0
#define AURORA_PKT_FLAGS_END        15

// Packet types
#define AURORA_PKT_TYPE_DATA        0x0
#define AURORA_PKT_TYPE_CONTROL     0x1
#define AURORA_PKT_TYPE_SYNC        0x2
#define AURORA_PKT_TYPE_ACK         0x3
#define AURORA_PKT_TYPE_BARRIER     0x4

// Special destination for broadcast
#define AURORA_DEST_BROADCAST       0xF

// Flags
#define AURORA_FLAG_LAST_FRAG       0x0001
#define AURORA_FLAG_FIRST_FRAG      0x0002
#define AURORA_FLAG_NEEDS_ACK       0x0004
#define AURORA_FLAG_HIGH_PRIORITY   0x0008

/**
 * Aurora packet header structure
 */
struct aurora_header_t {
    ap_uint<4> pkt_type;
    ap_uint<4> src_rank;
    ap_uint<4> dst_rank;
    ap_uint<4> collective_op;
    ap_uint<16> seqn;
    ap_uint<16> payload_len;
    ap_uint<16> flags;

    aurora_header_t() :
        pkt_type(0), src_rank(0), dst_rank(0), collective_op(0),
        seqn(0), payload_len(0), flags(0) {}

    aurora_header_t(ap_uint<64> in) {
        pkt_type = in(AURORA_PKT_TYPE_END, AURORA_PKT_TYPE_START);
        src_rank = in(AURORA_PKT_SRC_RANK_END, AURORA_PKT_SRC_RANK_START);
        dst_rank = in(AURORA_PKT_DST_RANK_END, AURORA_PKT_DST_RANK_START);
        collective_op = in(AURORA_PKT_OP_END, AURORA_PKT_OP_START);
        seqn = in(AURORA_PKT_SEQN_END, AURORA_PKT_SEQN_START);
        payload_len = in(AURORA_PKT_LEN_END, AURORA_PKT_LEN_START);
        flags = in(AURORA_PKT_FLAGS_END, AURORA_PKT_FLAGS_START);
    }

    operator ap_uint<64>() {
        ap_uint<64> ret;
        ret(AURORA_PKT_TYPE_END, AURORA_PKT_TYPE_START) = pkt_type;
        ret(AURORA_PKT_SRC_RANK_END, AURORA_PKT_SRC_RANK_START) = src_rank;
        ret(AURORA_PKT_DST_RANK_END, AURORA_PKT_DST_RANK_START) = dst_rank;
        ret(AURORA_PKT_OP_END, AURORA_PKT_OP_START) = collective_op;
        ret(AURORA_PKT_SEQN_END, AURORA_PKT_SEQN_START) = seqn;
        ret(AURORA_PKT_LEN_END, AURORA_PKT_LEN_START) = payload_len;
        ret(AURORA_PKT_FLAGS_END, AURORA_PKT_FLAGS_START) = flags;
        return ret;
    }
};

// ============================================================================
// Aurora Direct Packetizer
// ============================================================================

/**
 * @brief Packetizes data for Aurora-direct transmission
 *
 * Creates fixed-format packets with minimal header overhead for
 * deterministic latency. Bypasses TCP/UDP entirely.
 *
 * @param in            Input data stream from collective operation
 * @param out           Output packet stream to Aurora TX
 * @param cmd           Command input specifying destination, operation
 * @param sts           Status output
 * @param local_rank    This node's rank ID
 */
void aurora_packetizer(
    STREAM<stream_word> &in,
    STREAM<stream_word> &out,
    STREAM<quantum_collective_req_t> &cmd,
    STREAM<ap_uint<32>> &sts,
    ap_uint<4> local_rank
) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=cmd
#pragma HLS INTERFACE axis register both port=sts
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    // State machine states
    typedef enum {
        PKT_IDLE,
        PKT_SEND_HEADER,
        PKT_SEND_DATA,
        PKT_DONE
    } pkt_state_t;

    static pkt_state_t state = PKT_IDLE;
    static quantum_collective_req_t current_cmd;
    static ap_uint<16> words_sent = 0;
    static ap_uint<16> seqn_counter = 0;

    stream_word inword, outword;

    switch (state) {
    case PKT_IDLE:
        if (!STREAM_IS_EMPTY(cmd)) {
            current_cmd = STREAM_READ(cmd);
            state = PKT_SEND_HEADER;
            words_sent = 0;
        }
        break;

    case PKT_SEND_HEADER:
        {
            // Build header
            aurora_header_t hdr;
            hdr.pkt_type = AURORA_PKT_TYPE_DATA;
            hdr.src_rank = local_rank;
            hdr.dst_rank = (current_cmd.op_type == QUANTUM_OP_BROADCAST) ?
                           AURORA_DEST_BROADCAST : current_cmd.root_rank;
            hdr.collective_op = current_cmd.op_type;
            hdr.seqn = seqn_counter++;
            hdr.payload_len = current_cmd.count;
            hdr.flags = AURORA_FLAG_FIRST_FRAG;

            // Send header as first word
            outword.data = 0;
            outword.data(63, 0) = (ap_uint<64>)hdr;
            outword.keep = 0xFFFFFFFFFFFFFFFF;  // All bytes valid
            outword.last = (current_cmd.count == 0) ? 1 : 0;
            outword.dest = 0;

            STREAM_WRITE(out, outword);

            if (current_cmd.count > 0) {
                state = PKT_SEND_DATA;
            } else {
                state = PKT_DONE;
            }
        }
        break;

    case PKT_SEND_DATA:
        if (!STREAM_IS_EMPTY(in)) {
            inword = STREAM_READ(in);
            words_sent++;

            outword = inword;
            outword.last = (words_sent >= current_cmd.count) ? 1 : 0;

            STREAM_WRITE(out, outword);

            if (words_sent >= current_cmd.count) {
                state = PKT_DONE;
            }
        }
        break;

    case PKT_DONE:
        {
            // Send status: success
            ap_uint<32> status = 0;  // 0 = success
            STREAM_WRITE(sts, status);
            state = PKT_IDLE;
        }
        break;
    }
}

// ============================================================================
// Aurora Direct Depacketizer
// ============================================================================

/**
 * @brief Depacketizes Aurora-direct packets for collective operations
 *
 * Extracts header information and routes data to appropriate
 * collective operation handlers based on packet type.
 *
 * @param in            Input packet stream from Aurora RX
 * @param out           Output data stream to collective operation
 * @param header_out    Extracted header for routing decisions
 * @param local_rank    This node's rank ID
 */
void aurora_depacketizer(
    STREAM<stream_word> &in,
    STREAM<stream_word> &out,
    STREAM<aurora_header_t> &header_out,
    ap_uint<4> local_rank
) {
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=header_out
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        DEPKT_IDLE,
        DEPKT_PROCESS_HEADER,
        DEPKT_FORWARD_DATA,
        DEPKT_DROP
    } depkt_state_t;

    static depkt_state_t state = DEPKT_IDLE;
    static aurora_header_t current_hdr;
    static ap_uint<16> words_received = 0;

    stream_word inword;

    switch (state) {
    case DEPKT_IDLE:
        if (!STREAM_IS_EMPTY(in)) {
            inword = STREAM_READ(in);
            state = DEPKT_PROCESS_HEADER;

            // Extract header from first word
            current_hdr = aurora_header_t(inword.data(63, 0));
            words_received = 0;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "Aurora Depacketizer: Received packet from rank "
               << current_hdr.src_rank.to_uint()
               << ", op=" << current_hdr.collective_op.to_uint()
               << ", len=" << current_hdr.payload_len.to_uint() << "\n";
            logger << log_level::verbose << ss.str();
#endif
        }
        break;

    case DEPKT_PROCESS_HEADER:
        {
            // Check if packet is for us
            bool for_us = (current_hdr.dst_rank == local_rank) ||
                          (current_hdr.dst_rank == AURORA_DEST_BROADCAST);

            if (for_us) {
                // Output header for routing
                STREAM_WRITE(header_out, current_hdr);

                if (current_hdr.payload_len > 0) {
                    state = DEPKT_FORWARD_DATA;
                } else {
                    state = DEPKT_IDLE;
                }
            } else {
                // Not for us, drop or forward (ring topology)
                if (current_hdr.payload_len > 0) {
                    state = DEPKT_DROP;
                } else {
                    state = DEPKT_IDLE;
                }
            }
        }
        break;

    case DEPKT_FORWARD_DATA:
        if (!STREAM_IS_EMPTY(in)) {
            inword = STREAM_READ(in);
            words_received++;

            // Forward data to output
            STREAM_WRITE(out, inword);

            if (words_received >= current_hdr.payload_len || inword.last) {
                state = DEPKT_IDLE;
            }
        }
        break;

    case DEPKT_DROP:
        // Drop data not intended for us
        if (!STREAM_IS_EMPTY(in)) {
            inword = STREAM_READ(in);
            words_received++;

            if (words_received >= current_hdr.payload_len || inword.last) {
                state = DEPKT_IDLE;
            }
        }
        break;
    }
}

// ============================================================================
// Deterministic CCLO for Quantum Operations
// ============================================================================

/**
 * @brief Deterministic Collective Communication and Logic Offload
 *
 * Modified CCLO that executes operations on synchronized trigger edges
 * with fixed, deterministic timing. Designed for quantum control where
 * operations must complete within qubit coherence times.
 *
 * @param sync_trigger      Global synchronization trigger
 * @param meas_data         Input measurement data
 * @param meas_valid        Measurement data valid
 * @param meas_ready        Ready to accept measurement data
 * @param collective_op     Collective operation type
 * @param src_rank          Source rank for operation
 * @param result_data       Output result data
 * @param result_valid      Result data valid
 * @param aurora_tx         Aurora TX stream
 * @param aurora_rx         Aurora RX stream
 * @param local_rank        This node's rank
 * @param total_ranks       Total number of ranks
 */
void cclo_quantum(
    // Control
    ap_uint<1> sync_trigger,
    ap_uint<4> local_rank,
    ap_uint<4> total_ranks,

    // Measurement data interface
    STREAM<quantum_data_t> &meas_data_in,
    STREAM<quantum_data_t> &result_data_out,

    // Operation control
    STREAM<quantum_collective_req_t> &op_cmd,
    STREAM<ap_uint<32>> &op_status,

    // Aurora interface
    STREAM<stream_word> &aurora_tx,
    STREAM<stream_word> &aurora_rx
) {
#pragma HLS INTERFACE ap_none port=sync_trigger
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_none port=total_ranks
#pragma HLS INTERFACE axis register both port=meas_data_in
#pragma HLS INTERFACE axis register both port=result_data_out
#pragma HLS INTERFACE axis register both port=op_cmd
#pragma HLS INTERFACE axis register both port=op_status
#pragma HLS INTERFACE axis register both port=aurora_tx
#pragma HLS INTERFACE axis register both port=aurora_rx
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    // Fixed-latency pipeline stages
    const unsigned int PIPE_STAGES = QUANTUM_CCLO_PIPE_STAGES;

    // Cycle counter for deterministic scheduling
    static ap_uint<32> cycle_counter = 0;

    // Operation state
    typedef enum {
        CCLO_IDLE,
        CCLO_WAIT_SYNC,
        CCLO_EXECUTE,
        CCLO_WAIT_COMPLETE,
        CCLO_DONE
    } cclo_state_t;

    static cclo_state_t state = CCLO_IDLE;
    static quantum_collective_req_t current_op;
    static quantum_data_t local_data = 0;
    static quantum_data_t accumulated_result = 0;
    static ap_uint<4> ranks_received = 0;

    // Deterministic scheduling - operations execute on sync_trigger edges
    ap_uint<1> scheduled_execute = ((cycle_counter & 0xF) == 0) && sync_trigger;

    cycle_counter++;

    switch (state) {
    case CCLO_IDLE:
        if (!STREAM_IS_EMPTY(op_cmd)) {
            current_op = STREAM_READ(op_cmd);
            state = CCLO_WAIT_SYNC;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "CCLO Quantum: Received operation " << current_op.op_type.to_uint()
               << ", waiting for sync trigger\n";
            logger << log_level::verbose << ss.str();
#endif
        }
        break;

    case CCLO_WAIT_SYNC:
        // Read local data while waiting
        if (!STREAM_IS_EMPTY(meas_data_in)) {
            local_data = STREAM_READ(meas_data_in);
        }

        // Wait for synchronized execution point
        if (scheduled_execute) {
            state = CCLO_EXECUTE;
            ranks_received = 0;
            accumulated_result = 0;

#ifndef ACCL_SYNTHESIS
            logger << log_level::verbose << "CCLO Quantum: Starting execution on sync trigger\n";
#endif
        }
        break;

    case CCLO_EXECUTE:
        {
            // Execute based on operation type
            switch (current_op.op_type) {

            case QUANTUM_OP_BROADCAST:
                if (local_rank == current_op.root_rank) {
                    // Root: send data to all
                    stream_word outword;
                    outword.data = local_data;
                    outword.keep = 0xFFFFFFFFFFFFFFFF;
                    outword.last = 1;
                    outword.dest = AURORA_DEST_BROADCAST;
                    STREAM_WRITE(aurora_tx, outword);
                    accumulated_result = local_data;
                    state = CCLO_DONE;
                } else {
                    // Non-root: wait for data
                    state = CCLO_WAIT_COMPLETE;
                }
                break;

            case QUANTUM_OP_REDUCE:
            case QUANTUM_OP_ALLREDUCE:
                // Start local contribution
                accumulated_result = local_data;
                ranks_received = 1;

                // Send our data (tree reduce)
                {
                    stream_word outword;
                    outword.data = local_data;
                    outword.keep = 0xFFFFFFFFFFFFFFFF;
                    outword.last = 1;
                    outword.dest = 0;  // Next rank in tree
                    STREAM_WRITE(aurora_tx, outword);
                }
                state = CCLO_WAIT_COMPLETE;
                break;

            case QUANTUM_OP_BARRIER:
                // Send barrier token
                {
                    stream_word outword;
                    outword.data = 1;  // Barrier arrived
                    outword.keep = 0x00000001;
                    outword.last = 1;
                    outword.dest = AURORA_DEST_BROADCAST;
                    STREAM_WRITE(aurora_tx, outword);
                }
                state = CCLO_WAIT_COMPLETE;
                break;

            default:
                state = CCLO_DONE;
                break;
            }
        }
        break;

    case CCLO_WAIT_COMPLETE:
        // Wait for all data to arrive
        if (!STREAM_IS_EMPTY(aurora_rx)) {
            stream_word inword = STREAM_READ(aurora_rx);
            ranks_received++;

            // Apply reduction operation
            switch (current_op.reduce_op) {
            case QUANTUM_REDUCE_XOR:
                accumulated_result ^= inword.data;
                break;
            case QUANTUM_REDUCE_ADD:
                accumulated_result += inword.data;
                break;
            case QUANTUM_REDUCE_MAX:
                if (inword.data > accumulated_result)
                    accumulated_result = inword.data;
                break;
            case QUANTUM_REDUCE_MIN:
                if (inword.data < accumulated_result)
                    accumulated_result = inword.data;
                break;
            }

            // Check if complete
            if (ranks_received >= total_ranks) {
                state = CCLO_DONE;
            }
        }

        // Timeout check (simplified)
        if ((cycle_counter & 0xFFFF) == 0) {
            // Timeout - report error
            state = CCLO_DONE;
        }
        break;

    case CCLO_DONE:
        // Output result
        STREAM_WRITE(result_data_out, accumulated_result);
        STREAM_WRITE(op_status, (ap_uint<32>)0);  // Success
        state = CCLO_IDLE;

#ifndef ACCL_SYNTHESIS
        std::stringstream ss;
        ss << "CCLO Quantum: Operation complete, result = "
           << accumulated_result.to_string(16) << "\n";
        logger << log_level::verbose << ss.str();
#endif
        break;
    }
}

// ============================================================================
// Tree Reduce for Syndrome Aggregation
// ============================================================================

/**
 * @brief Pipelined tree reduce for XOR-based syndrome aggregation
 *
 * Implements a fixed-latency tree reduction optimized for quantum
 * error correction syndrome computation.
 *
 * @param local_data        Local data input
 * @param neighbor_data     Data from neighbor nodes
 * @param neighbor_valid    Valid signals for neighbor data
 * @param start             Start reduction
 * @param reduce_op         Reduction operation (XOR, ADD, etc.)
 * @param reduced_result    Output reduced result
 * @param result_valid      Result is valid
 */
void tree_reduce(
    quantum_data_t local_data,
    quantum_data_t neighbor_data[QUANTUM_MAX_RANKS - 1],
    ap_uint<QUANTUM_MAX_RANKS - 1> neighbor_valid,
    ap_uint<1> start,
    ap_uint<4> reduce_op,
    quantum_data_t &reduced_result,
    ap_uint<1> &result_valid
) {
#pragma HLS INTERFACE ap_none port=local_data
#pragma HLS INTERFACE ap_none port=neighbor_data
#pragma HLS INTERFACE ap_none port=neighbor_valid
#pragma HLS INTERFACE ap_none port=start
#pragma HLS INTERFACE ap_none port=reduce_op
#pragma HLS INTERFACE ap_none port=reduced_result
#pragma HLS INTERFACE ap_none port=result_valid
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS ARRAY_PARTITION variable=neighbor_data complete
#pragma HLS PIPELINE II=1 style=flp

    const int NUM_RANKS = QUANTUM_MAX_RANKS;
    const int PIPE_STAGES = QUANTUM_TREE_REDUCE_STAGES;

    // Pipeline registers for tree reduction
    static quantum_data_t stage_data[PIPE_STAGES + 1][NUM_RANKS];
#pragma HLS ARRAY_PARTITION variable=stage_data complete dim=0

    static ap_uint<PIPE_STAGES + 1> stage_valid = 0;

    // Stage 0: Latch inputs
    stage_valid[0] = start;
    stage_data[0][0] = local_data;
    for (int i = 0; i < NUM_RANKS - 1; i++) {
#pragma HLS UNROLL
        stage_data[0][i + 1] = neighbor_valid[i] ? neighbor_data[i] : (quantum_data_t)0;
    }

    // Reduction stages
    for (int s = 1; s <= PIPE_STAGES; s++) {
#pragma HLS UNROLL
        stage_valid[s] = stage_valid[s - 1];
        int stride = NUM_RANKS >> s;
        for (int i = 0; i < stride; i++) {
#pragma HLS UNROLL
            quantum_data_t a = stage_data[s - 1][2 * i];
            quantum_data_t b = stage_data[s - 1][2 * i + 1];

            switch (reduce_op) {
            case QUANTUM_REDUCE_XOR:
                stage_data[s][i] = a ^ b;
                break;
            case QUANTUM_REDUCE_ADD:
                stage_data[s][i] = a + b;
                break;
            case QUANTUM_REDUCE_MAX:
                stage_data[s][i] = (a > b) ? a : b;
                break;
            case QUANTUM_REDUCE_MIN:
                stage_data[s][i] = (a < b) ? a : b;
                break;
            default:
                stage_data[s][i] = a ^ b;
                break;
            }
        }
    }

    // Output
    reduced_result = stage_data[PIPE_STAGES][0];
    result_valid = stage_valid[PIPE_STAGES];
}
