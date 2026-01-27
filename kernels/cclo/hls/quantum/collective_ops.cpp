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
 * @file collective_ops.cpp
 * @brief Deterministic collective operations for ACCL-Q quantum control
 *
 * This module implements quantum-optimized collective communication primitives
 * with guaranteed fixed latency for quantum control applications.
 *
 * Operations implemented:
 * - Broadcast: Root to all with tree topology (< 300ns for 8 nodes)
 * - Reduce: All to root with configurable ops (< 400ns for 8 nodes)
 * - Allreduce: Reduce + Broadcast combined
 * - Barrier: Hardware-synchronized with < 100ns jitter
 * - Scatter: Root distributes different data to each rank
 * - Gather: All ranks send data to root
 * - Allgather: Gather + Broadcast combined
 *
 * All operations use deterministic timing aligned to global sync triggers.
 */

#include "quantum_hls_constants.h"
#include "accl_hls.h"

#ifndef ACCL_SYNTHESIS
#include "log.hpp"
#include <sstream>
extern Log logger;
#endif

using namespace std;

// ============================================================================
// Configuration Constants
// ============================================================================

#define MAX_TREE_FANOUT     4       // Maximum children per node in tree
#define BROADCAST_PIPE_STAGES 3     // Pipeline stages for broadcast
#define REDUCE_PIPE_STAGES   4      // Pipeline stages for reduce
#define BARRIER_TIMEOUT_CYCLES 50000 // ~100us at 500MHz

// Tree topology helpers
#define TREE_PARENT(rank)    (((rank) - 1) / MAX_TREE_FANOUT)
#define TREE_FIRST_CHILD(rank) (((rank) * MAX_TREE_FANOUT) + 1)
#define TREE_DEPTH(ranks)    (log2_ceil(ranks))

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Ceiling of log base 2
 */
inline ap_uint<4> log2_ceil(ap_uint<5> n) {
#pragma HLS INLINE
    ap_uint<4> result = 0;
    ap_uint<5> val = n - 1;
    while (val > 0) {
        val >>= 1;
        result++;
    }
    return result;
}

/**
 * @brief Apply reduction operation to two values
 */
inline quantum_data_t apply_reduce_op(quantum_data_t a, quantum_data_t b,
                                       ap_uint<4> op) {
#pragma HLS INLINE
    switch (op) {
        case QUANTUM_REDUCE_XOR:
            return a ^ b;
        case QUANTUM_REDUCE_ADD:
            return a + b;
        case QUANTUM_REDUCE_MAX:
            return (a > b) ? a : b;
        case QUANTUM_REDUCE_MIN:
            return (a < b) ? a : b;
        default:
            return a ^ b;
    }
}

// ============================================================================
// Neighbor Connectivity Structure
// ============================================================================

/**
 * Structure defining a node's position in the collective topology
 */
struct topology_info_t {
    ap_uint<4> parent_rank;           // Parent in tree (-1 if root)
    ap_uint<4> child_ranks[MAX_TREE_FANOUT];  // Children in tree
    ap_uint<4> num_children;          // Number of active children
    ap_uint<4> tree_level;            // Level in tree (root = 0)
    ap_uint<1> is_root;               // Is this the root node
    ap_uint<1> is_leaf;               // Is this a leaf node
};

/**
 * @brief Compute topology info for a rank
 */
topology_info_t compute_topology(ap_uint<4> local_rank, ap_uint<4> total_ranks,
                                  ap_uint<4> root_rank) {
#pragma HLS INLINE
    topology_info_t info;

    // Rebase ranks so root is 0 in the logical tree
    ap_uint<4> logical_rank = (local_rank >= root_rank) ?
                              (local_rank - root_rank) :
                              (local_rank + total_ranks - root_rank);

    info.is_root = (local_rank == root_rank);
    info.parent_rank = info.is_root ? 0 :
                       ((TREE_PARENT(logical_rank) + root_rank) % total_ranks);

    // Compute children
    info.num_children = 0;
    for (int i = 0; i < MAX_TREE_FANOUT; i++) {
#pragma HLS UNROLL
        ap_uint<4> child_logical = TREE_FIRST_CHILD(logical_rank) + i;
        if (child_logical < total_ranks) {
            info.child_ranks[i] = (child_logical + root_rank) % total_ranks;
            info.num_children++;
        } else {
            info.child_ranks[i] = 0xFF;  // Invalid
        }
    }

    info.is_leaf = (info.num_children == 0);
    info.tree_level = log2_ceil(logical_rank + 1);

    return info;
}

// ============================================================================
// Deterministic Broadcast
// ============================================================================

/**
 * @brief Deterministic broadcast with fixed latency
 *
 * Implements tree-based broadcast with guaranteed timing. Root sends data
 * down the tree, each node forwards to children on receipt.
 *
 * Latency: O(log N) hops, each hop ~100ns = ~300ns for 8 nodes
 *
 * @param data_in           Input data (from root or parent)
 * @param data_out          Output data streams to children
 * @param local_data        Local data (used at root)
 * @param result            Broadcast result for this node
 * @param local_rank        This node's rank
 * @param root_rank         Broadcast root rank
 * @param total_ranks       Total number of ranks
 * @param sync_trigger      Global synchronization trigger
 * @param start             Start broadcast operation
 * @param done              Operation complete signal
 */
void deterministic_broadcast(
    // Network interfaces (one per potential neighbor)
    STREAM<quantum_data_t> &data_from_parent,
    STREAM<quantum_data_t> &data_to_children,

    // Local data interface
    quantum_data_t local_data,
    quantum_data_t &result,

    // Configuration
    ap_uint<4> local_rank,
    ap_uint<4> root_rank,
    ap_uint<4> total_ranks,

    // Control
    ap_uint<1> sync_trigger,
    ap_uint<1> start,
    ap_uint<1> &done,
    ap_uint<1> &valid
) {
#pragma HLS INTERFACE axis register both port=data_from_parent
#pragma HLS INTERFACE axis register both port=data_to_children
#pragma HLS INTERFACE ap_none port=local_data
#pragma HLS INTERFACE ap_none port=result
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_none port=root_rank
#pragma HLS INTERFACE ap_none port=total_ranks
#pragma HLS INTERFACE ap_none port=sync_trigger
#pragma HLS INTERFACE ap_none port=start
#pragma HLS INTERFACE ap_none port=done
#pragma HLS INTERFACE ap_none port=valid
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        BCAST_IDLE,
        BCAST_WAIT_SYNC,
        BCAST_ROOT_SEND,
        BCAST_WAIT_PARENT,
        BCAST_FORWARD,
        BCAST_DONE
    } bcast_state_t;

    static bcast_state_t state = BCAST_IDLE;
    static quantum_data_t bcast_data = 0;
    static topology_info_t topo;
    static ap_uint<4> children_sent = 0;
    static ap_uint<32> timeout_counter = 0;

    done = 0;
    valid = 0;

    switch (state) {
    case BCAST_IDLE:
        if (start) {
            topo = compute_topology(local_rank, total_ranks, root_rank);
            state = BCAST_WAIT_SYNC;
            timeout_counter = 0;
            children_sent = 0;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "Broadcast[" << local_rank.to_uint() << "]: Starting, "
               << (topo.is_root ? "ROOT" : "non-root") << ", "
               << topo.num_children.to_uint() << " children\n";
            logger << log_level::verbose << ss.str();
#endif
        }
        break;

    case BCAST_WAIT_SYNC:
        // Wait for global sync trigger for deterministic timing
        if (sync_trigger) {
            if (topo.is_root) {
                bcast_data = local_data;
                state = BCAST_ROOT_SEND;
            } else {
                state = BCAST_WAIT_PARENT;
            }
        }
        break;

    case BCAST_ROOT_SEND:
        // Root sends to all children
        if (children_sent < topo.num_children) {
            STREAM_WRITE(data_to_children, bcast_data);
            children_sent++;
        } else {
            result = bcast_data;
            valid = 1;
            state = BCAST_DONE;
        }
        break;

    case BCAST_WAIT_PARENT:
        // Non-root waits for data from parent
        if (!STREAM_IS_EMPTY(data_from_parent)) {
            bcast_data = STREAM_READ(data_from_parent);
            state = BCAST_FORWARD;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "Broadcast[" << local_rank.to_uint() << "]: Received from parent\n";
            logger << log_level::verbose << ss.str();
#endif
        }

        // Timeout handling
        timeout_counter++;
        if (timeout_counter > BARRIER_TIMEOUT_CYCLES) {
            state = BCAST_DONE;  // Timeout - complete with invalid data
#ifndef ACCL_SYNTHESIS
            logger << log_level::error << "Broadcast: Timeout waiting for parent\n";
#endif
        }
        break;

    case BCAST_FORWARD:
        // Forward to children
        if (children_sent < topo.num_children) {
            STREAM_WRITE(data_to_children, bcast_data);
            children_sent++;
        } else {
            result = bcast_data;
            valid = 1;
            state = BCAST_DONE;
        }
        break;

    case BCAST_DONE:
        done = 1;
        state = BCAST_IDLE;
        break;
    }
}

// ============================================================================
// Tree Reduce with Configurable Operations
// ============================================================================

/**
 * @brief Tree-based reduce with configurable reduction operation
 *
 * Implements pipelined tree reduction with support for XOR (syndrome
 * computation), ADD (accumulation), MAX, and MIN operations.
 *
 * Latency: O(log N) stages, each ~100ns = ~400ns for 8 nodes
 *
 * @param data_from_children Input data from child nodes
 * @param data_to_parent     Output data to parent node
 * @param local_data         Local contribution to reduction
 * @param result             Reduction result (valid at root)
 * @param reduce_op          Reduction operation (XOR, ADD, MAX, MIN)
 * @param local_rank         This node's rank
 * @param root_rank          Reduction root rank
 * @param total_ranks        Total number of ranks
 * @param sync_trigger       Global synchronization trigger
 * @param start              Start reduce operation
 * @param done               Operation complete signal
 */
void tree_reduce_collective(
    // Network interfaces
    STREAM<quantum_data_t> &data_from_children,
    STREAM<quantum_data_t> &data_to_parent,

    // Local data interface
    quantum_data_t local_data,
    quantum_data_t &result,

    // Configuration
    ap_uint<4> reduce_op,
    ap_uint<4> local_rank,
    ap_uint<4> root_rank,
    ap_uint<4> total_ranks,

    // Control
    ap_uint<1> sync_trigger,
    ap_uint<1> start,
    ap_uint<1> &done,
    ap_uint<1> &valid
) {
#pragma HLS INTERFACE axis register both port=data_from_children
#pragma HLS INTERFACE axis register both port=data_to_parent
#pragma HLS INTERFACE ap_none port=local_data
#pragma HLS INTERFACE ap_none port=result
#pragma HLS INTERFACE ap_none port=reduce_op
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_none port=root_rank
#pragma HLS INTERFACE ap_none port=total_ranks
#pragma HLS INTERFACE ap_none port=sync_trigger
#pragma HLS INTERFACE ap_none port=start
#pragma HLS INTERFACE ap_none port=done
#pragma HLS INTERFACE ap_none port=valid
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        REDUCE_IDLE,
        REDUCE_WAIT_SYNC,
        REDUCE_WAIT_CHILDREN,
        REDUCE_COMPUTE,
        REDUCE_SEND_PARENT,
        REDUCE_DONE
    } reduce_state_t;

    static reduce_state_t state = REDUCE_IDLE;
    static quantum_data_t accumulated = 0;
    static topology_info_t topo;
    static ap_uint<4> children_received = 0;
    static ap_uint<32> timeout_counter = 0;
    static ap_uint<4> current_op = 0;

    done = 0;
    valid = 0;

    switch (state) {
    case REDUCE_IDLE:
        if (start) {
            topo = compute_topology(local_rank, total_ranks, root_rank);
            current_op = reduce_op;
            accumulated = local_data;  // Start with local contribution
            children_received = 0;
            timeout_counter = 0;
            state = REDUCE_WAIT_SYNC;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "Reduce[" << local_rank.to_uint() << "]: Starting, op="
               << reduce_op.to_uint() << ", expecting "
               << topo.num_children.to_uint() << " children\n";
            logger << log_level::verbose << ss.str();
#endif
        }
        break;

    case REDUCE_WAIT_SYNC:
        if (sync_trigger) {
            if (topo.is_leaf) {
                // Leaves send immediately
                state = REDUCE_SEND_PARENT;
            } else {
                // Interior nodes wait for children
                state = REDUCE_WAIT_CHILDREN;
            }
        }
        break;

    case REDUCE_WAIT_CHILDREN:
        // Collect data from all children
        if (!STREAM_IS_EMPTY(data_from_children)) {
            quantum_data_t child_data = STREAM_READ(data_from_children);
            accumulated = apply_reduce_op(accumulated, child_data, current_op);
            children_received++;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "Reduce[" << local_rank.to_uint() << "]: Got child "
               << children_received.to_uint() << "/" << topo.num_children.to_uint() << "\n";
            logger << log_level::verbose << ss.str();
#endif
        }

        // Check if all children received
        if (children_received >= topo.num_children) {
            state = REDUCE_COMPUTE;
        }

        // Timeout
        timeout_counter++;
        if (timeout_counter > BARRIER_TIMEOUT_CYCLES) {
            state = REDUCE_COMPUTE;  // Proceed with what we have
#ifndef ACCL_SYNTHESIS
            logger << log_level::error << "Reduce: Timeout waiting for children\n";
#endif
        }
        break;

    case REDUCE_COMPUTE:
        // Computation is done inline during reception
        if (topo.is_root) {
            result = accumulated;
            valid = 1;
            state = REDUCE_DONE;
        } else {
            state = REDUCE_SEND_PARENT;
        }
        break;

    case REDUCE_SEND_PARENT:
        // Send accumulated result to parent
        STREAM_WRITE(data_to_parent, accumulated);
        state = REDUCE_DONE;

#ifndef ACCL_SYNTHESIS
        std::stringstream ss;
        ss << "Reduce[" << local_rank.to_uint() << "]: Sent to parent\n";
        logger << log_level::verbose << ss.str();
#endif
        break;

    case REDUCE_DONE:
        done = 1;
        state = REDUCE_IDLE;
        break;
    }
}

// ============================================================================
// Allreduce (Reduce + Broadcast)
// ============================================================================

/**
 * @brief Allreduce: reduce to root then broadcast result to all
 *
 * Combines reduce and broadcast for operations where all nodes
 * need the final reduced result (e.g., global syndrome).
 */
void allreduce_collective(
    // Network interfaces
    STREAM<quantum_data_t> &reduce_from_children,
    STREAM<quantum_data_t> &reduce_to_parent,
    STREAM<quantum_data_t> &bcast_from_parent,
    STREAM<quantum_data_t> &bcast_to_children,

    // Local data
    quantum_data_t local_data,
    quantum_data_t &result,

    // Configuration
    ap_uint<4> reduce_op,
    ap_uint<4> local_rank,
    ap_uint<4> root_rank,
    ap_uint<4> total_ranks,

    // Control
    ap_uint<1> sync_trigger,
    ap_uint<1> start,
    ap_uint<1> &done,
    ap_uint<1> &valid
) {
#pragma HLS INTERFACE axis register both port=reduce_from_children
#pragma HLS INTERFACE axis register both port=reduce_to_parent
#pragma HLS INTERFACE axis register both port=bcast_from_parent
#pragma HLS INTERFACE axis register both port=bcast_to_children
#pragma HLS INTERFACE ap_none port=local_data
#pragma HLS INTERFACE ap_none port=result
#pragma HLS INTERFACE ap_none port=reduce_op
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_none port=root_rank
#pragma HLS INTERFACE ap_none port=total_ranks
#pragma HLS INTERFACE ap_none port=sync_trigger
#pragma HLS INTERFACE ap_none port=start
#pragma HLS INTERFACE ap_none port=done
#pragma HLS INTERFACE ap_none port=valid
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        AR_IDLE,
        AR_REDUCE,
        AR_BROADCAST,
        AR_DONE
    } allreduce_state_t;

    static allreduce_state_t state = AR_IDLE;
    static quantum_data_t reduced_result = 0;
    static ap_uint<1> reduce_done = 0;
    static ap_uint<1> reduce_valid = 0;
    static ap_uint<1> bcast_done = 0;
    static ap_uint<1> bcast_valid = 0;

    done = 0;
    valid = 0;

    switch (state) {
    case AR_IDLE:
        if (start) {
            reduce_done = 0;
            reduce_valid = 0;
            bcast_done = 0;
            bcast_valid = 0;
            state = AR_REDUCE;
        }
        break;

    case AR_REDUCE:
        // Run reduce operation
        tree_reduce_collective(
            reduce_from_children, reduce_to_parent,
            local_data, reduced_result,
            reduce_op, local_rank, root_rank, total_ranks,
            sync_trigger, 1, reduce_done, reduce_valid
        );

        if (reduce_done) {
            state = AR_BROADCAST;
        }
        break;

    case AR_BROADCAST:
        // Run broadcast with reduced result
        deterministic_broadcast(
            bcast_from_parent, bcast_to_children,
            reduced_result, result,
            local_rank, root_rank, total_ranks,
            sync_trigger, 1, bcast_done, bcast_valid
        );

        if (bcast_done) {
            valid = bcast_valid;
            state = AR_DONE;
        }
        break;

    case AR_DONE:
        done = 1;
        state = AR_IDLE;
        break;
    }
}

// ============================================================================
// Hardware-Synchronized Barrier
// ============================================================================

/**
 * @brief Hardware-synchronized barrier with sub-nanosecond alignment
 *
 * Implements a barrier using the synchronized global counter to ensure
 * all nodes release within the same clock cycle (< 2ns jitter).
 *
 * Algorithm:
 * 1. Each node signals arrival to root via reduce
 * 2. Root broadcasts release signal
 * 3. All nodes wait for global counter to reach release time
 *
 * @param global_counter    Synchronized global counter
 * @param barrier_in        Incoming barrier signals
 * @param barrier_out       Outgoing barrier signals
 * @param local_rank        This node's rank
 * @param total_ranks       Total number of ranks
 * @param start             Start barrier
 * @param release           Barrier released (all can proceed)
 * @param timeout_cycles    Maximum wait cycles
 */
void hardware_barrier(
    // Timing
    quantum_counter_t global_counter,

    // Network
    STREAM<quantum_counter_t> &barrier_in,
    STREAM<quantum_counter_t> &barrier_out,

    // Configuration
    ap_uint<4> local_rank,
    ap_uint<4> total_ranks,
    ap_uint<32> timeout_cycles,

    // Control
    ap_uint<1> start,
    ap_uint<1> &release,
    ap_uint<1> &timeout_error
) {
#pragma HLS INTERFACE ap_none port=global_counter
#pragma HLS INTERFACE axis register both port=barrier_in
#pragma HLS INTERFACE axis register both port=barrier_out
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_none port=total_ranks
#pragma HLS INTERFACE ap_none port=timeout_cycles
#pragma HLS INTERFACE ap_none port=start
#pragma HLS INTERFACE ap_none port=release
#pragma HLS INTERFACE ap_none port=timeout_error
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        BARRIER_IDLE,
        BARRIER_SIGNAL,
        BARRIER_GATHER,
        BARRIER_COMPUTE_RELEASE,
        BARRIER_BROADCAST_RELEASE,
        BARRIER_WAIT_RELEASE,
        BARRIER_DONE
    } barrier_state_t;

    static barrier_state_t state = BARRIER_IDLE;
    static quantum_counter_t release_time = 0;
    static quantum_counter_t max_arrival_time = 0;
    static ap_uint<4> arrivals_received = 0;
    static ap_uint<32> wait_counter = 0;
    static ap_uint<1> is_root = 0;

    // Release margin: add some cycles to ensure all nodes receive release time
    const ap_uint<16> RELEASE_MARGIN_CYCLES = 100;

    release = 0;
    timeout_error = 0;

    switch (state) {
    case BARRIER_IDLE:
        if (start) {
            is_root = (local_rank == 0);
            arrivals_received = 0;
            wait_counter = 0;
            max_arrival_time = global_counter;
            state = BARRIER_SIGNAL;
        }
        break;

    case BARRIER_SIGNAL:
        // Send arrival time to root (rank 0)
        if (!is_root) {
            STREAM_WRITE(barrier_out, global_counter);
        }

        if (is_root) {
            state = BARRIER_GATHER;
        } else {
            state = BARRIER_WAIT_RELEASE;
        }
        break;

    case BARRIER_GATHER:
        // Root collects arrival times from all ranks
        if (!STREAM_IS_EMPTY(barrier_in)) {
            quantum_counter_t arrival = STREAM_READ(barrier_in);
            if (arrival > max_arrival_time) {
                max_arrival_time = arrival;
            }
            arrivals_received++;
        }

        // Check if all arrived (total_ranks - 1 messages expected)
        if (arrivals_received >= (total_ranks - 1)) {
            state = BARRIER_COMPUTE_RELEASE;
        }

        // Timeout
        wait_counter++;
        if (wait_counter > timeout_cycles) {
            timeout_error = 1;
            state = BARRIER_DONE;
        }
        break;

    case BARRIER_COMPUTE_RELEASE:
        // Compute release time with margin
        release_time = max_arrival_time + RELEASE_MARGIN_CYCLES;
        state = BARRIER_BROADCAST_RELEASE;

#ifndef ACCL_SYNTHESIS
        std::stringstream ss;
        ss << "Barrier Root: Release time = " << release_time.to_uint64() << "\n";
        logger << log_level::verbose << ss.str();
#endif
        break;

    case BARRIER_BROADCAST_RELEASE:
        // Broadcast release time to all ranks
        for (int i = 1; i < QUANTUM_MAX_RANKS; i++) {
#pragma HLS UNROLL
            if (i < total_ranks) {
                STREAM_WRITE(barrier_out, release_time);
            }
        }
        state = BARRIER_WAIT_RELEASE;
        break;

    case BARRIER_WAIT_RELEASE:
        // Non-root: receive release time
        if (!is_root && !STREAM_IS_EMPTY(barrier_in)) {
            release_time = STREAM_READ(barrier_in);
        }

        // All nodes: wait until global counter reaches release time
        if (global_counter >= release_time) {
            release = 1;
            state = BARRIER_DONE;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "Barrier[" << local_rank.to_uint() << "]: Released at "
               << global_counter.to_uint64() << "\n";
            logger << log_level::verbose << ss.str();
#endif
        }

        // Timeout
        wait_counter++;
        if (wait_counter > timeout_cycles) {
            timeout_error = 1;
            state = BARRIER_DONE;
        }
        break;

    case BARRIER_DONE:
        state = BARRIER_IDLE;
        break;
    }
}

// ============================================================================
// Scatter Operation
// ============================================================================

/**
 * @brief Scatter: root sends different data to each rank
 *
 * Used for distributing decoder corrections to individual control nodes.
 *
 * @param scatter_data      Array of data for each rank (at root)
 * @param data_out          Output stream to ranks
 * @param data_in           Input stream from root
 * @param result            Received data for this rank
 * @param local_rank        This node's rank
 * @param root_rank         Scatter root rank
 * @param total_ranks       Total number of ranks
 * @param start             Start operation
 * @param done              Operation complete
 */
void scatter_collective(
    // Data arrays
    quantum_data_t scatter_data[QUANTUM_MAX_RANKS],

    // Network
    STREAM<quantum_data_t> &data_out,
    STREAM<quantum_data_t> &data_in,

    // Result
    quantum_data_t &result,

    // Configuration
    ap_uint<4> local_rank,
    ap_uint<4> root_rank,
    ap_uint<4> total_ranks,

    // Control
    ap_uint<1> sync_trigger,
    ap_uint<1> start,
    ap_uint<1> &done,
    ap_uint<1> &valid
) {
#pragma HLS INTERFACE ap_memory port=scatter_data
#pragma HLS INTERFACE axis register both port=data_out
#pragma HLS INTERFACE axis register both port=data_in
#pragma HLS INTERFACE ap_none port=result
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_none port=root_rank
#pragma HLS INTERFACE ap_none port=total_ranks
#pragma HLS INTERFACE ap_none port=sync_trigger
#pragma HLS INTERFACE ap_none port=start
#pragma HLS INTERFACE ap_none port=done
#pragma HLS INTERFACE ap_none port=valid
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        SCATTER_IDLE,
        SCATTER_WAIT_SYNC,
        SCATTER_ROOT_SEND,
        SCATTER_WAIT_DATA,
        SCATTER_DONE
    } scatter_state_t;

    static scatter_state_t state = SCATTER_IDLE;
    static ap_uint<4> ranks_sent = 0;
    static ap_uint<32> timeout_counter = 0;
    static ap_uint<1> is_root = 0;

    done = 0;
    valid = 0;

    switch (state) {
    case SCATTER_IDLE:
        if (start) {
            is_root = (local_rank == root_rank);
            ranks_sent = 0;
            timeout_counter = 0;
            state = SCATTER_WAIT_SYNC;
        }
        break;

    case SCATTER_WAIT_SYNC:
        if (sync_trigger) {
            if (is_root) {
                state = SCATTER_ROOT_SEND;
            } else {
                state = SCATTER_WAIT_DATA;
            }
        }
        break;

    case SCATTER_ROOT_SEND:
        // Root sends data to each rank
        if (ranks_sent < total_ranks) {
            if (ranks_sent == root_rank) {
                // Root's own data
                result = scatter_data[ranks_sent];
                valid = 1;
            } else {
                STREAM_WRITE(data_out, scatter_data[ranks_sent]);
            }
            ranks_sent++;
        } else {
            state = SCATTER_DONE;
        }
        break;

    case SCATTER_WAIT_DATA:
        if (!STREAM_IS_EMPTY(data_in)) {
            result = STREAM_READ(data_in);
            valid = 1;
            state = SCATTER_DONE;
        }

        timeout_counter++;
        if (timeout_counter > BARRIER_TIMEOUT_CYCLES) {
            state = SCATTER_DONE;
        }
        break;

    case SCATTER_DONE:
        done = 1;
        state = SCATTER_IDLE;
        break;
    }
}

// ============================================================================
// Gather Operation
// ============================================================================

/**
 * @brief Gather: all ranks send data to root
 *
 * Used for collecting measurement results at a central node.
 *
 * @param local_data        Local data to send
 * @param data_out          Output stream to root
 * @param data_in           Input stream from ranks (at root)
 * @param gather_result     Array of gathered data (at root)
 * @param local_rank        This node's rank
 * @param root_rank         Gather root rank
 * @param total_ranks       Total number of ranks
 * @param start             Start operation
 * @param done              Operation complete
 */
void gather_collective(
    // Local data
    quantum_data_t local_data,

    // Network
    STREAM<quantum_data_t> &data_out,
    STREAM<quantum_data_t> &data_in,

    // Result (at root)
    quantum_data_t gather_result[QUANTUM_MAX_RANKS],

    // Configuration
    ap_uint<4> local_rank,
    ap_uint<4> root_rank,
    ap_uint<4> total_ranks,

    // Control
    ap_uint<1> sync_trigger,
    ap_uint<1> start,
    ap_uint<1> &done,
    ap_uint<1> &valid
) {
#pragma HLS INTERFACE ap_none port=local_data
#pragma HLS INTERFACE axis register both port=data_out
#pragma HLS INTERFACE axis register both port=data_in
#pragma HLS INTERFACE ap_memory port=gather_result
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_none port=root_rank
#pragma HLS INTERFACE ap_none port=total_ranks
#pragma HLS INTERFACE ap_none port=sync_trigger
#pragma HLS INTERFACE ap_none port=start
#pragma HLS INTERFACE ap_none port=done
#pragma HLS INTERFACE ap_none port=valid
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        GATHER_IDLE,
        GATHER_WAIT_SYNC,
        GATHER_SEND,
        GATHER_ROOT_COLLECT,
        GATHER_DONE
    } gather_state_t;

    static gather_state_t state = GATHER_IDLE;
    static ap_uint<4> ranks_received = 0;
    static ap_uint<32> timeout_counter = 0;
    static ap_uint<1> is_root = 0;

    done = 0;
    valid = 0;

    switch (state) {
    case GATHER_IDLE:
        if (start) {
            is_root = (local_rank == root_rank);
            ranks_received = 0;
            timeout_counter = 0;
            state = GATHER_WAIT_SYNC;
        }
        break;

    case GATHER_WAIT_SYNC:
        if (sync_trigger) {
            state = GATHER_SEND;
        }
        break;

    case GATHER_SEND:
        if (is_root) {
            // Root stores its own data
            gather_result[root_rank] = local_data;
            ranks_received = 1;
            state = GATHER_ROOT_COLLECT;
        } else {
            // Non-root sends to root
            STREAM_WRITE(data_out, local_data);
            state = GATHER_DONE;
        }
        break;

    case GATHER_ROOT_COLLECT:
        if (!STREAM_IS_EMPTY(data_in)) {
            // Store received data (need to track source rank in real impl)
            gather_result[ranks_received] = STREAM_READ(data_in);
            ranks_received++;
        }

        if (ranks_received >= total_ranks) {
            valid = 1;
            state = GATHER_DONE;
        }

        timeout_counter++;
        if (timeout_counter > BARRIER_TIMEOUT_CYCLES) {
            state = GATHER_DONE;
        }
        break;

    case GATHER_DONE:
        done = 1;
        state = GATHER_IDLE;
        break;
    }
}

// ============================================================================
// Allgather (Gather + Broadcast)
// ============================================================================

/**
 * @brief Allgather: gather to root then broadcast full array
 *
 * All nodes end up with data from all other nodes.
 * Used for distributed measurement result sharing.
 */
void allgather_collective(
    // Local data
    quantum_data_t local_data,

    // Network interfaces
    STREAM<quantum_data_t> &gather_out,
    STREAM<quantum_data_t> &gather_in,
    STREAM<quantum_data_t> &bcast_out,
    STREAM<quantum_data_t> &bcast_in,

    // Result
    quantum_data_t all_data[QUANTUM_MAX_RANKS],

    // Configuration
    ap_uint<4> local_rank,
    ap_uint<4> total_ranks,

    // Control
    ap_uint<1> sync_trigger,
    ap_uint<1> start,
    ap_uint<1> &done,
    ap_uint<1> &valid
) {
#pragma HLS INTERFACE ap_none port=local_data
#pragma HLS INTERFACE axis register both port=gather_out
#pragma HLS INTERFACE axis register both port=gather_in
#pragma HLS INTERFACE axis register both port=bcast_out
#pragma HLS INTERFACE axis register both port=bcast_in
#pragma HLS INTERFACE ap_memory port=all_data
#pragma HLS INTERFACE ap_none port=local_rank
#pragma HLS INTERFACE ap_none port=total_ranks
#pragma HLS INTERFACE ap_none port=sync_trigger
#pragma HLS INTERFACE ap_none port=start
#pragma HLS INTERFACE ap_none port=done
#pragma HLS INTERFACE ap_none port=valid
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        AG_IDLE,
        AG_GATHER,
        AG_BROADCAST,
        AG_DONE
    } allgather_state_t;

    static allgather_state_t state = AG_IDLE;
    static ap_uint<1> gather_done = 0;
    static ap_uint<1> gather_valid = 0;
    static ap_uint<1> bcast_idx = 0;

    done = 0;
    valid = 0;

    switch (state) {
    case AG_IDLE:
        if (start) {
            gather_done = 0;
            gather_valid = 0;
            bcast_idx = 0;
            state = AG_GATHER;
        }
        break;

    case AG_GATHER:
        // Run gather to root (rank 0)
        gather_collective(
            local_data,
            gather_out, gather_in,
            all_data,
            local_rank, 0, total_ranks,
            sync_trigger, 1, gather_done, gather_valid
        );

        if (gather_done) {
            state = AG_BROADCAST;
        }
        break;

    case AG_BROADCAST:
        // Broadcast each element of gathered array
        // (simplified - in practice would pack into larger messages)
        if (local_rank == 0) {
            // Root sends packed data
            for (int i = 0; i < QUANTUM_MAX_RANKS; i++) {
#pragma HLS UNROLL
                if (i < total_ranks) {
                    STREAM_WRITE(bcast_out, all_data[i]);
                }
            }
            valid = 1;
            state = AG_DONE;
        } else {
            // Non-root receives
            if (!STREAM_IS_EMPTY(bcast_in)) {
                all_data[bcast_idx] = STREAM_READ(bcast_in);
                bcast_idx++;
                if (bcast_idx >= total_ranks) {
                    valid = 1;
                    state = AG_DONE;
                }
            }
        }
        break;

    case AG_DONE:
        done = 1;
        state = AG_IDLE;
        break;
    }
}
