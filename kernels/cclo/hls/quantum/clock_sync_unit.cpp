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
 * @file clock_sync_unit.cpp
 * @brief Clock synchronization module for ACCL-Q quantum control systems
 *
 * This module maintains sub-nanosecond phase alignment and counter
 * synchronization across all nodes in the quantum control system.
 * It uses Aurora 64B/66B link clock compensation sequences for fine
 * synchronization.
 *
 * Key features:
 * - Phase detection between reference clock and system clock
 * - Counter synchronization state machine
 * - Aurora-based sync message protocol
 * - Support for master/slave synchronization topology
 */

#include "quantum_hls_constants.h"
#include "accl_hls.h"

#ifndef ACCL_SYNTHESIS
#include "log.hpp"
extern Log logger;
#endif

using namespace std;

// ============================================================================
// Clock Synchronization State Machine States
// ============================================================================

typedef enum {
    SYNC_IDLE,
    SYNC_SEND_REQUEST,
    SYNC_WAIT_RESPONSE,
    SYNC_ADJUST_COUNTER,
    SYNC_VERIFY,
    SYNC_SYNCHRONIZED
} sync_state_t;

// ============================================================================
// Internal Data Structures
// ============================================================================

/**
 * Phase measurement data for clock alignment
 */
struct phase_data_t {
    ap_int<16> phase_error;       // Measured phase error
    ap_uint<16> sample_count;     // Number of samples for averaging
    bool stable;                  // Phase is stable within tolerance
};

/**
 * Sync round-trip timing data
 */
struct rtt_data_t {
    quantum_counter_t send_time;
    quantum_counter_t recv_time;
    quantum_counter_t remote_time;
    ap_int<32> offset;            // Calculated clock offset
};

// ============================================================================
// Clock Synchronization Unit
// ============================================================================

/**
 * @brief Main clock synchronization function
 *
 * Maintains phase alignment and counter synchronization across nodes.
 * Operates in master or slave mode based on is_master input.
 *
 * @param sys_clk           System clock (implicit in HLS)
 * @param rst_n             Active-low reset
 * @param is_master         True if this node is the sync master
 * @param sync_trigger      Input trigger to initiate sync
 * @param global_counter    Output: synchronized global counter
 * @param sync_valid        Output: true when counter is synchronized
 * @param phase_error       Output: measured phase error (for debugging)
 * @param aurora_rx_data    Input: received sync messages from Aurora
 * @param aurora_rx_valid   Input: aurora RX valid signal
 * @param aurora_tx_data    Output: sync messages to transmit via Aurora
 * @param aurora_tx_valid   Output: aurora TX valid signal
 */
void clock_sync_unit(
    // Control signals
    ap_uint<1> rst_n,
    ap_uint<1> is_master,
    ap_uint<1> sync_trigger,

    // Synchronized counter output
    quantum_counter_t &global_counter,
    ap_uint<1> &sync_valid,
    ap_int<16> &phase_error_out,

    // Aurora interface
    STREAM<ap_uint<64>> &aurora_rx_data,
    STREAM<ap_uint<64>> &aurora_tx_data
) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE ap_none port=rst_n
#pragma HLS INTERFACE ap_none port=is_master
#pragma HLS INTERFACE ap_none port=sync_trigger
#pragma HLS INTERFACE ap_none port=global_counter
#pragma HLS INTERFACE ap_none port=sync_valid
#pragma HLS INTERFACE ap_none port=phase_error_out
#pragma HLS INTERFACE axis register both port=aurora_rx_data
#pragma HLS INTERFACE axis register both port=aurora_tx_data
#pragma HLS PIPELINE II=1 style=flp

    // ========================================================================
    // Static State Variables
    // ========================================================================

    static sync_state_t state = SYNC_IDLE;
    static quantum_counter_t local_counter = 0;
    static quantum_counter_t adjusted_counter = 0;
    static ap_uint<1> is_synchronized = 0;

    // RTT measurement state
    static rtt_data_t rtt = {0, 0, 0, 0};
    static ap_uint<16> sync_attempts = 0;
    static ap_uint<16> timeout_counter = 0;

    // Phase detection state
    static phase_data_t phase = {0, 0, false};

    // Constants
    const ap_uint<16> SYNC_TIMEOUT = 10000;  // Timeout in clock cycles
    const ap_uint<16> MAX_ATTEMPTS = 10;
    const ap_int<16> PHASE_TOLERANCE = 2;    // Acceptable phase error

    // ========================================================================
    // Reset Logic
    // ========================================================================

    if (!rst_n) {
        state = SYNC_IDLE;
        local_counter = 0;
        adjusted_counter = 0;
        is_synchronized = 0;
        sync_attempts = 0;
        timeout_counter = 0;
        rtt.send_time = 0;
        rtt.recv_time = 0;
        rtt.remote_time = 0;
        rtt.offset = 0;
        phase.phase_error = 0;
        phase.sample_count = 0;
        phase.stable = false;
        global_counter = 0;
        sync_valid = 0;
        phase_error_out = 0;
        return;
    }

    // ========================================================================
    // Local Counter Increment
    // ========================================================================

    local_counter = local_counter + 1;

    // ========================================================================
    // Master Mode: Respond to Sync Requests
    // ========================================================================

    if (is_master) {
        // Master is always synchronized
        adjusted_counter = local_counter;
        is_synchronized = 1;

        // Check for incoming sync requests
        if (!STREAM_IS_EMPTY(aurora_rx_data)) {
            ap_uint<64> rx_msg = STREAM_READ(aurora_rx_data);
            quantum_sync_msg_t sync_msg(rx_msg);

            if (sync_msg.is_valid() && sync_msg.msg_type == QUANTUM_MSG_COUNTER_REQ) {
                // Respond with current counter value
                quantum_sync_msg_t response;
                response.marker = QUANTUM_SYNC_MARKER;
                response.msg_type = QUANTUM_MSG_COUNTER_RESP;
                response.payload = local_counter;

                STREAM_WRITE(aurora_tx_data, (ap_uint<64>)response);

#ifndef ACCL_SYNTHESIS
                std::stringstream ss;
                ss << "Clock Sync Master: Responded to sync request with counter = "
                   << local_counter.to_uint64() << "\n";
                logger << log_level::verbose << ss.str();
#endif
            }
        }
    }

    // ========================================================================
    // Slave Mode: State Machine for Synchronization
    // ========================================================================

    else {
        switch (state) {

        case SYNC_IDLE:
            // Wait for sync trigger
            if (sync_trigger && !is_synchronized) {
                state = SYNC_SEND_REQUEST;
                sync_attempts = 0;
                timeout_counter = 0;
            }
            // Continue using adjusted counter if already synced
            break;

        case SYNC_SEND_REQUEST:
            {
                // Send sync request to master
                quantum_sync_msg_t request;
                request.marker = QUANTUM_SYNC_MARKER;
                request.msg_type = QUANTUM_MSG_COUNTER_REQ;
                request.payload = 0;  // Request doesn't need payload

                STREAM_WRITE(aurora_tx_data, (ap_uint<64>)request);

                // Record send time for RTT calculation
                rtt.send_time = local_counter;

                state = SYNC_WAIT_RESPONSE;
                timeout_counter = 0;

#ifndef ACCL_SYNTHESIS
                std::stringstream ss;
                ss << "Clock Sync Slave: Sent sync request at counter = "
                   << local_counter.to_uint64() << "\n";
                logger << log_level::verbose << ss.str();
#endif
            }
            break;

        case SYNC_WAIT_RESPONSE:
            timeout_counter++;

            // Check for response
            if (!STREAM_IS_EMPTY(aurora_rx_data)) {
                ap_uint<64> rx_msg = STREAM_READ(aurora_rx_data);
                quantum_sync_msg_t sync_msg(rx_msg);

                if (sync_msg.is_valid() && sync_msg.msg_type == QUANTUM_MSG_COUNTER_RESP) {
                    rtt.recv_time = local_counter;
                    rtt.remote_time = sync_msg.payload;
                    state = SYNC_ADJUST_COUNTER;

#ifndef ACCL_SYNTHESIS
                    std::stringstream ss;
                    ss << "Clock Sync Slave: Received response, remote_time = "
                       << rtt.remote_time.to_uint64()
                       << ", RTT = " << (rtt.recv_time - rtt.send_time).to_uint64() << "\n";
                    logger << log_level::verbose << ss.str();
#endif
                }
            }

            // Timeout handling
            if (timeout_counter >= SYNC_TIMEOUT) {
                sync_attempts++;
                if (sync_attempts < MAX_ATTEMPTS) {
                    state = SYNC_SEND_REQUEST;
                } else {
                    // Give up, use local counter
                    state = SYNC_IDLE;
#ifndef ACCL_SYNTHESIS
                    logger << log_level::error << "Clock Sync Slave: Sync failed after max attempts\n";
#endif
                }
            }
            break;

        case SYNC_ADJUST_COUNTER:
            {
                // Calculate clock offset using NTP-like algorithm
                // offset = remote_time - local_time + RTT/2
                quantum_counter_t rtt_half = (rtt.recv_time - rtt.send_time) >> 1;
                quantum_counter_t local_time_at_remote = rtt.send_time + rtt_half;

                // Calculate offset (may be negative, so use signed arithmetic)
                rtt.offset = (ap_int<32>)(rtt.remote_time - local_time_at_remote);

                // Apply adjustment
                adjusted_counter = local_counter + rtt.offset;

                state = SYNC_VERIFY;
                timeout_counter = 0;

#ifndef ACCL_SYNTHESIS
                std::stringstream ss;
                ss << "Clock Sync Slave: Calculated offset = " << rtt.offset.to_int()
                   << ", adjusted_counter = " << adjusted_counter.to_uint64() << "\n";
                logger << log_level::verbose << ss.str();
#endif
            }
            break;

        case SYNC_VERIFY:
            // Update adjusted counter each cycle
            adjusted_counter = local_counter + rtt.offset;

            // Perform verification sync to check accuracy
            timeout_counter++;
            if (timeout_counter >= 100) {  // Wait a bit before verifying
                // For now, assume sync is good if we got here
                // In production, would do another round-trip to verify
                state = SYNC_SYNCHRONIZED;
                is_synchronized = 1;

#ifndef ACCL_SYNTHESIS
                logger << log_level::info << "Clock Sync Slave: Synchronization complete\n";
#endif
            }
            break;

        case SYNC_SYNCHRONIZED:
            // Continuously update adjusted counter
            adjusted_counter = local_counter + rtt.offset;

            // Periodically re-sync (e.g., every 2^20 cycles ~= 2ms at 500MHz)
            if ((local_counter & 0xFFFFF) == 0) {
                // Could trigger re-sync here for drift compensation
                // For now, maintain current sync
            }

            // Handle re-sync trigger
            if (sync_trigger) {
                state = SYNC_SEND_REQUEST;
                is_synchronized = 0;
            }
            break;
        }
    }

    // ========================================================================
    // Output Assignment
    // ========================================================================

    global_counter = adjusted_counter;
    sync_valid = is_synchronized;
    phase_error_out = phase.phase_error;
}

// ============================================================================
// Phase Detector Module (for external reference clock)
// ============================================================================

/**
 * @brief Detects phase difference between system clock and reference clock
 *
 * Used when an external reference clock is distributed to all boards.
 * Measures the phase relationship and outputs error for PLL adjustment.
 *
 * @param ref_clk_edge     Rising edge of reference clock (sampled)
 * @param phase_error      Output: phase error measurement
 * @param phase_valid      Output: phase measurement is valid
 */
void phase_detector(
    ap_uint<1> ref_clk_edge,
    ap_int<16> &phase_error,
    ap_uint<1> &phase_valid
) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE ap_none port=ref_clk_edge
#pragma HLS INTERFACE ap_none port=phase_error
#pragma HLS INTERFACE ap_none port=phase_valid
#pragma HLS PIPELINE II=1 style=flp

    static ap_uint<16> cycle_counter = 0;
    static ap_uint<16> ref_edge_counter = 0;
    static ap_uint<1> prev_ref_clk = 0;
    static ap_int<32> accumulated_error = 0;
    static ap_uint<8> sample_count = 0;

    const ap_uint<16> EXPECTED_PERIOD = 50;  // 10 MHz ref in 500 MHz domain
    const ap_uint<8> SAMPLES_FOR_AVG = 64;

    cycle_counter++;

    // Detect rising edge of reference clock
    ap_uint<1> ref_rising_edge = ref_clk_edge && !prev_ref_clk;
    prev_ref_clk = ref_clk_edge;

    if (ref_rising_edge) {
        // Measure deviation from expected period
        ap_int<16> error = (ap_int<16>)ref_edge_counter - (ap_int<16>)EXPECTED_PERIOD;
        accumulated_error += error;
        sample_count++;

        ref_edge_counter = 0;

        if (sample_count >= SAMPLES_FOR_AVG) {
            phase_error = accumulated_error >> 6;  // Divide by 64
            phase_valid = 1;
            accumulated_error = 0;
            sample_count = 0;
        } else {
            phase_valid = 0;
        }
    } else {
        ref_edge_counter++;
        phase_valid = 0;
    }
}

// ============================================================================
// Global Trigger Distribution
// ============================================================================

/**
 * @brief Distributes synchronized triggers across all nodes
 *
 * Ensures all nodes receive triggers with sub-nanosecond alignment
 * by using the synchronized global counter.
 *
 * @param global_counter    Input: synchronized global counter
 * @param trigger_time      Input: scheduled trigger time
 * @param trigger_arm       Input: arm the trigger
 * @param trigger_out       Output: local trigger signal
 * @param trigger_pending   Output: trigger is armed and pending
 */
void trigger_distributor(
    quantum_counter_t global_counter,
    quantum_counter_t trigger_time,
    ap_uint<1> trigger_arm,
    ap_uint<1> &trigger_out,
    ap_uint<1> &trigger_pending
) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE ap_none port=global_counter
#pragma HLS INTERFACE ap_none port=trigger_time
#pragma HLS INTERFACE ap_none port=trigger_arm
#pragma HLS INTERFACE ap_none port=trigger_out
#pragma HLS INTERFACE ap_none port=trigger_pending
#pragma HLS PIPELINE II=1 style=flp

    static ap_uint<1> armed = 0;
    static quantum_counter_t scheduled_time = 0;

    // Arm trigger
    if (trigger_arm && !armed) {
        armed = 1;
        scheduled_time = trigger_time;
    }

    // Fire trigger at scheduled time
    if (armed && global_counter >= scheduled_time) {
        trigger_out = 1;
        armed = 0;
    } else {
        trigger_out = 0;
    }

    trigger_pending = armed;
}
