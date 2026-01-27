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
 * @file latency_testbench.cpp
 * @brief Latency measurement infrastructure for ACCL-Q validation
 *
 * This module provides hardware-based latency measurement capabilities
 * for validating sub-microsecond timing requirements of quantum control
 * operations.
 *
 * Features:
 * - High-resolution timestamp capture (2ns resolution at 500 MHz)
 * - Loopback testing with known delays
 * - Histogram generation for jitter analysis
 * - Counter correlation across nodes
 */

#include "quantum_hls_constants.h"
#include "accl_hls.h"

#ifndef ACCL_SYNTHESIS
#include "log.hpp"
#include <iostream>
#include <iomanip>
extern Log logger;
#endif

using namespace std;

// ============================================================================
// Latency Measurement Structures
// ============================================================================

/**
 * Single latency measurement record
 */
struct latency_record_t {
    quantum_counter_t start_time;
    quantum_counter_t end_time;
    ap_uint<16> operation_id;
    ap_uint<8> operation_type;
    ap_uint<8> status;  // 0 = success, non-zero = error code
};

/**
 * Latency histogram bin
 */
struct histogram_bin_t {
    ap_uint<32> count;
    ap_uint<32> min_latency_ns;
    ap_uint<32> max_latency_ns;
};

/**
 * Latency statistics structure
 */
struct latency_stats_hw_t {
    ap_uint<64> total_samples;
    ap_uint<64> sum_latency;      // For mean calculation
    ap_uint<64> sum_sq_latency;   // For std dev calculation
    ap_uint<32> min_latency;
    ap_uint<32> max_latency;
};

// ============================================================================
// Constants
// ============================================================================

#define HISTOGRAM_BINS          64
#define HISTOGRAM_BIN_WIDTH_NS  10   // Each bin covers 10ns
#define MAX_RECORDS             1024
#define LATENCY_OVERFLOW_BIN    (HISTOGRAM_BINS - 1)

// ============================================================================
// Latency Measurement Unit
// ============================================================================

/**
 * @brief Hardware latency measurement unit
 *
 * Captures timestamps at operation start and end, computing latency
 * with clock-cycle precision.
 *
 * @param global_counter    Synchronized global counter input
 * @param op_start          Operation start trigger
 * @param op_end            Operation end trigger
 * @param op_id             Operation identifier
 * @param op_type           Operation type code
 * @param record_out        Output latency record
 * @param record_valid      Record output is valid
 * @param stats_out         Running statistics output
 * @param clear_stats       Clear accumulated statistics
 */
void latency_measurement_unit(
    // Timing inputs
    quantum_counter_t global_counter,

    // Operation triggers
    ap_uint<1> op_start,
    ap_uint<1> op_end,
    ap_uint<16> op_id,
    ap_uint<8> op_type,

    // Outputs
    STREAM<latency_record_t> &record_out,
    latency_stats_hw_t &stats_out,

    // Control
    ap_uint<1> clear_stats,
    ap_uint<1> enable
) {
#pragma HLS INTERFACE ap_none port=global_counter
#pragma HLS INTERFACE ap_none port=op_start
#pragma HLS INTERFACE ap_none port=op_end
#pragma HLS INTERFACE ap_none port=op_id
#pragma HLS INTERFACE ap_none port=op_type
#pragma HLS INTERFACE axis register both port=record_out
#pragma HLS INTERFACE ap_none port=stats_out
#pragma HLS INTERFACE ap_none port=clear_stats
#pragma HLS INTERFACE ap_none port=enable
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    // State for in-flight measurement
    static ap_uint<1> measurement_active = 0;
    static quantum_counter_t start_timestamp = 0;
    static ap_uint<16> current_op_id = 0;
    static ap_uint<8> current_op_type = 0;

    // Running statistics
    static latency_stats_hw_t stats = {0, 0, 0, 0xFFFFFFFF, 0};

    // Clear statistics on request
    if (clear_stats) {
        stats.total_samples = 0;
        stats.sum_latency = 0;
        stats.sum_sq_latency = 0;
        stats.min_latency = 0xFFFFFFFF;
        stats.max_latency = 0;
        measurement_active = 0;
    }

    if (!enable) {
        stats_out = stats;
        return;
    }

    // Capture start timestamp
    if (op_start && !measurement_active) {
        start_timestamp = global_counter;
        current_op_id = op_id;
        current_op_type = op_type;
        measurement_active = 1;

#ifndef ACCL_SYNTHESIS
        std::stringstream ss;
        ss << "Latency Unit: Started measurement for op " << op_id.to_uint()
           << " at time " << global_counter.to_uint64() << "\n";
        logger << log_level::verbose << ss.str();
#endif
    }

    // Capture end timestamp and compute latency
    if (op_end && measurement_active) {
        quantum_counter_t end_timestamp = global_counter;
        ap_uint<32> latency_cycles = end_timestamp - start_timestamp;
        ap_uint<32> latency_ns = latency_cycles * QUANTUM_CLOCK_PERIOD_NS;

        // Create record
        latency_record_t record;
        record.start_time = start_timestamp;
        record.end_time = end_timestamp;
        record.operation_id = current_op_id;
        record.operation_type = current_op_type;
        record.status = 0;  // Success

        STREAM_WRITE(record_out, record);

        // Update statistics
        stats.total_samples++;
        stats.sum_latency += latency_ns;
        stats.sum_sq_latency += (ap_uint<64>)latency_ns * latency_ns;

        if (latency_ns < stats.min_latency) {
            stats.min_latency = latency_ns;
        }
        if (latency_ns > stats.max_latency) {
            stats.max_latency = latency_ns;
        }

        measurement_active = 0;

#ifndef ACCL_SYNTHESIS
        std::stringstream ss;
        ss << "Latency Unit: Completed measurement for op " << current_op_id.to_uint()
           << ", latency = " << latency_ns.to_uint() << " ns\n";
        logger << log_level::verbose << ss.str();
#endif
    }

    stats_out = stats;
}

// ============================================================================
// Histogram Generator
// ============================================================================

/**
 * @brief Generates latency histogram for jitter analysis
 *
 * Bins latency measurements into histogram for visualization
 * and statistical analysis of timing distribution.
 *
 * @param record_in         Input latency records
 * @param histogram         Output histogram bins
 * @param clear             Clear histogram
 */
void histogram_generator(
    STREAM<latency_record_t> &record_in,
    histogram_bin_t histogram[HISTOGRAM_BINS],
    ap_uint<1> clear
) {
#pragma HLS INTERFACE axis register both port=record_in
#pragma HLS INTERFACE ap_memory port=histogram
#pragma HLS INTERFACE ap_none port=clear
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    static histogram_bin_t bins[HISTOGRAM_BINS];
#pragma HLS ARRAY_PARTITION variable=bins complete

    // Clear on request
    if (clear) {
        for (int i = 0; i < HISTOGRAM_BINS; i++) {
#pragma HLS UNROLL
            bins[i].count = 0;
            bins[i].min_latency_ns = i * HISTOGRAM_BIN_WIDTH_NS;
            bins[i].max_latency_ns = (i + 1) * HISTOGRAM_BIN_WIDTH_NS - 1;
        }
    }

    // Process incoming records
    if (!STREAM_IS_EMPTY(record_in)) {
        latency_record_t record = STREAM_READ(record_in);

        // Compute latency in nanoseconds
        ap_uint<32> latency_cycles = record.end_time - record.start_time;
        ap_uint<32> latency_ns = latency_cycles * QUANTUM_CLOCK_PERIOD_NS;

        // Determine bin
        ap_uint<8> bin_idx = latency_ns / HISTOGRAM_BIN_WIDTH_NS;
        if (bin_idx >= HISTOGRAM_BINS) {
            bin_idx = LATENCY_OVERFLOW_BIN;
        }

        bins[bin_idx].count++;
    }

    // Copy to output
    for (int i = 0; i < HISTOGRAM_BINS; i++) {
#pragma HLS UNROLL
        histogram[i] = bins[i];
    }
}

// ============================================================================
// Loopback Tester
// ============================================================================

/**
 * @brief Loopback test generator for latency validation
 *
 * Generates test patterns with known characteristics for
 * round-trip latency measurement.
 *
 * @param start_test        Start test sequence
 * @param test_count        Number of test iterations
 * @param test_data_out     Test data output stream
 * @param test_data_in      Loopback data input stream
 * @param latency_out       Measured round-trip latencies
 * @param test_complete     Test sequence complete
 * @param global_counter    Synchronized global counter
 */
void loopback_tester(
    // Control
    ap_uint<1> start_test,
    ap_uint<16> test_count,
    quantum_counter_t global_counter,

    // Data streams
    STREAM<quantum_data_t> &test_data_out,
    STREAM<quantum_data_t> &test_data_in,

    // Results
    STREAM<ap_uint<32>> &latency_out,
    ap_uint<1> &test_complete,
    ap_uint<16> &tests_completed
) {
#pragma HLS INTERFACE ap_none port=start_test
#pragma HLS INTERFACE ap_none port=test_count
#pragma HLS INTERFACE ap_none port=global_counter
#pragma HLS INTERFACE axis register both port=test_data_out
#pragma HLS INTERFACE axis register both port=test_data_in
#pragma HLS INTERFACE axis register both port=latency_out
#pragma HLS INTERFACE ap_none port=test_complete
#pragma HLS INTERFACE ap_none port=tests_completed
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    typedef enum {
        LB_IDLE,
        LB_SEND,
        LB_WAIT,
        LB_COMPLETE
    } lb_state_t;

    static lb_state_t state = LB_IDLE;
    static ap_uint<16> target_count = 0;
    static ap_uint<16> sent_count = 0;
    static ap_uint<16> received_count = 0;
    static quantum_counter_t send_times[256];  // Circular buffer for timestamps
#pragma HLS ARRAY_PARTITION variable=send_times complete
    static ap_uint<8> send_idx = 0;
    static ap_uint<8> recv_idx = 0;
    static ap_uint<32> timeout_counter = 0;

    const ap_uint<32> TIMEOUT = 100000;  // Timeout in cycles

    test_complete = 0;
    tests_completed = received_count;

    switch (state) {
    case LB_IDLE:
        if (start_test) {
            target_count = test_count;
            sent_count = 0;
            received_count = 0;
            send_idx = 0;
            recv_idx = 0;
            state = LB_SEND;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "Loopback Tester: Starting " << test_count.to_uint() << " iterations\n";
            logger << log_level::info << ss.str();
#endif
        }
        break;

    case LB_SEND:
        if (sent_count < target_count) {
            // Record send time
            send_times[send_idx] = global_counter;

            // Generate test pattern with embedded sequence number
            quantum_data_t test_pattern = 0;
            test_pattern(15, 0) = sent_count;
            test_pattern(31, 16) = 0xCAFE;  // Magic number
            test_pattern(511, 32) = global_counter;  // Timestamp

            STREAM_WRITE(test_data_out, test_pattern);

            sent_count++;
            send_idx++;

            // Move to wait state if we've sent enough
            if (sent_count >= target_count) {
                state = LB_WAIT;
                timeout_counter = 0;
            }
        }
        break;

    case LB_WAIT:
        // Check for loopback responses
        if (!STREAM_IS_EMPTY(test_data_in)) {
            quantum_data_t received = STREAM_READ(test_data_in);

            // Verify magic number
            if (received(31, 16) == 0xCAFE) {
                quantum_counter_t send_time = send_times[recv_idx];
                ap_uint<32> latency_cycles = global_counter - send_time;
                ap_uint<32> latency_ns = latency_cycles * QUANTUM_CLOCK_PERIOD_NS;

                STREAM_WRITE(latency_out, latency_ns);

                received_count++;
                recv_idx++;

#ifndef ACCL_SYNTHESIS
                std::stringstream ss;
                ss << "Loopback Tester: Received " << received_count.to_uint()
                   << "/" << target_count.to_uint()
                   << ", latency = " << latency_ns.to_uint() << " ns\n";
                logger << log_level::verbose << ss.str();
#endif
            }
        }

        // Check completion
        if (received_count >= target_count) {
            state = LB_COMPLETE;
        }

        // Timeout handling
        timeout_counter++;
        if (timeout_counter >= TIMEOUT) {
#ifndef ACCL_SYNTHESIS
            logger << log_level::error << "Loopback Tester: Timeout waiting for responses\n";
#endif
            state = LB_COMPLETE;
        }
        break;

    case LB_COMPLETE:
        test_complete = 1;
        state = LB_IDLE;

#ifndef ACCL_SYNTHESIS
        std::stringstream ss;
        ss << "Loopback Tester: Complete. Received " << received_count.to_uint()
           << " of " << target_count.to_uint() << " responses\n";
        logger << log_level::info << ss.str();
#endif
        break;
    }
}

// ============================================================================
// Counter Correlation Module
// ============================================================================

/**
 * @brief Correlates counter values between two nodes
 *
 * Used to verify clock synchronization by comparing timestamps
 * from different nodes.
 *
 * @param local_counter     Local synchronized counter
 * @param remote_counter    Remote counter value (received via Aurora)
 * @param remote_valid      Remote counter is valid
 * @param offset_out        Calculated offset between counters
 * @param correlation_valid Output: correlation measurement valid
 */
void counter_correlator(
    quantum_counter_t local_counter,
    quantum_counter_t remote_counter,
    ap_uint<1> remote_valid,
    ap_int<32> &offset_out,
    ap_uint<1> &correlation_valid
) {
#pragma HLS INTERFACE ap_none port=local_counter
#pragma HLS INTERFACE ap_none port=remote_counter
#pragma HLS INTERFACE ap_none port=remote_valid
#pragma HLS INTERFACE ap_none port=offset_out
#pragma HLS INTERFACE ap_none port=correlation_valid
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1 style=flp

    static ap_int<32> accumulated_offset = 0;
    static ap_uint<16> sample_count = 0;
    static ap_int<32> min_offset = 0x7FFFFFFF;
    static ap_int<32> max_offset = -0x7FFFFFFF;

    const ap_uint<16> SAMPLES_FOR_VALID = 16;

    if (remote_valid) {
        // Calculate offset (local - remote)
        ap_int<32> current_offset = (ap_int<32>)(local_counter - remote_counter);

        accumulated_offset += current_offset;
        sample_count++;

        if (current_offset < min_offset) min_offset = current_offset;
        if (current_offset > max_offset) max_offset = current_offset;

        if (sample_count >= SAMPLES_FOR_VALID) {
            offset_out = accumulated_offset >> 4;  // Average over 16 samples
            correlation_valid = 1;

            // Reset for next batch
            accumulated_offset = 0;
            sample_count = 0;

#ifndef ACCL_SYNTHESIS
            std::stringstream ss;
            ss << "Counter Correlator: Offset = " << offset_out
               << " cycles, range = [" << min_offset << ", " << max_offset << "]\n";
            logger << log_level::info << ss.str();
#endif

            min_offset = 0x7FFFFFFF;
            max_offset = -0x7FFFFFFF;
        } else {
            correlation_valid = 0;
        }
    } else {
        correlation_valid = 0;
    }
}

// ============================================================================
// Test Bench Main (Simulation Only)
// ============================================================================

#ifndef ACCL_SYNTHESIS
/**
 * @brief Simulation testbench for latency measurement validation
 */
int main() {
    std::cout << "=== ACCL-Q Latency Measurement Testbench ===" << std::endl;
    std::cout << "Clock period: " << QUANTUM_CLOCK_PERIOD_NS << " ns" << std::endl;
    std::cout << "Target P2P latency: " << QUANTUM_P2P_LATENCY_CYCLES * QUANTUM_CLOCK_PERIOD_NS << " ns" << std::endl;
    std::cout << "Target broadcast latency: " << QUANTUM_BCAST_LATENCY_CYCLES * QUANTUM_CLOCK_PERIOD_NS << " ns" << std::endl;
    std::cout << "Target reduce latency: " << QUANTUM_REDUCE_LATENCY_CYCLES * QUANTUM_CLOCK_PERIOD_NS << " ns" << std::endl;

    // Simulate basic latency measurement
    std::cout << "\n--- Testing Latency Measurement Unit ---" << std::endl;

    hls::stream<latency_record_t> records;
    latency_stats_hw_t stats;
    quantum_counter_t counter = 0;

    // Simulate 10 operations with varying latencies
    for (int i = 0; i < 10; i++) {
        quantum_counter_t start = counter;

        // Simulate operation (50-150 cycles)
        int op_latency = 50 + (i * 10);

        latency_measurement_unit(start, 1, 0, i, 1, records, stats, 0, 1);

        counter += op_latency;

        latency_measurement_unit(counter, 0, 1, i, 1, records, stats, 0, 1);

        counter += 10;  // Gap between operations
    }

    std::cout << "Statistics after 10 operations:" << std::endl;
    std::cout << "  Total samples: " << stats.total_samples.to_uint64() << std::endl;
    std::cout << "  Min latency: " << stats.min_latency.to_uint() << " ns" << std::endl;
    std::cout << "  Max latency: " << stats.max_latency.to_uint() << " ns" << std::endl;
    std::cout << "  Mean latency: " << (stats.sum_latency / stats.total_samples).to_uint64() << " ns" << std::endl;

    std::cout << "\n=== Testbench Complete ===" << std::endl;

    return 0;
}
#endif
