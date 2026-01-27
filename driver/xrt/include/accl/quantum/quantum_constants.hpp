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

#include <cstdint>

namespace ACCL {
namespace Quantum {

/**
 * ACCL-Q (Quantum-optimized ACCL) Configuration Constants
 *
 * These constants define the timing, latency, and synchronization parameters
 * required for quantum control systems operating within qubit coherence times.
 */

// ============================================================================
// Timing and Clock Configuration
// ============================================================================

/** System clock period in nanoseconds (500 MHz default) */
constexpr unsigned int CLOCK_PERIOD_NS = 2;

/** System clock frequency in MHz */
constexpr unsigned int CLOCK_FREQ_MHZ = 500;

/** Maximum supported ranks/nodes in the quantum control system */
constexpr unsigned int MAX_RANKS = 16;

/** Data width for Aurora interface (bits) */
constexpr unsigned int DATA_WIDTH = 512;

/** Bytes per AXI-Stream word */
constexpr unsigned int BYTES_PER_WORD = DATA_WIDTH / 8;

// ============================================================================
// Latency Targets (all values in nanoseconds)
// ============================================================================

/** Target point-to-point latency for Aurora-direct communication */
constexpr unsigned int TARGET_P2P_LATENCY_NS = 200;

/** Target broadcast latency for 8 nodes */
constexpr unsigned int TARGET_BROADCAST_LATENCY_NS = 300;

/** Target reduce latency for 8 nodes */
constexpr unsigned int TARGET_REDUCE_LATENCY_NS = 400;

/** Target allreduce latency for 8 nodes */
constexpr unsigned int TARGET_ALLREDUCE_LATENCY_NS = 400;

/** Maximum acceptable jitter (standard deviation) */
constexpr unsigned int MAX_JITTER_NS = 10;

/** Maximum latency budget for measurement-based feedback */
constexpr unsigned int FEEDBACK_LATENCY_BUDGET_NS = 500;

// ============================================================================
// Aurora 64B/66B Configuration
// ============================================================================

/** Aurora PHY latency (fixed) */
constexpr unsigned int AURORA_PHY_LATENCY_NS = 40;

/** ACCL-Q protocol processing latency (fixed pipeline) */
constexpr unsigned int PROTOCOL_LATENCY_NS = 80;

/** Fiber propagation delay per meter (approximately 5 ns/m) */
constexpr unsigned int FIBER_DELAY_NS_PER_METER = 5;

/** Default fiber length assumption (meters) */
constexpr unsigned int DEFAULT_FIBER_LENGTH_M = 10;

// ============================================================================
// Clock Synchronization Constants
// ============================================================================

/** Counter width for global timestamp (48 bits = ~8.7 years at 500 MHz) */
constexpr unsigned int COUNTER_WIDTH = 48;

/** Maximum acceptable clock phase error in nanoseconds */
constexpr double MAX_PHASE_ERROR_NS = 1.0;

/** Maximum acceptable counter sync error in clock cycles */
constexpr unsigned int MAX_COUNTER_SYNC_ERROR_CYCLES = 2;

/** Sync message marker byte */
constexpr uint8_t SYNC_MARKER = 0xAA;

/** Sync message types */
enum class SyncMessageType : uint8_t {
    COUNTER_REQUEST  = 0x01,
    COUNTER_RESPONSE = 0x02,
    PHASE_ADJUST     = 0x03,
    SYNC_COMPLETE    = 0x04
};

/** Default clock synchronization timeout in microseconds */
constexpr unsigned int SYNC_TIMEOUT_US = 1000;

// ============================================================================
// Pipeline Configuration
// ============================================================================

/** Number of pipeline stages for deterministic CCLO operations */
constexpr unsigned int CCLO_PIPELINE_STAGES = 4;

/** Tree reduction pipeline stages (log2 of MAX_RANKS) */
constexpr unsigned int TREE_REDUCE_STAGES = 4;

/** Fixed cycle count for scheduled operations */
constexpr unsigned int SCHEDULED_OP_CYCLES = 16;

// ============================================================================
// Quantum Control Specific Constants
// ============================================================================

/** Typical T1 relaxation time range (microseconds) */
constexpr unsigned int TYPICAL_T1_MIN_US = 10;
constexpr unsigned int TYPICAL_T1_MAX_US = 1000;

/** Typical T2 dephasing time range (microseconds) */
constexpr unsigned int TYPICAL_T2_MIN_US = 5;
constexpr unsigned int TYPICAL_T2_MAX_US = 500;

/** Maximum measurement readout time (nanoseconds) */
constexpr unsigned int MAX_READOUT_TIME_NS = 1000;

/** Default barrier timeout in nanoseconds */
constexpr unsigned int BARRIER_TIMEOUT_NS = 10000;

// ============================================================================
// Reduce Operation Types
// ============================================================================

/** Supported reduce operations for quantum syndrome computation */
enum class ReduceOp : uint8_t {
    XOR = 0,  // For parity/syndrome computation
    ADD = 1,  // For accumulation
    MAX = 2,  // For finding maximum
    MIN = 3   // For finding minimum
};

// ============================================================================
// Synchronization Modes
// ============================================================================

/** Synchronization mode for collective operations */
enum class SyncMode : uint8_t {
    HARDWARE = 0,  // Use hardware trigger (lowest jitter)
    SOFTWARE = 1,  // Use software barrier (higher jitter)
    NONE     = 2   // No synchronization (for debugging)
};

// ============================================================================
// Operation Modes
// ============================================================================

/** ACCL-Q operation modes */
enum class ACCLMode : uint8_t {
    STANDARD     = 0,  // Standard ACCL behavior (TCP/UDP)
    DETERMINISTIC = 1,  // Deterministic timing mode (Aurora-direct)
    LOW_LATENCY  = 2   // Optimized for minimum latency
};

// ============================================================================
// Notification Types
// ============================================================================

/** Fragment notification types (matching eth_intf.h) */
enum class NotificationType : uint8_t {
    SOM = 0,  // Start of Message
    SOF = 1,  // Start of Fragment
    EOF_TYPE = 2   // End of Fragment
};

// ============================================================================
// Message Types for Quantum Control
// ============================================================================

/** Message types for quantum-specific operations */
enum class QuantumMsgType : uint8_t {
    MEASUREMENT_DATA    = 0x10,  // Qubit measurement results
    SYNDROME_DATA       = 0x11,  // QEC syndrome information
    TRIGGER_SYNC        = 0x12,  // Synchronized trigger request
    PHASE_CORRECTION    = 0x13,  // Phase correction command
    CONDITIONAL_OP      = 0x14   // Conditional operation based on measurement
};

// ============================================================================
// Latency Statistics Structure
// ============================================================================

/** Structure for tracking latency statistics */
struct LatencyStats {
    uint64_t mean_ns;
    uint64_t std_ns;
    uint64_t min_ns;
    uint64_t max_ns;
    uint64_t sample_count;
};

} // namespace Quantum
} // namespace ACCL
