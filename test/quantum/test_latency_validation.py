#!/usr/bin/env python3
"""
ACCL-Q Latency Validation Test Suite

This module provides software-based validation of ACCL-Q latency requirements
for quantum control systems. It includes:
- Latency target verification
- Jitter analysis with histogram generation
- Statistical validation against requirements
- Qubit emulation for realistic testing

Requirements from ACCL_Quantum_Control_Technical_Guide.docx:
- Point-to-point latency: < 200 ns
- Broadcast latency (8 nodes): < 300 ns
- Reduce latency (8 nodes): < 400 ns
- Jitter: < 10 ns standard deviation
- Clock phase alignment: < 1 ns
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time

# ============================================================================
# Constants (matching quantum_constants.hpp)
# ============================================================================

CLOCK_PERIOD_NS = 2  # 500 MHz
MAX_RANKS = 16
DATA_WIDTH = 512

# Latency targets (nanoseconds)
TARGET_P2P_LATENCY_NS = 200
TARGET_BROADCAST_LATENCY_NS = 300
TARGET_REDUCE_LATENCY_NS = 400
TARGET_ALLREDUCE_LATENCY_NS = 400
MAX_JITTER_NS = 10
FEEDBACK_LATENCY_BUDGET_NS = 500

# Component latencies
AURORA_PHY_LATENCY_NS = 40
PROTOCOL_LATENCY_NS = 80
FIBER_DELAY_NS_PER_METER = 5


# ============================================================================
# Data Structures
# ============================================================================

class ReduceOp(Enum):
    """Supported reduce operations"""
    XOR = 0
    ADD = 1
    MAX = 2
    MIN = 3


class SyncMode(Enum):
    """Synchronization modes"""
    HARDWARE = 0
    SOFTWARE = 1
    NONE = 2


@dataclass
class LatencyStats:
    """Latency statistics structure"""
    mean_ns: float
    std_ns: float
    min_ns: float
    max_ns: float
    sample_count: int
    histogram: Optional[np.ndarray] = None
    bin_edges: Optional[np.ndarray] = None


@dataclass
class LatencyTarget:
    """Latency target specification"""
    name: str
    target_ns: float
    max_jitter_ns: float


# ============================================================================
# Latency Calculation Functions
# ============================================================================

def calculate_p2p_latency(fiber_length_m: float = 10.0) -> float:
    """
    Calculate point-to-point latency for Aurora-direct communication.

    Args:
        fiber_length_m: Fiber optic cable length in meters

    Returns:
        Total latency in nanoseconds
    """
    fiber_delay = fiber_length_m * FIBER_DELAY_NS_PER_METER
    total = AURORA_PHY_LATENCY_NS + PROTOCOL_LATENCY_NS + fiber_delay
    return total


def calculate_broadcast_latency(num_ranks: int, fiber_length_m: float = 10.0) -> float:
    """
    Calculate broadcast latency for N ranks.

    In a ring topology, broadcast takes (N-1) hops.
    In optimized tree topology, it takes log2(N) hops.

    Args:
        num_ranks: Number of ranks in the system
        fiber_length_m: Fiber length between nodes

    Returns:
        Total broadcast latency in nanoseconds
    """
    p2p = calculate_p2p_latency(fiber_length_m)
    # Using tree topology for optimal latency
    hops = int(np.ceil(np.log2(num_ranks)))
    return p2p * hops


def calculate_reduce_latency(num_ranks: int, fiber_length_m: float = 10.0) -> float:
    """
    Calculate tree-reduce latency for N ranks.

    Args:
        num_ranks: Number of ranks in the system
        fiber_length_m: Fiber length between nodes

    Returns:
        Total reduce latency in nanoseconds
    """
    p2p = calculate_p2p_latency(fiber_length_m)
    # Tree reduce has log2(N) stages
    stages = int(np.ceil(np.log2(num_ranks)))
    # Each stage adds one hop latency plus computation time
    compute_per_stage = 10  # ~10ns for XOR/ADD operation
    return stages * (p2p + compute_per_stage)


# ============================================================================
# Latency Measurement Emulation
# ============================================================================

class LatencyMeasurementUnit:
    """
    Software emulation of hardware latency measurement unit.
    """

    def __init__(self):
        self.records: List[Dict] = []
        self.stats = LatencyStats(
            mean_ns=0, std_ns=0, min_ns=float('inf'),
            max_ns=0, sample_count=0
        )

    def measure(self, start_time_ns: float, end_time_ns: float,
                op_id: int, op_type: str) -> Dict:
        """Record a latency measurement."""
        latency = end_time_ns - start_time_ns

        record = {
            'start_time': start_time_ns,
            'end_time': end_time_ns,
            'latency_ns': latency,
            'op_id': op_id,
            'op_type': op_type
        }
        self.records.append(record)

        # Update running statistics
        n = len(self.records)
        latencies = [r['latency_ns'] for r in self.records]

        self.stats = LatencyStats(
            mean_ns=np.mean(latencies),
            std_ns=np.std(latencies),
            min_ns=np.min(latencies),
            max_ns=np.max(latencies),
            sample_count=n
        )

        return record

    def get_histogram(self, bin_width_ns: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate latency histogram."""
        latencies = [r['latency_ns'] for r in self.records]
        max_latency = max(latencies) if latencies else 1000
        bins = np.arange(0, max_latency + bin_width_ns, bin_width_ns)
        hist, edges = np.histogram(latencies, bins=bins)
        self.stats.histogram = hist
        self.stats.bin_edges = edges
        return hist, edges

    def clear(self):
        """Clear all measurements."""
        self.records = []
        self.stats = LatencyStats(
            mean_ns=0, std_ns=0, min_ns=float('inf'),
            max_ns=0, sample_count=0
        )


# ============================================================================
# Qubit Emulator for Realistic Testing
# ============================================================================

class QubitEmulator:
    """
    Generates realistic measurement patterns with configurable timing.
    Used for testing ACCL-Q without real quantum hardware.
    """

    def __init__(self, num_qubits: int, t1_us: float = 50, t2_us: float = 30):
        """
        Initialize qubit emulator.

        Args:
            num_qubits: Number of qubits to emulate
            t1_us: T1 relaxation time in microseconds
            t2_us: T2 dephasing time in microseconds
        """
        self.num_qubits = num_qubits
        self.t1 = t1_us * 1e-6  # Convert to seconds
        self.t2 = t2_us * 1e-6

    def generate_measurement(self, state_prep: np.ndarray,
                             readout_time_ns: float) -> np.ndarray:
        """
        Generate measurement outcome based on prepared state and decoherence.

        Args:
            state_prep: Initial qubit states (0 or 1 for each qubit)
            readout_time_ns: Time for readout in nanoseconds

        Returns:
            Measurement outcomes array
        """
        readout_time_s = readout_time_ns * 1e-9

        # Simulate T1 decay
        decay_prob = 1 - np.exp(-readout_time_s / self.t1)

        # Apply decay to excited state qubits
        outcomes = state_prep.copy()
        for i in range(self.num_qubits):
            if outcomes[i] == 1 and np.random.random() < decay_prob:
                outcomes[i] = 0

        return outcomes

    def generate_syndrome(self, error_rate: float = 0.01) -> np.ndarray:
        """
        Generate random error syndrome for QEC testing.

        Args:
            error_rate: Probability of error per qubit

        Returns:
            Syndrome bits array
        """
        errors = np.random.random(self.num_qubits) < error_rate
        # Simple parity syndrome
        syndrome = np.zeros(self.num_qubits // 2, dtype=np.int32)
        for i in range(len(syndrome)):
            syndrome[i] = errors[2*i] ^ errors[2*i + 1]
        return syndrome


# ============================================================================
# ACCL-Q Driver Emulation
# ============================================================================

class ACCLQuantumDriverEmulator:
    """
    Software emulation of ACCL-Q driver for testing.
    """

    def __init__(self, num_ranks: int, local_rank: int,
                 fiber_length_m: float = 10.0):
        """
        Initialize ACCL-Q emulator.

        Args:
            num_ranks: Total number of ranks
            local_rank: This node's rank
            fiber_length_m: Fiber length between nodes
        """
        self.num_ranks = num_ranks
        self.local_rank = local_rank
        self.fiber_length = fiber_length_m
        self.latency_unit = LatencyMeasurementUnit()
        self.op_counter = 0

    def _simulate_latency(self, base_latency: float,
                          jitter_std: float = 2.0) -> float:
        """Add realistic jitter to latency."""
        return base_latency + np.random.normal(0, jitter_std)

    def broadcast(self, data: np.ndarray, root: int,
                  sync_mode: SyncMode = SyncMode.HARDWARE) -> np.ndarray:
        """Emulate broadcast operation with latency measurement."""
        start_time = time.perf_counter_ns()

        # Simulate broadcast latency
        latency = calculate_broadcast_latency(self.num_ranks, self.fiber_length)
        simulated_latency = self._simulate_latency(latency)

        # Simulate the operation time
        time.sleep(simulated_latency * 1e-9)

        end_time = time.perf_counter_ns()

        # Record measurement
        self.latency_unit.measure(
            start_time, start_time + simulated_latency,
            self.op_counter, 'broadcast'
        )
        self.op_counter += 1

        return data  # In emulation, all ranks get the same data

    def reduce(self, data: np.ndarray, op: ReduceOp, root: int,
               sync_mode: SyncMode = SyncMode.HARDWARE) -> np.ndarray:
        """Emulate reduce operation with latency measurement."""
        start_time = time.perf_counter_ns()

        # Simulate reduce latency
        latency = calculate_reduce_latency(self.num_ranks, self.fiber_length)
        simulated_latency = self._simulate_latency(latency)

        # Perform local reduction (emulating distributed behavior)
        if op == ReduceOp.XOR:
            result = np.bitwise_xor.reduce(data)
        elif op == ReduceOp.ADD:
            result = np.sum(data)
        elif op == ReduceOp.MAX:
            result = np.max(data)
        elif op == ReduceOp.MIN:
            result = np.min(data)
        else:
            result = data

        # Record measurement
        self.latency_unit.measure(
            start_time, start_time + simulated_latency,
            self.op_counter, 'reduce'
        )
        self.op_counter += 1

        return result

    def allreduce(self, data: np.ndarray, op: ReduceOp,
                  sync_mode: SyncMode = SyncMode.HARDWARE) -> np.ndarray:
        """Emulate allreduce operation."""
        # Allreduce = reduce + broadcast
        result = self.reduce(data, op, 0, sync_mode)
        return self.broadcast(np.array([result]), 0, sync_mode)

    def allgather(self, data: np.ndarray,
                  sync_mode: SyncMode = SyncMode.HARDWARE) -> np.ndarray:
        """Emulate allgather operation."""
        start_time = time.perf_counter_ns()

        # Allgather has similar latency to allreduce
        latency = calculate_broadcast_latency(self.num_ranks, self.fiber_length)
        simulated_latency = self._simulate_latency(latency * 1.2)  # Slightly more

        # Record measurement
        self.latency_unit.measure(
            start_time, start_time + simulated_latency,
            self.op_counter, 'allgather'
        )
        self.op_counter += 1

        # In real system, would collect from all ranks
        return np.tile(data, self.num_ranks)

    def barrier(self, timeout_ns: int = 10000):
        """Emulate barrier synchronization."""
        start_time = time.perf_counter_ns()

        # Barrier is essentially an allreduce of 1 bit
        latency = calculate_reduce_latency(self.num_ranks, self.fiber_length)
        simulated_latency = self._simulate_latency(latency * 0.5)

        self.latency_unit.measure(
            start_time, start_time + simulated_latency,
            self.op_counter, 'barrier'
        )
        self.op_counter += 1

    def get_latency_stats(self) -> LatencyStats:
        """Return latency statistics."""
        return self.latency_unit.stats


# ============================================================================
# Validation Functions
# ============================================================================

def validate_latency_targets(stats: LatencyStats,
                             targets: List[LatencyTarget]) -> Dict[str, bool]:
    """
    Validate measured latencies against targets.

    Args:
        stats: Measured latency statistics
        targets: List of latency targets to check

    Returns:
        Dictionary of target names to pass/fail status
    """
    results = {}
    for target in targets:
        mean_pass = stats.mean_ns <= target.target_ns
        jitter_pass = stats.std_ns <= target.max_jitter_ns
        results[target.name] = mean_pass and jitter_pass

        print(f"\n{target.name}:")
        print(f"  Target: {target.target_ns} ns, Max jitter: {target.max_jitter_ns} ns")
        print(f"  Measured: mean={stats.mean_ns:.1f} ns, std={stats.std_ns:.1f} ns")
        print(f"  Status: {'PASS' if results[target.name] else 'FAIL'}")

    return results


def run_benchmark(driver: ACCLQuantumDriverEmulator,
                  iterations: int = 1000) -> Dict[str, LatencyStats]:
    """
    Run comprehensive latency benchmark.

    Args:
        driver: ACCL-Q driver emulator
        iterations: Number of iterations per operation

    Returns:
        Dictionary of operation names to statistics
    """
    print(f"\n=== Running Latency Benchmark ({iterations} iterations) ===\n")

    results = {}

    # Test broadcast
    print("Testing broadcast...")
    driver.latency_unit.clear()
    for i in range(iterations):
        data = np.random.randint(0, 2, 64, dtype=np.int32)
        driver.broadcast(data, 0)
    results['broadcast'] = driver.get_latency_stats()

    # Test reduce
    print("Testing reduce...")
    driver.latency_unit.clear()
    for i in range(iterations):
        data = np.random.randint(0, 2, 64, dtype=np.int32)
        driver.reduce(data, ReduceOp.XOR, 0)
    results['reduce'] = driver.get_latency_stats()

    # Test allreduce
    print("Testing allreduce...")
    driver.latency_unit.clear()
    for i in range(iterations):
        data = np.random.randint(0, 2, 64, dtype=np.int32)
        driver.allreduce(data, ReduceOp.XOR)
    results['allreduce'] = driver.get_latency_stats()

    # Test barrier
    print("Testing barrier...")
    driver.latency_unit.clear()
    for i in range(iterations):
        driver.barrier()
    results['barrier'] = driver.get_latency_stats()

    return results


# ============================================================================
# Main Test Execution
# ============================================================================

def main():
    """Main test execution."""
    print("=" * 60)
    print("ACCL-Q Latency Validation Test Suite")
    print("=" * 60)

    # Calculate theoretical latencies
    print("\n--- Theoretical Latency Calculations ---")
    print(f"Point-to-point (10m fiber): {calculate_p2p_latency(10):.1f} ns")
    print(f"Broadcast (8 ranks): {calculate_broadcast_latency(8):.1f} ns")
    print(f"Reduce (8 ranks): {calculate_reduce_latency(8):.1f} ns")

    # Define targets
    targets = [
        LatencyTarget("point-to-point", TARGET_P2P_LATENCY_NS, MAX_JITTER_NS),
        LatencyTarget("broadcast", TARGET_BROADCAST_LATENCY_NS, MAX_JITTER_NS),
        LatencyTarget("reduce", TARGET_REDUCE_LATENCY_NS, MAX_JITTER_NS),
        LatencyTarget("allreduce", TARGET_ALLREDUCE_LATENCY_NS, MAX_JITTER_NS),
    ]

    # Create emulator
    driver = ACCLQuantumDriverEmulator(num_ranks=8, local_rank=0)

    # Run benchmark
    benchmark_results = run_benchmark(driver, iterations=100)

    # Validate against targets
    print("\n--- Validation Results ---")
    for op_name, stats in benchmark_results.items():
        matching_targets = [t for t in targets if t.name == op_name]
        if matching_targets:
            validate_latency_targets(stats, matching_targets)

    # Test with qubit emulator
    print("\n--- Qubit Emulator Integration Test ---")
    emulator = QubitEmulator(num_qubits=8)

    # Generate some measurements and syndromes
    state = np.random.randint(0, 2, 8)
    meas = emulator.generate_measurement(state, readout_time_ns=100)
    syndrome = emulator.generate_syndrome(error_rate=0.05)

    print(f"Initial state: {state}")
    print(f"Measurement result: {meas}")
    print(f"Syndrome: {syndrome}")

    # Test syndrome distribution via allreduce
    syndrome_result = driver.allreduce(syndrome, ReduceOp.XOR)
    print(f"Global syndrome (XOR): {syndrome_result}")

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
