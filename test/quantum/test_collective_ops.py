#!/usr/bin/env python3
"""
ACCL-Q Collective Operations Test Suite

Comprehensive validation of quantum-optimized collective operations:
- Broadcast (tree-based, deterministic timing)
- Reduce (XOR, ADD, MAX, MIN)
- Allreduce
- Barrier (hardware-synchronized)
- Scatter/Gather
- Allgather

Tests verify both correctness and latency targets.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple, Optional
from enum import Enum
import time
from abc import ABC, abstractmethod
import pytest

# ============================================================================
# Constants
# ============================================================================

CLOCK_PERIOD_NS = 2  # 500 MHz
MAX_RANKS = 16
MAX_TREE_FANOUT = 4

# Latency targets (nanoseconds)
TARGET_P2P_LATENCY_NS = 200
TARGET_BROADCAST_LATENCY_NS = 300
TARGET_REDUCE_LATENCY_NS = 400
TARGET_BARRIER_JITTER_NS = 100


class ReduceOp(Enum):
    XOR = 0
    ADD = 1
    MAX = 2
    MIN = 3


class CollectiveOp(Enum):
    BROADCAST = 0
    REDUCE = 1
    ALLREDUCE = 2
    BARRIER = 3
    SCATTER = 4
    GATHER = 5
    ALLGATHER = 6


# ============================================================================
# Tree Topology
# ============================================================================

@dataclass
class TreeTopology:
    """Represents a node's position in a tree topology."""
    rank: int
    total_ranks: int
    root_rank: int
    fanout: int = MAX_TREE_FANOUT

    @property
    def logical_rank(self) -> int:
        """Rank rebased so root is 0."""
        if self.rank >= self.root_rank:
            return self.rank - self.root_rank
        return self.rank + self.total_ranks - self.root_rank

    @property
    def is_root(self) -> bool:
        return self.rank == self.root_rank

    @property
    def parent_rank(self) -> Optional[int]:
        if self.is_root:
            return None
        logical_parent = (self.logical_rank - 1) // self.fanout
        return (logical_parent + self.root_rank) % self.total_ranks

    @property
    def children_ranks(self) -> List[int]:
        children = []
        first_child = self.logical_rank * self.fanout + 1
        for i in range(self.fanout):
            child_logical = first_child + i
            if child_logical < self.total_ranks:
                child_rank = (child_logical + self.root_rank) % self.total_ranks
                children.append(child_rank)
        return children

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ranks) == 0

    @property
    def depth(self) -> int:
        """Depth from root (root = 0)."""
        depth = 0
        lr = self.logical_rank
        while lr > 0:
            lr = (lr - 1) // self.fanout
            depth += 1
        return depth


def compute_tree_depth(num_ranks: int, fanout: int = MAX_TREE_FANOUT) -> int:
    """Compute depth of tree for given number of ranks."""
    depth = 0
    n = num_ranks
    while n > 1:
        n = (n + fanout - 1) // fanout
        depth += 1
    return depth


# ============================================================================
# Collective Operation Implementations
# ============================================================================

def reduce_operation(values: List[np.ndarray], op: ReduceOp) -> np.ndarray:
    """Apply reduction operation to list of values."""
    if len(values) == 0:
        return np.array([0], dtype=np.uint64)

    result = values[0].copy()
    for v in values[1:]:
        if op == ReduceOp.XOR:
            result = np.bitwise_xor(result, v)
        elif op == ReduceOp.ADD:
            result = result + v
        elif op == ReduceOp.MAX:
            result = np.maximum(result, v)
        elif op == ReduceOp.MIN:
            result = np.minimum(result, v)
    return result


class CollectiveSimulator:
    """
    Simulates collective operations with timing.
    """

    def __init__(self, num_ranks: int, p2p_latency_ns: float = 100.0):
        self.num_ranks = num_ranks
        self.p2p_latency_ns = p2p_latency_ns
        self.latency_records: List[Dict] = []

    def _record_latency(self, op: CollectiveOp, latency_ns: float,
                        details: Dict = None):
        record = {
            'operation': op.name,
            'latency_ns': latency_ns,
            'ranks': self.num_ranks,
            'details': details or {}
        }
        self.latency_records.append(record)
        return latency_ns

    def broadcast(self, data: np.ndarray, root: int) -> Tuple[List[np.ndarray], float]:
        """
        Simulate tree broadcast.

        Returns:
            Tuple of (results for each rank, total latency in ns)
        """
        tree_depth = compute_tree_depth(self.num_ranks)
        latency = tree_depth * self.p2p_latency_ns

        # All ranks receive the same data
        results = [data.copy() for _ in range(self.num_ranks)]

        self._record_latency(CollectiveOp.BROADCAST, latency,
                            {'root': root, 'tree_depth': tree_depth})
        return results, latency

    def reduce(self, local_data: List[np.ndarray], op: ReduceOp,
               root: int) -> Tuple[np.ndarray, float]:
        """
        Simulate tree reduce.

        Args:
            local_data: Data from each rank
            op: Reduction operation
            root: Root rank to receive result

        Returns:
            Tuple of (reduced result, total latency in ns)
        """
        tree_depth = compute_tree_depth(self.num_ranks)
        # Each level adds latency + small compute time
        compute_time_per_level = 5  # ns
        latency = tree_depth * (self.p2p_latency_ns + compute_time_per_level)

        result = reduce_operation(local_data, op)

        self._record_latency(CollectiveOp.REDUCE, latency,
                            {'root': root, 'op': op.name, 'tree_depth': tree_depth})
        return result, latency

    def allreduce(self, local_data: List[np.ndarray],
                  op: ReduceOp) -> Tuple[List[np.ndarray], float]:
        """
        Simulate allreduce (reduce + broadcast).

        Returns:
            Tuple of (results for each rank, total latency in ns)
        """
        # Reduce to root
        reduced, reduce_latency = self.reduce(local_data, op, 0)

        # Broadcast result
        results, bcast_latency = self.broadcast(reduced, 0)

        total_latency = reduce_latency + bcast_latency

        self._record_latency(CollectiveOp.ALLREDUCE, total_latency,
                            {'op': op.name})
        return results, total_latency

    def barrier(self, arrival_times: List[float]) -> Tuple[float, float]:
        """
        Simulate hardware-synchronized barrier.

        Args:
            arrival_times: When each rank arrives at barrier

        Returns:
            Tuple of (release time, jitter in ns)
        """
        max_arrival = max(arrival_times)
        margin = 50  # ns

        release_time = max_arrival + margin

        # Jitter should be minimal with hardware sync
        # Simulate small jitter from clock sync imperfection
        jitter = np.random.uniform(0, 2)  # 0-2 ns

        self._record_latency(CollectiveOp.BARRIER, margin + jitter,
                            {'max_wait': max_arrival - min(arrival_times)})
        return release_time, jitter

    def scatter(self, data_per_rank: List[np.ndarray],
                root: int) -> Tuple[List[np.ndarray], float]:
        """
        Simulate scatter from root.

        Returns:
            Tuple of (data received by each rank, latency in ns)
        """
        # Single hop from root to all (parallel)
        latency = self.p2p_latency_ns

        results = [data_per_rank[r].copy() for r in range(self.num_ranks)]

        self._record_latency(CollectiveOp.SCATTER, latency, {'root': root})
        return results, latency

    def gather(self, local_data: List[np.ndarray],
               root: int) -> Tuple[List[np.ndarray], float]:
        """
        Simulate gather to root.

        Returns:
            Tuple of (gathered data at root, latency in ns)
        """
        # Single hop from all to root (parallel receives)
        latency = self.p2p_latency_ns

        gathered = [d.copy() for d in local_data]

        self._record_latency(CollectiveOp.GATHER, latency, {'root': root})
        return gathered, latency

    def allgather(self, local_data: List[np.ndarray]) -> Tuple[List[List[np.ndarray]], float]:
        """
        Simulate allgather (gather + broadcast).

        Returns:
            Tuple of (all data at each rank, latency in ns)
        """
        # Gather to root
        gathered, gather_latency = self.gather(local_data, 0)

        # Broadcast full array (simplified - would be multiple broadcasts)
        # In practice, use ring or recursive doubling for efficiency
        bcast_latency = self.p2p_latency_ns * compute_tree_depth(self.num_ranks)

        total_latency = gather_latency + bcast_latency

        # All ranks have all data
        results = [gathered.copy() for _ in range(self.num_ranks)]

        self._record_latency(CollectiveOp.ALLGATHER, total_latency)
        return results, total_latency

    def get_statistics(self) -> Dict[str, Dict]:
        """Compute statistics for each operation type."""
        stats = {}
        for op in CollectiveOp:
            records = [r for r in self.latency_records if r['operation'] == op.name]
            if records:
                latencies = [r['latency_ns'] for r in records]
                stats[op.name] = {
                    'count': len(records),
                    'mean_ns': np.mean(latencies),
                    'std_ns': np.std(latencies),
                    'min_ns': np.min(latencies),
                    'max_ns': np.max(latencies)
                }
        return stats


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def sim():
    """Create CollectiveSimulator fixture for tests."""
    return CollectiveSimulator(num_ranks=8, p2p_latency_ns=100)


@pytest.fixture
def iterations():
    """Default iteration count for tests."""
    return 100


@pytest.fixture
def op():
    """Default reduce operation for tests."""
    return ReduceOp.XOR


# ============================================================================
# Test Functions
# ============================================================================

def test_broadcast(sim: CollectiveSimulator, iterations: int = 100) -> Dict:
    """Test broadcast operation."""
    print("\nTesting Broadcast...")

    passed = 0
    failed = 0

    for i in range(iterations):
        root = np.random.randint(0, sim.num_ranks)
        data = np.random.randint(0, 2**32, size=8, dtype=np.uint64)

        results, latency = sim.broadcast(data, root)

        # Verify all ranks have correct data
        correct = all(np.array_equal(r, data) for r in results)

        if correct and latency <= TARGET_BROADCAST_LATENCY_NS:
            passed += 1
        else:
            failed += 1
            if failed <= 5:  # Print first few failures
                print(f"  FAIL iter {i}: correct={correct}, latency={latency}ns")

    print(f"  Result: {passed}/{iterations} passed")
    return {'passed': passed, 'failed': failed}


def test_reduce(sim: CollectiveSimulator, op: ReduceOp,
                iterations: int = 100) -> Dict:
    """Test reduce operation."""
    print(f"\nTesting Reduce ({op.name})...")

    passed = 0
    failed = 0

    for i in range(iterations):
        root = np.random.randint(0, sim.num_ranks)

        # Generate local data for each rank
        if op == ReduceOp.ADD:
            local_data = [np.random.randint(0, 1000, size=4, dtype=np.uint64)
                         for _ in range(sim.num_ranks)]
        else:
            local_data = [np.random.randint(0, 2**16, size=4, dtype=np.uint64)
                         for _ in range(sim.num_ranks)]

        result, latency = sim.reduce(local_data, op, root)

        # Verify result
        expected = reduce_operation(local_data, op)
        correct = np.array_equal(result, expected)

        if correct and latency <= TARGET_REDUCE_LATENCY_NS:
            passed += 1
        else:
            failed += 1

    print(f"  Result: {passed}/{iterations} passed")
    return {'passed': passed, 'failed': failed}


def test_barrier(sim: CollectiveSimulator, iterations: int = 100) -> Dict:
    """Test barrier operation."""
    print("\nTesting Barrier...")

    passed = 0
    failed = 0
    max_jitter = 0

    for i in range(iterations):
        # Simulate staggered arrivals
        base_time = 1000  # ns
        arrivals = [base_time + np.random.uniform(0, 50)
                   for _ in range(sim.num_ranks)]

        release_time, jitter = sim.barrier(arrivals)

        max_jitter = max(max_jitter, jitter)

        # Verify all ranks wait for release
        correct = all(release_time >= t for t in arrivals)

        if correct and jitter <= TARGET_BARRIER_JITTER_NS:
            passed += 1
        else:
            failed += 1

    print(f"  Result: {passed}/{iterations} passed, max_jitter={max_jitter:.1f}ns")
    return {'passed': passed, 'failed': failed, 'max_jitter': max_jitter}


def test_scatter_gather(sim: CollectiveSimulator, iterations: int = 100) -> Dict:
    """Test scatter and gather operations."""
    print("\nTesting Scatter/Gather...")

    passed = 0
    failed = 0

    for i in range(iterations):
        root = np.random.randint(0, sim.num_ranks)

        # Scatter: root sends different data to each rank
        scatter_data = [np.array([r * 100 + i], dtype=np.uint64)
                       for r in range(sim.num_ranks)]
        scatter_results, scatter_latency = sim.scatter(scatter_data, root)

        # Gather: collect data at root
        gather_results, gather_latency = sim.gather(scatter_results, root)

        # Verify round-trip
        correct = all(np.array_equal(scatter_data[r], gather_results[r])
                     for r in range(sim.num_ranks))

        if correct:
            passed += 1
        else:
            failed += 1

    print(f"  Result: {passed}/{iterations} passed")
    return {'passed': passed, 'failed': failed}


def test_allgather(sim: CollectiveSimulator, iterations: int = 100) -> Dict:
    """Test allgather operation."""
    print("\nTesting Allgather...")

    passed = 0
    failed = 0

    for i in range(iterations):
        local_data = [np.array([r], dtype=np.uint64)
                     for r in range(sim.num_ranks)]

        results, latency = sim.allgather(local_data)

        # Verify all ranks have all data
        correct = True
        for rank_results in results:
            for r, expected in enumerate(local_data):
                if not np.array_equal(rank_results[r], expected):
                    correct = False
                    break

        if correct:
            passed += 1
        else:
            failed += 1

    print(f"  Result: {passed}/{iterations} passed")
    return {'passed': passed, 'failed': failed}


# ============================================================================
# Quantum-Specific Tests
# ============================================================================

def test_syndrome_aggregation(sim: CollectiveSimulator,
                              num_qubits: int = 16,
                              iterations: int = 100) -> Dict:
    """
    Test XOR-based syndrome aggregation for QEC.

    In quantum error correction, local syndromes are XORed together
    to compute a global syndrome for decoding.
    """
    print(f"\nTesting QEC Syndrome Aggregation ({num_qubits} qubits)...")

    passed = 0
    failed = 0

    for i in range(iterations):
        # Generate random local syndromes (simulating measurement errors)
        error_rate = 0.01
        local_syndromes = []
        for r in range(sim.num_ranks):
            syndrome = np.zeros(num_qubits // sim.num_ranks, dtype=np.uint64)
            for q in range(len(syndrome)):
                if np.random.random() < error_rate:
                    syndrome[q] = 1
            local_syndromes.append(syndrome)

        # Compute global syndrome via allreduce XOR
        results, latency = sim.allreduce(local_syndromes, ReduceOp.XOR)

        # Verify all ranks have same global syndrome
        correct = all(np.array_equal(results[0], r) for r in results)

        # Verify latency is within budget for QEC
        # Typically need < 500ns for real-time decoding
        within_budget = latency <= 500

        if correct and within_budget:
            passed += 1
        else:
            failed += 1

    print(f"  Result: {passed}/{iterations} passed")
    return {'passed': passed, 'failed': failed}


def test_measurement_distribution(sim: CollectiveSimulator,
                                   iterations: int = 100) -> Dict:
    """
    Test measurement result distribution for conditional operations.

    When one qubit's measurement determines operations on other qubits,
    the result must be distributed to all control boards quickly.
    """
    print("\nTesting Measurement Distribution...")

    passed = 0
    failed = 0

    for i in range(iterations):
        # One rank has the measurement result
        source_rank = np.random.randint(0, sim.num_ranks)
        measurement = np.array([np.random.randint(0, 2)], dtype=np.uint64)

        # Broadcast measurement to all ranks
        results, latency = sim.broadcast(measurement, source_rank)

        # Verify all ranks have the measurement
        correct = all(np.array_equal(r, measurement) for r in results)

        # Must complete within coherence time budget
        # Assuming 500ns budget for feedback
        within_budget = latency <= 300

        if correct and within_budget:
            passed += 1
        else:
            failed += 1

    print(f"  Result: {passed}/{iterations} passed")
    return {'passed': passed, 'failed': failed}


# ============================================================================
# Main Test Entry
# ============================================================================

def main():
    print("=" * 60)
    print("ACCL-Q Collective Operations Test Suite")
    print("=" * 60)

    # Configuration
    num_ranks = 8
    iterations = 100

    print(f"\nConfiguration:")
    print(f"  Ranks: {num_ranks}")
    print(f"  Iterations: {iterations}")
    print(f"  Tree fanout: {MAX_TREE_FANOUT}")
    print(f"  Tree depth: {compute_tree_depth(num_ranks)}")

    # Create simulator
    sim = CollectiveSimulator(num_ranks, p2p_latency_ns=100)

    # Run basic collective tests
    results = {}
    results['broadcast'] = test_broadcast(sim, iterations)
    results['reduce_xor'] = test_reduce(sim, ReduceOp.XOR, iterations)
    results['reduce_add'] = test_reduce(sim, ReduceOp.ADD, iterations)
    results['reduce_max'] = test_reduce(sim, ReduceOp.MAX, iterations)
    results['barrier'] = test_barrier(sim, iterations)
    results['scatter_gather'] = test_scatter_gather(sim, iterations)
    results['allgather'] = test_allgather(sim, iterations)

    # Run quantum-specific tests
    results['syndrome'] = test_syndrome_aggregation(sim, iterations=iterations)
    results['measurement_dist'] = test_measurement_distribution(sim, iterations)

    # Print latency statistics
    print("\n" + "=" * 60)
    print("Latency Statistics")
    print("=" * 60)

    stats = sim.get_statistics()
    for op_name, op_stats in stats.items():
        print(f"\n{op_name}:")
        print(f"  Count: {op_stats['count']}")
        print(f"  Latency: mean={op_stats['mean_ns']:.1f}ns, "
              f"std={op_stats['std_ns']:.1f}ns, "
              f"min={op_stats['min_ns']:.1f}ns, "
              f"max={op_stats['max_ns']:.1f}ns")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    total_passed = sum(r.get('passed', 0) for r in results.values())
    total_failed = sum(r.get('failed', 0) for r in results.values())

    print(f"\nTotal: {total_passed} passed, {total_failed} failed")

    # Target validation
    print("\nLatency Target Validation:")
    print(f"  Broadcast: {'PASS' if stats.get('BROADCAST', {}).get('max_ns', 999) <= TARGET_BROADCAST_LATENCY_NS else 'FAIL'}")
    print(f"  Reduce: {'PASS' if stats.get('REDUCE', {}).get('max_ns', 999) <= TARGET_REDUCE_LATENCY_NS else 'FAIL'}")
    print(f"  Barrier jitter: {'PASS' if results['barrier'].get('max_jitter', 999) <= TARGET_BARRIER_JITTER_NS else 'FAIL'}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())
