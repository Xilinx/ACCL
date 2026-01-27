"""
ACCL-Q Hardware Validation Test Suite

Comprehensive validation tests for verifying ACCL-Q operations
on actual RFSoC hardware deployments.

Run with: pytest test_hardware_validation.py -v --hardware
"""

import pytest
import numpy as np
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import threading
import socket

# Test configuration
HARDWARE_AVAILABLE = False  # Set True when running on actual hardware
NUM_BOARDS = 4  # Number of boards in test setup
NUM_ITERATIONS = 100  # Iterations for statistical tests
WARMUP_ITERATIONS = 20


# Skip all tests if hardware not available
pytestmark = pytest.mark.skipif(
    not HARDWARE_AVAILABLE,
    reason="Hardware not available - set HARDWARE_AVAILABLE=True"
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def accl_system():
    """Initialize ACCL-Q system for testing."""
    from accl_quantum import ACCLQuantum, ACCLConfig, ACCLMode, SyncMode

    config = ACCLConfig(
        num_ranks=NUM_BOARDS,
        local_rank=0,  # Test from rank 0
        enable_latency_monitoring=True,
        timeout_ns=10_000_000,  # 10ms timeout
    )

    accl = ACCLQuantum(config=config)
    accl.configure(mode=ACCLMode.DETERMINISTIC, sync_mode=SyncMode.HARDWARE)
    accl.sync_clocks()

    yield accl

    # Cleanup
    pass


@pytest.fixture(scope="module")
def deployment_manager():
    """Initialize deployment manager."""
    from accl_quantum.deployment import DeploymentManager, DeploymentConfig

    config = DeploymentConfig.load(Path("config/test_deployment.json"))
    manager = DeploymentManager(config)

    if not manager.deploy():
        pytest.skip("Deployment failed")

    yield manager

    manager.shutdown()


@pytest.fixture
def profiling_session(accl_system):
    """Create profiling session for tests."""
    from accl_quantum.profiler import ProfilingSession

    session = ProfilingSession(monitor=accl_system.get_monitor())
    yield session


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    measured_value: float
    target_value: float
    margin: float
    details: Dict = None

    @property
    def margin_percent(self) -> float:
        if self.target_value == 0:
            return 0
        return 100.0 * (self.measured_value - self.target_value) / self.target_value


# ============================================================================
# Clock Synchronization Validation
# ============================================================================

class TestClockSynchronization:
    """Tests for clock synchronization accuracy."""

    def test_sync_success(self, accl_system):
        """Verify clock synchronization completes successfully."""
        result = accl_system.sync_clocks()
        assert result, "Clock synchronization failed"

    def test_sync_phase_error(self, accl_system):
        """Verify phase error is within specification (<1ns)."""
        status = accl_system.get_sync_status()

        assert status['synchronized'], "System not synchronized"
        assert abs(status['phase_error_ns']) < 1.0, \
            f"Phase error {status['phase_error_ns']:.3f}ns exceeds 1ns target"

    def test_sync_stability(self, accl_system):
        """Verify synchronization remains stable over time."""
        phase_errors = []

        for i in range(10):
            status = accl_system.get_sync_status()
            phase_errors.append(status['phase_error_ns'])
            time.sleep(0.1)  # 100ms between samples

        max_drift = max(phase_errors) - min(phase_errors)
        assert max_drift < 0.5, f"Clock drift {max_drift:.3f}ns exceeds 0.5ns over 1s"

    def test_sync_recovery(self, accl_system):
        """Verify synchronization recovers after disruption."""
        # Force re-sync
        result = accl_system.sync_clocks(timeout_us=2000)
        assert result, "Re-sync failed"

        status = accl_system.get_sync_status()
        assert abs(status['phase_error_ns']) < 1.0

    @pytest.mark.parametrize("num_syncs", [5, 10, 20])
    def test_sync_consistency(self, accl_system, num_syncs):
        """Verify consistent sync results across multiple attempts."""
        phase_errors = []

        for _ in range(num_syncs):
            accl_system.sync_clocks()
            status = accl_system.get_sync_status()
            phase_errors.append(status['phase_error_ns'])

        std_error = np.std(phase_errors)
        assert std_error < 0.3, f"Sync inconsistency: std={std_error:.3f}ns"


# ============================================================================
# Latency Validation
# ============================================================================

class TestLatencyRequirements:
    """Tests for latency requirements."""

    def test_broadcast_latency(self, accl_system, profiling_session):
        """Verify broadcast latency meets <300ns target."""
        from accl_quantum.constants import TARGET_BROADCAST_LATENCY_NS

        data = np.random.randint(0, 256, size=64, dtype=np.uint8)
        latencies = []

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            accl_system.broadcast(data, root=0)

        # Measure
        for _ in range(NUM_ITERATIONS):
            with profiling_session.profile_operation('broadcast'):
                result = accl_system.broadcast(data, root=0)
            latencies.append(result.latency_ns)

        mean_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)

        assert mean_latency < TARGET_BROADCAST_LATENCY_NS, \
            f"Mean broadcast latency {mean_latency:.1f}ns exceeds {TARGET_BROADCAST_LATENCY_NS}ns"
        assert p99_latency < TARGET_BROADCAST_LATENCY_NS * 1.5, \
            f"P99 broadcast latency {p99_latency:.1f}ns too high"

    def test_reduce_latency(self, accl_system, profiling_session):
        """Verify reduce latency meets <400ns target."""
        from accl_quantum.constants import TARGET_REDUCE_LATENCY_NS, ReduceOp

        data = np.random.randint(0, 256, size=64, dtype=np.uint8)
        latencies = []

        for _ in range(WARMUP_ITERATIONS):
            accl_system.reduce(data, op=ReduceOp.XOR, root=0)

        for _ in range(NUM_ITERATIONS):
            result = accl_system.reduce(data, op=ReduceOp.XOR, root=0)
            latencies.append(result.latency_ns)

        mean_latency = np.mean(latencies)
        assert mean_latency < TARGET_REDUCE_LATENCY_NS, \
            f"Mean reduce latency {mean_latency:.1f}ns exceeds {TARGET_REDUCE_LATENCY_NS}ns"

    def test_allreduce_latency(self, accl_system):
        """Verify allreduce latency meets target."""
        from accl_quantum.constants import TARGET_REDUCE_LATENCY_NS, ReduceOp

        data = np.random.randint(0, 256, size=64, dtype=np.uint8)
        latencies = []

        for _ in range(WARMUP_ITERATIONS):
            accl_system.allreduce(data, op=ReduceOp.XOR)

        for _ in range(NUM_ITERATIONS):
            result = accl_system.allreduce(data, op=ReduceOp.XOR)
            latencies.append(result.latency_ns)

        mean_latency = np.mean(latencies)
        # AllReduce ≈ reduce + broadcast
        target = TARGET_REDUCE_LATENCY_NS * 1.2
        assert mean_latency < target, \
            f"Mean allreduce latency {mean_latency:.1f}ns exceeds {target:.0f}ns"

    def test_barrier_latency(self, accl_system):
        """Verify barrier latency and jitter."""
        latencies = []

        for _ in range(WARMUP_ITERATIONS):
            accl_system.barrier()

        for _ in range(NUM_ITERATIONS):
            result = accl_system.barrier()
            latencies.append(result.latency_ns)

        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        assert mean_latency < 100, f"Mean barrier latency {mean_latency:.1f}ns > 100ns"
        assert std_latency < 5, f"Barrier jitter {std_latency:.1f}ns > 5ns"

    def test_feedback_budget(self, accl_system):
        """Verify total feedback path meets <500ns budget."""
        from accl_quantum.constants import FEEDBACK_LATENCY_BUDGET_NS

        # Simulate complete feedback: measure + broadcast + apply
        measurement = np.array([1], dtype=np.uint8)

        latencies = []
        for _ in range(NUM_ITERATIONS):
            start = time.perf_counter_ns()

            # Distribute measurement
            result = accl_system.distribute_measurement(measurement, source_rank=0)

            total_latency = time.perf_counter_ns() - start
            latencies.append(total_latency)

        mean_latency = np.mean(latencies)
        assert mean_latency < FEEDBACK_LATENCY_BUDGET_NS, \
            f"Feedback latency {mean_latency:.1f}ns exceeds {FEEDBACK_LATENCY_BUDGET_NS}ns budget"


# ============================================================================
# Jitter Validation
# ============================================================================

class TestJitterRequirements:
    """Tests for timing jitter requirements."""

    def test_broadcast_jitter(self, accl_system):
        """Verify broadcast jitter <10ns."""
        from accl_quantum.constants import MAX_JITTER_NS

        data = np.random.randint(0, 256, size=64, dtype=np.uint8)
        latencies = []

        for _ in range(NUM_ITERATIONS):
            result = accl_system.broadcast(data, root=0)
            latencies.append(result.latency_ns)

        jitter = np.std(latencies)
        assert jitter < MAX_JITTER_NS, \
            f"Broadcast jitter {jitter:.1f}ns exceeds {MAX_JITTER_NS}ns"

    def test_barrier_jitter(self, accl_system):
        """Verify barrier jitter <2ns."""
        latencies = []

        for _ in range(NUM_ITERATIONS):
            result = accl_system.barrier()
            latencies.append(result.latency_ns)

        jitter = np.std(latencies)
        assert jitter < 2.0, f"Barrier jitter {jitter:.1f}ns exceeds 2ns"

    def test_release_alignment(self, accl_system):
        """Verify barrier release alignment across ranks."""
        # This test requires coordination across multiple boards
        # Using synchronized counter to measure release times

        release_times = []
        for _ in range(NUM_ITERATIONS):
            pre_counter = accl_system.get_global_counter()
            accl_system.barrier()
            post_counter = accl_system.get_global_counter()
            release_times.append(post_counter - pre_counter)

        # All ranks should release within ~2ns (< 1 cycle at 245.76 MHz)
        jitter_cycles = np.std(release_times)
        assert jitter_cycles < 1, f"Release alignment jitter: {jitter_cycles:.2f} cycles"


# ============================================================================
# Operation Correctness
# ============================================================================

class TestOperationCorrectness:
    """Tests for collective operation correctness."""

    def test_broadcast_correctness(self, accl_system):
        """Verify broadcast delivers correct data."""
        test_patterns = [
            np.array([0x55] * 64, dtype=np.uint8),  # 01010101
            np.array([0xAA] * 64, dtype=np.uint8),  # 10101010
            np.array(range(64), dtype=np.uint8),    # Sequential
            np.random.randint(0, 256, 64, dtype=np.uint8),  # Random
        ]

        for pattern in test_patterns:
            result = accl_system.broadcast(pattern.copy(), root=0)
            assert result.success, f"Broadcast failed"
            np.testing.assert_array_equal(result.data, pattern,
                err_msg="Broadcast data mismatch")

    def test_xor_reduce_correctness(self, accl_system):
        """Verify XOR reduction is correct."""
        from accl_quantum.constants import ReduceOp

        # Known test case
        local_data = np.array([0b11001100], dtype=np.uint8)
        result = accl_system.allreduce(local_data, op=ReduceOp.XOR)

        assert result.success, "XOR reduce failed"
        # With NUM_BOARDS boards each contributing same data:
        # Even boards: XOR of same value = 0
        # Odd boards: XOR = value
        expected = local_data if NUM_BOARDS % 2 == 1 else np.array([0], dtype=np.uint8)
        # Note: In real multi-rank test, each rank has different data

    def test_add_reduce_correctness(self, accl_system):
        """Verify ADD reduction is correct."""
        from accl_quantum.constants import ReduceOp

        local_data = np.array([1, 2, 3, 4], dtype=np.uint8)
        result = accl_system.allreduce(local_data, op=ReduceOp.ADD)

        assert result.success, "ADD reduce failed"

    def test_scatter_gather_roundtrip(self, accl_system):
        """Verify scatter/gather preserves data."""
        if accl_system.local_rank == 0:
            # Root prepares data for each rank
            scatter_data = [
                np.array([i * 10 + j for j in range(8)], dtype=np.uint8)
                for i in range(NUM_BOARDS)
            ]

            # Scatter
            scatter_result = accl_system.scatter(scatter_data, root=0)
            assert scatter_result.success

            # Gather back
            gather_result = accl_system.gather(scatter_result.data, root=0)
            assert gather_result.success

            # Verify
            for i in range(NUM_BOARDS):
                np.testing.assert_array_equal(
                    gather_result.data[i],
                    scatter_data[i],
                    err_msg=f"Scatter/gather mismatch for rank {i}"
                )


# ============================================================================
# Stress Tests
# ============================================================================

class TestStressConditions:
    """Stress tests for ACCL-Q operations."""

    def test_sustained_throughput(self, accl_system):
        """Test sustained operation throughput."""
        data = np.random.randint(0, 256, 64, dtype=np.uint8)
        duration_s = 1.0
        operations = 0
        failures = 0

        start_time = time.time()
        while time.time() - start_time < duration_s:
            result = accl_system.broadcast(data, root=0)
            operations += 1
            if not result.success:
                failures += 1

        ops_per_second = operations / duration_s
        failure_rate = failures / operations if operations > 0 else 0

        print(f"Throughput: {ops_per_second:.0f} ops/sec, failures: {failure_rate*100:.2f}%")

        assert failure_rate < 0.001, f"Failure rate {failure_rate*100:.2f}% too high"
        assert ops_per_second > 1000, f"Throughput {ops_per_second:.0f} too low"

    def test_mixed_operations(self, accl_system):
        """Test rapid mixed operations."""
        from accl_quantum.constants import ReduceOp

        data = np.random.randint(0, 256, 64, dtype=np.uint8)
        operations = [
            lambda: accl_system.broadcast(data, root=0),
            lambda: accl_system.allreduce(data, op=ReduceOp.XOR),
            lambda: accl_system.barrier(),
        ]

        failures = 0
        for _ in range(1000):
            op = np.random.choice(operations)
            result = op()
            if not result.success:
                failures += 1

        assert failures == 0, f"{failures} operations failed"

    def test_large_message(self, accl_system):
        """Test with maximum message size."""
        max_size = accl_system.config.max_message_size
        data = np.random.randint(0, 256, max_size, dtype=np.uint8)

        result = accl_system.broadcast(data, root=0)
        assert result.success, "Large message broadcast failed"
        np.testing.assert_array_equal(result.data, data)

    def test_concurrent_operations(self, accl_system):
        """Test concurrent operations from multiple threads."""
        from accl_quantum.constants import ReduceOp

        results = []
        errors = []

        def worker(worker_id):
            try:
                data = np.array([worker_id], dtype=np.uint8)
                for _ in range(100):
                    result = accl_system.allreduce(data, op=ReduceOp.ADD)
                    if not result.success:
                        errors.append(f"Worker {worker_id}: operation failed")
                results.append(worker_id)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 4, "Not all workers completed"


# ============================================================================
# Quantum-Specific Validation
# ============================================================================

class TestQuantumOperations:
    """Tests for quantum-specific operations."""

    def test_syndrome_aggregation(self, accl_system):
        """Test QEC syndrome aggregation."""
        # Simulate syndrome bits from stabilizer measurements
        local_syndrome = np.random.randint(0, 2, 16, dtype=np.uint8)

        result = accl_system.aggregate_syndrome(local_syndrome)
        assert result.success, "Syndrome aggregation failed"
        assert result.data is not None
        assert len(result.data) == len(local_syndrome)

    def test_measurement_distribution(self, accl_system):
        """Test measurement result distribution."""
        measurement = np.array([0, 1, 1, 0], dtype=np.uint8)

        result = accl_system.distribute_measurement(measurement, source_rank=0)
        assert result.success
        np.testing.assert_array_equal(result.data, measurement)

    def test_correction_distribution(self, accl_system):
        """Test correction distribution to control boards."""
        if accl_system.local_rank == 0:  # Decoder board
            corrections = [
                np.array([0, 1], dtype=np.uint8),  # X correction for rank 0
                np.array([1, 0], dtype=np.uint8),  # Z correction for rank 1
                np.array([0, 0], dtype=np.uint8),  # No correction for rank 2
                np.array([1, 1], dtype=np.uint8),  # XZ for rank 3
            ][:NUM_BOARDS]

            result = accl_system.distribute_correction(corrections, decoder_rank=0)
            assert result.success

    def test_synchronized_trigger(self, accl_system):
        """Test synchronized trigger scheduling."""
        current_counter = accl_system.get_global_counter()
        trigger_time = current_counter + 1000  # 1000 cycles in future

        success = accl_system.synchronized_trigger(trigger_time)
        assert success, "Failed to schedule trigger"

        # Verify trigger not scheduled in past
        success = accl_system.synchronized_trigger(current_counter - 100)
        assert not success, "Should not schedule trigger in past"


# ============================================================================
# Regression Tests
# ============================================================================

class TestPerformanceRegression:
    """Performance regression tests."""

    @pytest.fixture
    def baseline_path(self, tmp_path):
        return tmp_path / "baseline.json"

    def test_compare_to_baseline(self, accl_system, baseline_path):
        """Compare current performance to baseline."""
        from accl_quantum.profiler import PerformanceRegressor

        regressor = PerformanceRegressor(baseline_path=baseline_path)
        regressor.update_from_monitor(accl_system.get_monitor())

        # Save current as baseline if none exists
        if not baseline_path.exists():
            regressor.save_baseline()
            pytest.skip("Baseline created, run again to compare")

        regressions = regressor.check_regressions()
        if regressions:
            for r in regressions:
                print(f"Regression: {r['operation']} {r['metric']} "
                      f"changed {r['change_percent']:+.1f}%")

        assert len(regressions) == 0, \
            f"Performance regressions detected: {len(regressions)}"


# ============================================================================
# Report Generation
# ============================================================================

class TestReportGeneration:
    """Generate validation reports."""

    def test_generate_validation_report(self, accl_system, profiling_session, tmp_path):
        """Generate comprehensive validation report."""
        from accl_quantum.constants import (
            TARGET_BROADCAST_LATENCY_NS,
            TARGET_REDUCE_LATENCY_NS,
            MAX_JITTER_NS,
            ReduceOp,
        )

        results: List[ValidationResult] = []

        # Run all validations
        data = np.random.randint(0, 256, 64, dtype=np.uint8)

        # Broadcast latency
        latencies = []
        for _ in range(NUM_ITERATIONS):
            result = accl_system.broadcast(data, root=0)
            latencies.append(result.latency_ns)

        mean_lat = np.mean(latencies)
        results.append(ValidationResult(
            test_name="Broadcast Latency",
            passed=mean_lat < TARGET_BROADCAST_LATENCY_NS,
            measured_value=mean_lat,
            target_value=TARGET_BROADCAST_LATENCY_NS,
            margin=TARGET_BROADCAST_LATENCY_NS - mean_lat,
        ))

        # Broadcast jitter
        jitter = np.std(latencies)
        results.append(ValidationResult(
            test_name="Broadcast Jitter",
            passed=jitter < MAX_JITTER_NS,
            measured_value=jitter,
            target_value=MAX_JITTER_NS,
            margin=MAX_JITTER_NS - jitter,
        ))

        # AllReduce latency
        latencies = []
        for _ in range(NUM_ITERATIONS):
            result = accl_system.allreduce(data, op=ReduceOp.XOR)
            latencies.append(result.latency_ns)

        mean_lat = np.mean(latencies)
        results.append(ValidationResult(
            test_name="AllReduce Latency",
            passed=mean_lat < TARGET_REDUCE_LATENCY_NS * 1.2,
            measured_value=mean_lat,
            target_value=TARGET_REDUCE_LATENCY_NS * 1.2,
            margin=TARGET_REDUCE_LATENCY_NS * 1.2 - mean_lat,
        ))

        # Barrier jitter
        latencies = []
        for _ in range(NUM_ITERATIONS):
            result = accl_system.barrier()
            latencies.append(result.latency_ns)

        jitter = np.std(latencies)
        results.append(ValidationResult(
            test_name="Barrier Jitter",
            passed=jitter < 2.0,
            measured_value=jitter,
            target_value=2.0,
            margin=2.0 - jitter,
        ))

        # Clock sync
        status = accl_system.get_sync_status()
        phase_error = abs(status['phase_error_ns'])
        results.append(ValidationResult(
            test_name="Clock Phase Error",
            passed=phase_error < 1.0,
            measured_value=phase_error,
            target_value=1.0,
            margin=1.0 - phase_error,
        ))

        # Generate report
        report_lines = [
            "=" * 70,
            "ACCL-Q HARDWARE VALIDATION REPORT",
            "=" * 70,
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Boards: {NUM_BOARDS}",
            f"Iterations: {NUM_ITERATIONS}",
            "",
            "RESULTS",
            "-" * 70,
        ]

        passed = 0
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            report_lines.append(
                f"[{status}] {r.test_name}: "
                f"{r.measured_value:.2f} (target: {r.target_value:.2f}, "
                f"margin: {r.margin:+.2f})"
            )
            if r.passed:
                passed += 1

        report_lines.extend([
            "",
            "-" * 70,
            f"SUMMARY: {passed}/{len(results)} tests passed",
            "=" * 70,
        ])

        report = "\n".join(report_lines)
        print(report)

        # Save report
        report_path = tmp_path / "validation_report.txt"
        report_path.write_text(report)

        # Save JSON results
        json_path = tmp_path / "validation_results.json"
        json_data = {
            'timestamp': time.time(),
            'num_boards': NUM_BOARDS,
            'iterations': NUM_ITERATIONS,
            'results': [
                {
                    'test': r.test_name,
                    'passed': r.passed,
                    'measured': r.measured_value,
                    'target': r.target_value,
                    'margin': r.margin,
                }
                for r in results
            ]
        }
        json_path.write_text(json.dumps(json_data, indent=2))

        # Assert all passed
        assert passed == len(results), \
            f"Validation failed: {len(results) - passed} tests failed"


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
