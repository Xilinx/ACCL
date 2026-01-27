"""
ACCL-Q Latency Statistics and Monitoring

Provides real-time latency tracking and statistical analysis for
validating quantum timing requirements.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
import threading

from .constants import (
    CollectiveOp,
    TARGET_P2P_LATENCY_NS,
    TARGET_BROADCAST_LATENCY_NS,
    TARGET_REDUCE_LATENCY_NS,
    MAX_JITTER_NS,
)


@dataclass
class LatencyStats:
    """Statistics for a set of latency measurements."""
    count: int = 0
    mean_ns: float = 0.0
    std_ns: float = 0.0
    min_ns: float = float('inf')
    max_ns: float = 0.0
    p50_ns: float = 0.0
    p95_ns: float = 0.0
    p99_ns: float = 0.0

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyStats":
        """Compute statistics from a list of samples."""
        if not samples:
            return cls()

        arr = np.array(samples)
        return cls(
            count=len(samples),
            mean_ns=float(np.mean(arr)),
            std_ns=float(np.std(arr)),
            min_ns=float(np.min(arr)),
            max_ns=float(np.max(arr)),
            p50_ns=float(np.percentile(arr, 50)),
            p95_ns=float(np.percentile(arr, 95)),
            p99_ns=float(np.percentile(arr, 99)),
        )

    def meets_target(self, target_ns: float, jitter_target_ns: float) -> bool:
        """Check if stats meet latency and jitter targets."""
        return self.mean_ns <= target_ns and self.std_ns <= jitter_target_ns

    def __str__(self) -> str:
        return (
            f"LatencyStats(n={self.count}, mean={self.mean_ns:.1f}ns, "
            f"std={self.std_ns:.1f}ns, min={self.min_ns:.1f}ns, "
            f"max={self.max_ns:.1f}ns, p99={self.p99_ns:.1f}ns)"
        )


@dataclass
class LatencyRecord:
    """Single latency measurement record."""
    timestamp_ns: int
    operation: CollectiveOp
    latency_ns: float
    num_ranks: int
    root_rank: Optional[int] = None
    success: bool = True
    metadata: Dict = field(default_factory=dict)


class LatencyMonitor:
    """
    Real-time latency monitoring for ACCL-Q operations.

    Features:
    - Per-operation latency tracking
    - Rolling window statistics
    - Target violation alerts
    - Histogram generation for jitter analysis
    """

    def __init__(self, window_size: int = 1000,
                 enable_alerts: bool = True):
        """
        Initialize latency monitor.

        Args:
            window_size: Number of samples to keep in rolling window
            enable_alerts: Enable alert callbacks on target violations
        """
        self.window_size = window_size
        self.enable_alerts = enable_alerts

        # Per-operation sample buffers
        self._samples: Dict[CollectiveOp, deque] = {
            op: deque(maxlen=window_size) for op in CollectiveOp
        }

        # Full history (for offline analysis)
        self._history: List[LatencyRecord] = []
        self._history_lock = threading.Lock()

        # Alert callbacks
        self._alert_callbacks: List[callable] = []

        # Latency targets per operation
        self._targets: Dict[CollectiveOp, float] = {
            CollectiveOp.BROADCAST: TARGET_BROADCAST_LATENCY_NS,
            CollectiveOp.REDUCE: TARGET_REDUCE_LATENCY_NS,
            CollectiveOp.ALLREDUCE: TARGET_REDUCE_LATENCY_NS,
            CollectiveOp.SCATTER: TARGET_P2P_LATENCY_NS,
            CollectiveOp.GATHER: TARGET_P2P_LATENCY_NS,
            CollectiveOp.ALLGATHER: TARGET_BROADCAST_LATENCY_NS,
            CollectiveOp.BARRIER: 100,  # Barrier jitter target
        }

        # Violation counters
        self._violations: Dict[CollectiveOp, int] = {op: 0 for op in CollectiveOp}

    def record(self, operation: CollectiveOp, latency_ns: float,
               num_ranks: int, root_rank: Optional[int] = None,
               success: bool = True, **metadata) -> None:
        """
        Record a latency measurement.

        Args:
            operation: Type of collective operation
            latency_ns: Measured latency in nanoseconds
            num_ranks: Number of ranks involved
            root_rank: Root rank (for rooted operations)
            success: Whether operation completed successfully
            **metadata: Additional metadata to store
        """
        record = LatencyRecord(
            timestamp_ns=time.time_ns(),
            operation=operation,
            latency_ns=latency_ns,
            num_ranks=num_ranks,
            root_rank=root_rank,
            success=success,
            metadata=metadata
        )

        # Add to rolling window
        self._samples[operation].append(latency_ns)

        # Add to history
        with self._history_lock:
            self._history.append(record)

        # Check for target violation
        target = self._targets.get(operation, float('inf'))
        if latency_ns > target:
            self._violations[operation] += 1
            if self.enable_alerts:
                self._trigger_alert(operation, latency_ns, target)

    def get_stats(self, operation: Optional[CollectiveOp] = None) -> Dict[CollectiveOp, LatencyStats]:
        """
        Get latency statistics for operations.

        Args:
            operation: Specific operation, or None for all

        Returns:
            Dictionary mapping operations to their statistics
        """
        if operation is not None:
            samples = list(self._samples[operation])
            return {operation: LatencyStats.from_samples(samples)}

        return {
            op: LatencyStats.from_samples(list(samples))
            for op, samples in self._samples.items()
            if len(samples) > 0
        }

    def get_histogram(self, operation: CollectiveOp,
                      bin_width_ns: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate histogram of latency distribution.

        Args:
            operation: Operation to analyze
            bin_width_ns: Width of histogram bins

        Returns:
            Tuple of (counts, bin_edges)
        """
        samples = list(self._samples[operation])
        if not samples:
            return np.array([]), np.array([])

        max_val = max(samples)
        bins = np.arange(0, max_val + bin_width_ns, bin_width_ns)
        counts, edges = np.histogram(samples, bins=bins)
        return counts, edges

    def get_violations(self) -> Dict[CollectiveOp, int]:
        """Get count of target violations per operation."""
        return self._violations.copy()

    def get_violation_rate(self, operation: CollectiveOp) -> float:
        """Get violation rate for an operation."""
        total = len(self._samples[operation])
        if total == 0:
            return 0.0
        return self._violations[operation] / total

    def add_alert_callback(self, callback: callable) -> None:
        """
        Add callback for target violation alerts.

        Callback signature: callback(operation, latency_ns, target_ns)
        """
        self._alert_callbacks.append(callback)

    def _trigger_alert(self, operation: CollectiveOp,
                       latency_ns: float, target_ns: float) -> None:
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(operation, latency_ns, target_ns)
            except Exception as e:
                print(f"Alert callback error: {e}")

    def clear(self) -> None:
        """Clear all recorded data."""
        for samples in self._samples.values():
            samples.clear()
        with self._history_lock:
            self._history.clear()
        self._violations = {op: 0 for op in CollectiveOp}

    def export_history(self) -> List[Dict]:
        """Export full history as list of dictionaries."""
        with self._history_lock:
            return [
                {
                    'timestamp_ns': r.timestamp_ns,
                    'operation': r.operation.name,
                    'latency_ns': r.latency_ns,
                    'num_ranks': r.num_ranks,
                    'root_rank': r.root_rank,
                    'success': r.success,
                    **r.metadata
                }
                for r in self._history
            ]

    def summary(self) -> str:
        """Generate summary report."""
        lines = ["ACCL-Q Latency Monitor Summary", "=" * 40]

        stats = self.get_stats()
        for op, s in stats.items():
            target = self._targets.get(op, 0)
            status = "✓" if s.meets_target(target, MAX_JITTER_NS) else "✗"
            lines.append(f"\n{op.name}:")
            lines.append(f"  {s}")
            lines.append(f"  Target: {target}ns, Status: {status}")
            lines.append(f"  Violations: {self._violations[op]}")

        return "\n".join(lines)


class LatencyProfiler:
    """
    Context manager for profiling operation latency.

    Usage:
        monitor = LatencyMonitor()
        with LatencyProfiler(monitor, CollectiveOp.BROADCAST, num_ranks=8):
            result = accl.broadcast(data, root=0)
    """

    def __init__(self, monitor: LatencyMonitor, operation: CollectiveOp,
                 num_ranks: int, root_rank: Optional[int] = None, **metadata):
        self.monitor = monitor
        self.operation = operation
        self.num_ranks = num_ranks
        self.root_rank = root_rank
        self.metadata = metadata
        self._start_ns = 0

    def __enter__(self):
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_ns = time.perf_counter_ns()
        latency_ns = end_ns - self._start_ns
        success = exc_type is None

        self.monitor.record(
            self.operation,
            latency_ns,
            self.num_ranks,
            self.root_rank,
            success,
            **self.metadata
        )
        return False  # Don't suppress exceptions
