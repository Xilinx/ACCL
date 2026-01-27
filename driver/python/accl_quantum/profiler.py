"""
ACCL-Q Profiling and Optimization Tools

Provides comprehensive profiling, bottleneck analysis, and optimization
recommendations for quantum control operations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import defaultdict
import time
import json
import threading
from pathlib import Path

from .constants import (
    CollectiveOp,
    TARGET_P2P_LATENCY_NS,
    TARGET_BROADCAST_LATENCY_NS,
    TARGET_REDUCE_LATENCY_NS,
    TARGET_SCATTER_LATENCY_NS,
    FEEDBACK_LATENCY_BUDGET_NS,
    MAX_JITTER_NS,
)
from .stats import LatencyStats, LatencyMonitor


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    NETWORK_LATENCY = "network_latency"
    SERIALIZATION = "serialization"
    SYNCHRONIZATION = "synchronization"
    COMPUTATION = "computation"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    CLOCK_SKEW = "clock_skew"
    CONTENTION = "contention"
    PROTOCOL_OVERHEAD = "protocol_overhead"


class OptimizationCategory(Enum):
    """Categories of optimization recommendations."""
    TOPOLOGY = "topology"
    BUFFER_SIZE = "buffer_size"
    ALGORITHM = "algorithm"
    HARDWARE = "hardware"
    CONFIGURATION = "configuration"
    CODE = "code"


@dataclass
class ProfileSample:
    """Single profiling sample."""
    timestamp_ns: int
    operation: str
    phase: str
    duration_ns: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyBreakdown:
    """Breakdown of latency into component phases."""
    total_ns: float
    phases: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.phases:
            self.phases = {}

    @property
    def overhead_ns(self) -> float:
        """Unaccounted overhead."""
        accounted = sum(self.phases.values())
        return max(0, self.total_ns - accounted)

    def percentage(self, phase: str) -> float:
        """Get percentage of total for a phase."""
        if self.total_ns <= 0:
            return 0.0
        return 100.0 * self.phases.get(phase, 0) / self.total_ns

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_ns': self.total_ns,
            'phases': self.phases,
            'overhead_ns': self.overhead_ns,
        }


@dataclass
class Bottleneck:
    """Identified performance bottleneck."""
    type: BottleneckType
    severity: float  # 0-1, higher is worse
    description: str
    affected_operations: List[str]
    evidence: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'severity': self.severity,
            'description': self.description,
            'affected_operations': self.affected_operations,
            'evidence': self.evidence,
        }


@dataclass
class Recommendation:
    """Optimization recommendation."""
    category: OptimizationCategory
    priority: int  # 1-5, higher is more important
    title: str
    description: str
    expected_improvement: str
    implementation_effort: str  # low, medium, high

    def to_dict(self) -> dict:
        return {
            'category': self.category.value,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'expected_improvement': self.expected_improvement,
            'implementation_effort': self.implementation_effort,
        }


class CriticalPathProfiler:
    """
    Profiles critical paths in ACCL-Q operations.

    Tracks timing through each phase of collective operations
    to identify bottlenecks.
    """

    def __init__(self):
        self._samples: List[ProfileSample] = []
        self._active_spans: Dict[str, int] = {}  # operation -> start time
        self._lock = threading.Lock()

        # Phase definitions for each operation
        self._operation_phases = {
            'broadcast': ['serialize', 'tree_down', 'deserialize'],
            'reduce': ['serialize', 'tree_up', 'combine', 'deserialize'],
            'allreduce': ['serialize', 'tree_up', 'combine', 'tree_down', 'deserialize'],
            'barrier': ['signal', 'wait', 'release'],
            'scatter': ['serialize', 'route', 'deserialize'],
            'gather': ['serialize', 'route', 'deserialize'],
            'feedback': ['measure', 'communicate', 'decode', 'apply'],
        }

    def start_operation(self, operation: str, metadata: Optional[Dict] = None) -> str:
        """
        Start profiling an operation.

        Args:
            operation: Operation name
            metadata: Optional metadata

        Returns:
            Operation ID for matching with end_operation
        """
        op_id = f"{operation}_{time.perf_counter_ns()}"
        with self._lock:
            self._active_spans[op_id] = time.perf_counter_ns()
        return op_id

    def end_operation(self, op_id: str) -> Optional[float]:
        """
        End profiling an operation.

        Args:
            op_id: Operation ID from start_operation

        Returns:
            Duration in nanoseconds
        """
        end_time = time.perf_counter_ns()
        with self._lock:
            if op_id not in self._active_spans:
                return None
            start_time = self._active_spans.pop(op_id)
            duration = end_time - start_time
            operation = op_id.rsplit('_', 1)[0]

            self._samples.append(ProfileSample(
                timestamp_ns=start_time,
                operation=operation,
                phase='total',
                duration_ns=duration,
            ))

            return duration

    def record_phase(self, operation: str, phase: str,
                     duration_ns: float, metadata: Optional[Dict] = None) -> None:
        """
        Record a phase timing.

        Args:
            operation: Operation name
            phase: Phase name
            duration_ns: Phase duration
            metadata: Optional metadata
        """
        with self._lock:
            self._samples.append(ProfileSample(
                timestamp_ns=time.perf_counter_ns(),
                operation=operation,
                phase=phase,
                duration_ns=duration_ns,
                metadata=metadata or {},
            ))

    def get_breakdown(self, operation: str) -> LatencyBreakdown:
        """
        Get latency breakdown for an operation.

        Args:
            operation: Operation name

        Returns:
            LatencyBreakdown with phase timings
        """
        with self._lock:
            op_samples = [s for s in self._samples if s.operation == operation]

        if not op_samples:
            return LatencyBreakdown(total_ns=0)

        # Get total latency
        total_samples = [s for s in op_samples if s.phase == 'total']
        total_ns = np.mean([s.duration_ns for s in total_samples]) if total_samples else 0

        # Get phase latencies
        phases = {}
        for phase in self._operation_phases.get(operation, []):
            phase_samples = [s for s in op_samples if s.phase == phase]
            if phase_samples:
                phases[phase] = np.mean([s.duration_ns for s in phase_samples])

        return LatencyBreakdown(total_ns=total_ns, phases=phases)

    def get_critical_path(self, operation: str) -> List[Tuple[str, float]]:
        """
        Identify critical path phases (ordered by duration).

        Args:
            operation: Operation name

        Returns:
            List of (phase, duration) tuples, sorted by duration descending
        """
        breakdown = self.get_breakdown(operation)
        return sorted(breakdown.phases.items(), key=lambda x: x[1], reverse=True)

    def clear(self) -> None:
        """Clear all profiling data."""
        with self._lock:
            self._samples.clear()
            self._active_spans.clear()


class BottleneckAnalyzer:
    """
    Analyzes profiling data to identify performance bottlenecks.

    Uses heuristics and thresholds to detect common performance issues.
    """

    def __init__(self, profiler: CriticalPathProfiler,
                 monitor: Optional[LatencyMonitor] = None):
        """
        Initialize analyzer.

        Args:
            profiler: Profiler with collected data
            monitor: Optional latency monitor for additional data
        """
        self.profiler = profiler
        self.monitor = monitor

        # Thresholds for bottleneck detection
        self._thresholds = {
            'network_latency_ratio': 0.7,      # Network > 70% of total
            'serialization_ratio': 0.3,        # Serialization > 30%
            'jitter_ratio': 0.2,               # Jitter > 20% of mean
            'sync_overhead_ratio': 0.4,        # Sync overhead > 40%
            'target_violation_rate': 0.05,     # > 5% violations
        }

    def analyze(self) -> List[Bottleneck]:
        """
        Analyze profiling data and identify bottlenecks.

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []

        # Analyze each operation type
        for op in ['broadcast', 'reduce', 'allreduce', 'barrier', 'feedback']:
            breakdown = self.profiler.get_breakdown(op)
            if breakdown.total_ns <= 0:
                continue

            # Check for network bottleneck
            network_phases = ['tree_down', 'tree_up', 'route', 'communicate']
            network_time = sum(breakdown.phases.get(p, 0) for p in network_phases)
            if network_time / breakdown.total_ns > self._thresholds['network_latency_ratio']:
                bottlenecks.append(Bottleneck(
                    type=BottleneckType.NETWORK_LATENCY,
                    severity=network_time / breakdown.total_ns,
                    description=f"Network communication dominates {op} latency",
                    affected_operations=[op],
                    evidence={
                        'network_time_ns': network_time,
                        'total_time_ns': breakdown.total_ns,
                        'ratio': network_time / breakdown.total_ns,
                    }
                ))

            # Check for serialization bottleneck
            serial_phases = ['serialize', 'deserialize']
            serial_time = sum(breakdown.phases.get(p, 0) for p in serial_phases)
            if serial_time / breakdown.total_ns > self._thresholds['serialization_ratio']:
                bottlenecks.append(Bottleneck(
                    type=BottleneckType.SERIALIZATION,
                    severity=serial_time / breakdown.total_ns,
                    description=f"Serialization overhead high in {op}",
                    affected_operations=[op],
                    evidence={
                        'serialization_time_ns': serial_time,
                        'total_time_ns': breakdown.total_ns,
                        'ratio': serial_time / breakdown.total_ns,
                    }
                ))

            # Check for large overhead (unaccounted time)
            if breakdown.overhead_ns / breakdown.total_ns > 0.2:
                bottlenecks.append(Bottleneck(
                    type=BottleneckType.PROTOCOL_OVERHEAD,
                    severity=breakdown.overhead_ns / breakdown.total_ns,
                    description=f"Significant unaccounted overhead in {op}",
                    affected_operations=[op],
                    evidence={
                        'overhead_ns': breakdown.overhead_ns,
                        'total_time_ns': breakdown.total_ns,
                        'ratio': breakdown.overhead_ns / breakdown.total_ns,
                    }
                ))

        # Analyze jitter from monitor
        if self.monitor:
            stats = self.monitor.get_stats()
            for op, s in stats.items():
                if s.mean_ns > 0 and s.std_ns / s.mean_ns > self._thresholds['jitter_ratio']:
                    bottlenecks.append(Bottleneck(
                        type=BottleneckType.CONTENTION,
                        severity=min(1.0, s.std_ns / s.mean_ns),
                        description=f"High jitter in {op.name} suggests contention",
                        affected_operations=[op.name],
                        evidence={
                            'mean_ns': s.mean_ns,
                            'std_ns': s.std_ns,
                            'jitter_ratio': s.std_ns / s.mean_ns,
                        }
                    ))

            # Check target violations
            violations = self.monitor.get_violations()
            for op, count in violations.items():
                rate = self.monitor.get_violation_rate(op)
                if rate > self._thresholds['target_violation_rate']:
                    bottlenecks.append(Bottleneck(
                        type=BottleneckType.NETWORK_LATENCY,
                        severity=min(1.0, rate * 5),  # Scale to 0-1
                        description=f"{op.name} frequently exceeds latency target",
                        affected_operations=[op.name],
                        evidence={
                            'violation_count': count,
                            'violation_rate': rate,
                        }
                    ))

        return bottlenecks

    def get_summary(self) -> dict:
        """Get analysis summary."""
        bottlenecks = self.analyze()

        by_type = defaultdict(list)
        for b in bottlenecks:
            by_type[b.type.value].append(b.to_dict())

        return {
            'total_bottlenecks': len(bottlenecks),
            'by_type': dict(by_type),
            'most_severe': max(bottlenecks, key=lambda b: b.severity).to_dict() if bottlenecks else None,
        }


class OptimizationAdvisor:
    """
    Provides optimization recommendations based on bottleneck analysis.

    Maps identified bottlenecks to actionable recommendations.
    """

    def __init__(self, analyzer: BottleneckAnalyzer):
        self.analyzer = analyzer

        # Recommendation templates for each bottleneck type
        self._recommendations = {
            BottleneckType.NETWORK_LATENCY: [
                Recommendation(
                    category=OptimizationCategory.TOPOLOGY,
                    priority=5,
                    title="Optimize tree fanout",
                    description="Increase tree fanout to reduce depth and hops. "
                                "Current fanout may be suboptimal for your cluster size.",
                    expected_improvement="10-30% latency reduction",
                    implementation_effort="low",
                ),
                Recommendation(
                    category=OptimizationCategory.HARDWARE,
                    priority=4,
                    title="Enable Aurora link bonding",
                    description="Bond multiple Aurora lanes for higher bandwidth "
                                "on critical paths.",
                    expected_improvement="2-4x bandwidth increase",
                    implementation_effort="medium",
                ),
            ],
            BottleneckType.SERIALIZATION: [
                Recommendation(
                    category=OptimizationCategory.BUFFER_SIZE,
                    priority=4,
                    title="Use zero-copy transfers",
                    description="Align buffers to cache lines and use zero-copy DMA "
                                "to eliminate serialization overhead.",
                    expected_improvement="50-80% serialization reduction",
                    implementation_effort="medium",
                ),
                Recommendation(
                    category=OptimizationCategory.CODE,
                    priority=3,
                    title="Reduce message size",
                    description="Use compact data representations (e.g., fixed-point "
                                "instead of float for syndromes).",
                    expected_improvement="20-40% serialization reduction",
                    implementation_effort="low",
                ),
            ],
            BottleneckType.SYNCHRONIZATION: [
                Recommendation(
                    category=OptimizationCategory.ALGORITHM,
                    priority=5,
                    title="Use asynchronous collectives",
                    description="Overlap communication with computation using "
                                "non-blocking collective operations.",
                    expected_improvement="Hide 50-90% of communication latency",
                    implementation_effort="medium",
                ),
            ],
            BottleneckType.CONTENTION: [
                Recommendation(
                    category=OptimizationCategory.CONFIGURATION,
                    priority=4,
                    title="Stagger operation timing",
                    description="Add small random delays to desynchronize traffic "
                                "patterns and reduce contention.",
                    expected_improvement="30-50% jitter reduction",
                    implementation_effort="low",
                ),
                Recommendation(
                    category=OptimizationCategory.TOPOLOGY,
                    priority=3,
                    title="Review link utilization",
                    description="Balance traffic across available links to avoid "
                                "hotspots.",
                    expected_improvement="20-40% jitter reduction",
                    implementation_effort="medium",
                ),
            ],
            BottleneckType.CLOCK_SKEW: [
                Recommendation(
                    category=OptimizationCategory.HARDWARE,
                    priority=5,
                    title="Improve clock distribution",
                    description="Use hardware clock distribution with matched cable "
                                "lengths and proper termination.",
                    expected_improvement="Sub-nanosecond sync accuracy",
                    implementation_effort="high",
                ),
                Recommendation(
                    category=OptimizationCategory.ALGORITHM,
                    priority=3,
                    title="Increase sync frequency",
                    description="Run clock synchronization more frequently to track "
                                "drift.",
                    expected_improvement="2-5x better sync accuracy",
                    implementation_effort="low",
                ),
            ],
            BottleneckType.PROTOCOL_OVERHEAD: [
                Recommendation(
                    category=OptimizationCategory.ALGORITHM,
                    priority=4,
                    title="Use lightweight protocol",
                    description="Switch to minimal protocol for known-good paths. "
                                "Eliminate unnecessary handshakes.",
                    expected_improvement="20-50% overhead reduction",
                    implementation_effort="medium",
                ),
            ],
        }

    def get_recommendations(self) -> List[Recommendation]:
        """
        Generate recommendations based on current bottlenecks.

        Returns:
            List of prioritized recommendations
        """
        bottlenecks = self.analyzer.analyze()
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck.type in self._recommendations:
                # Add recommendations with severity weighting
                for rec in self._recommendations[bottleneck.type]:
                    # Adjust priority based on bottleneck severity
                    adjusted_rec = Recommendation(
                        category=rec.category,
                        priority=min(5, int(rec.priority * (0.5 + bottleneck.severity))),
                        title=rec.title,
                        description=rec.description,
                        expected_improvement=rec.expected_improvement,
                        implementation_effort=rec.implementation_effort,
                    )
                    recommendations.append(adjusted_rec)

        # Deduplicate and sort by priority
        seen = set()
        unique_recommendations = []
        for rec in sorted(recommendations, key=lambda r: r.priority, reverse=True):
            if rec.title not in seen:
                seen.add(rec.title)
                unique_recommendations.append(rec)

        return unique_recommendations

    def get_top_recommendations(self, n: int = 5) -> List[Recommendation]:
        """Get top N recommendations."""
        return self.get_recommendations()[:n]


class PerformanceRegressor:
    """
    Detects performance regressions by comparing against baselines.

    Maintains historical performance data and alerts on degradation.
    """

    def __init__(self, baseline_path: Optional[Path] = None):
        """
        Initialize regressor.

        Args:
            baseline_path: Path to baseline performance data
        """
        self.baseline_path = baseline_path
        self._baseline: Dict[str, LatencyStats] = {}
        self._current: Dict[str, LatencyStats] = {}

        # Regression thresholds
        self._thresholds = {
            'mean_increase': 0.10,   # 10% increase in mean
            'p99_increase': 0.20,    # 20% increase in p99
            'jitter_increase': 0.50, # 50% increase in jitter
        }

        if baseline_path and baseline_path.exists():
            self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline from file."""
        with open(self.baseline_path, 'r') as f:
            data = json.load(f)
            for op, stats_data in data.items():
                self._baseline[op] = LatencyStats(**stats_data)

    def save_baseline(self, path: Optional[Path] = None) -> None:
        """Save current measurements as baseline."""
        path = path or self.baseline_path
        if not path:
            raise ValueError("No path specified for baseline")

        data = {}
        for op, stats in self._current.items():
            data[op] = {
                'count': stats.count,
                'mean_ns': stats.mean_ns,
                'std_ns': stats.std_ns,
                'min_ns': stats.min_ns,
                'max_ns': stats.max_ns,
                'p50_ns': stats.p50_ns,
                'p95_ns': stats.p95_ns,
                'p99_ns': stats.p99_ns,
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def update_current(self, operation: str, stats: LatencyStats) -> None:
        """Update current measurements for an operation."""
        self._current[operation] = stats

    def update_from_monitor(self, monitor: LatencyMonitor) -> None:
        """Update current measurements from a latency monitor."""
        for op, stats in monitor.get_stats().items():
            self._current[op.name] = stats

    def check_regressions(self) -> List[dict]:
        """
        Check for performance regressions.

        Returns:
            List of regression alerts
        """
        regressions = []

        for op, current in self._current.items():
            if op not in self._baseline:
                continue

            baseline = self._baseline[op]

            # Check mean latency regression
            if baseline.mean_ns > 0:
                mean_change = (current.mean_ns - baseline.mean_ns) / baseline.mean_ns
                if mean_change > self._thresholds['mean_increase']:
                    regressions.append({
                        'operation': op,
                        'metric': 'mean_latency',
                        'baseline_ns': baseline.mean_ns,
                        'current_ns': current.mean_ns,
                        'change_percent': mean_change * 100,
                        'threshold_percent': self._thresholds['mean_increase'] * 100,
                    })

            # Check p99 latency regression
            if baseline.p99_ns > 0:
                p99_change = (current.p99_ns - baseline.p99_ns) / baseline.p99_ns
                if p99_change > self._thresholds['p99_increase']:
                    regressions.append({
                        'operation': op,
                        'metric': 'p99_latency',
                        'baseline_ns': baseline.p99_ns,
                        'current_ns': current.p99_ns,
                        'change_percent': p99_change * 100,
                        'threshold_percent': self._thresholds['p99_increase'] * 100,
                    })

            # Check jitter regression
            if baseline.std_ns > 0:
                jitter_change = (current.std_ns - baseline.std_ns) / baseline.std_ns
                if jitter_change > self._thresholds['jitter_increase']:
                    regressions.append({
                        'operation': op,
                        'metric': 'jitter',
                        'baseline_ns': baseline.std_ns,
                        'current_ns': current.std_ns,
                        'change_percent': jitter_change * 100,
                        'threshold_percent': self._thresholds['jitter_increase'] * 100,
                    })

        return regressions

    def get_comparison(self) -> dict:
        """Get full baseline vs current comparison."""
        comparison = {}

        all_ops = set(self._baseline.keys()) | set(self._current.keys())
        for op in all_ops:
            baseline = self._baseline.get(op)
            current = self._current.get(op)

            comparison[op] = {
                'baseline': {
                    'mean_ns': baseline.mean_ns if baseline else None,
                    'p99_ns': baseline.p99_ns if baseline else None,
                    'std_ns': baseline.std_ns if baseline else None,
                } if baseline else None,
                'current': {
                    'mean_ns': current.mean_ns if current else None,
                    'p99_ns': current.p99_ns if current else None,
                    'std_ns': current.std_ns if current else None,
                } if current else None,
            }

            # Add change percentages
            if baseline and current and baseline.mean_ns > 0:
                comparison[op]['changes'] = {
                    'mean_percent': (current.mean_ns - baseline.mean_ns) / baseline.mean_ns * 100,
                    'p99_percent': (current.p99_ns - baseline.p99_ns) / baseline.p99_ns * 100 if baseline.p99_ns > 0 else None,
                    'std_percent': (current.std_ns - baseline.std_ns) / baseline.std_ns * 100 if baseline.std_ns > 0 else None,
                }

        return comparison


class LatencyVisualizer:
    """
    Generates text-based visualizations of latency data.

    Produces ASCII charts and tables for terminal display.
    """

    @staticmethod
    def breakdown_bar(breakdown: LatencyBreakdown, width: int = 60) -> str:
        """
        Generate ASCII bar chart of latency breakdown.

        Args:
            breakdown: Latency breakdown to visualize
            width: Width of the bar

        Returns:
            ASCII bar chart string
        """
        if breakdown.total_ns <= 0:
            return "[No data]"

        lines = []
        lines.append(f"Total: {breakdown.total_ns:.1f}ns")
        lines.append("=" * width)

        # Sort phases by duration
        sorted_phases = sorted(breakdown.phases.items(), key=lambda x: x[1], reverse=True)

        for phase, duration in sorted_phases:
            pct = duration / breakdown.total_ns
            bar_len = int(pct * (width - 20))
            bar = "#" * bar_len
            lines.append(f"{phase:12s} |{bar:<{width-20}}| {duration:>6.1f}ns ({pct*100:>4.1f}%)")

        if breakdown.overhead_ns > 0:
            pct = breakdown.overhead_ns / breakdown.total_ns
            bar_len = int(pct * (width - 20))
            bar = "." * bar_len
            lines.append(f"{'overhead':12s} |{bar:<{width-20}}| {breakdown.overhead_ns:>6.1f}ns ({pct*100:>4.1f}%)")

        return "\n".join(lines)

    @staticmethod
    def histogram(samples: List[float], bins: int = 20, width: int = 50) -> str:
        """
        Generate ASCII histogram.

        Args:
            samples: List of sample values
            bins: Number of histogram bins
            width: Width of the histogram bars

        Returns:
            ASCII histogram string
        """
        if not samples:
            return "[No data]"

        arr = np.array(samples)
        counts, edges = np.histogram(arr, bins=bins)
        max_count = max(counts)

        lines = []
        lines.append(f"n={len(samples)}, mean={np.mean(arr):.1f}, std={np.std(arr):.1f}")
        lines.append("-" * (width + 25))

        for i, count in enumerate(counts):
            bar_len = int(count / max_count * width) if max_count > 0 else 0
            bar = "#" * bar_len
            lines.append(f"{edges[i]:>8.1f}-{edges[i+1]:>8.1f} |{bar:<{width}}| {count}")

        return "\n".join(lines)

    @staticmethod
    def comparison_table(comparison: dict) -> str:
        """
        Generate comparison table.

        Args:
            comparison: Comparison data from PerformanceRegressor

        Returns:
            ASCII table string
        """
        lines = []
        header = f"{'Operation':<15} {'Baseline':>12} {'Current':>12} {'Change':>10}"
        lines.append(header)
        lines.append("=" * len(header))

        for op, data in sorted(comparison.items()):
            baseline = data.get('baseline', {})
            current = data.get('current', {})
            changes = data.get('changes', {})

            baseline_mean = baseline.get('mean_ns') if baseline else None
            current_mean = current.get('mean_ns') if current else None
            change_pct = changes.get('mean_percent') if changes else None

            baseline_str = f"{baseline_mean:.1f}ns" if baseline_mean else "N/A"
            current_str = f"{current_mean:.1f}ns" if current_mean else "N/A"
            change_str = f"{change_pct:+.1f}%" if change_pct else "N/A"

            # Add indicator for regressions
            indicator = ""
            if change_pct and change_pct > 10:
                indicator = " (!)"
            elif change_pct and change_pct < -10:
                indicator = " (*)"

            lines.append(f"{op:<15} {baseline_str:>12} {current_str:>12} {change_str:>10}{indicator}")

        lines.append("-" * len(header))
        lines.append("(!) = regression, (*) = improvement")

        return "\n".join(lines)


class ProfilingSession:
    """
    Complete profiling session manager.

    Coordinates profiler, analyzer, advisor, and visualizer
    for comprehensive performance analysis.
    """

    def __init__(self, monitor: Optional[LatencyMonitor] = None,
                 baseline_path: Optional[Path] = None):
        """
        Initialize profiling session.

        Args:
            monitor: Optional latency monitor to include
            baseline_path: Path to baseline data
        """
        self.profiler = CriticalPathProfiler()
        self.monitor = monitor
        self.analyzer = BottleneckAnalyzer(self.profiler, monitor)
        self.advisor = OptimizationAdvisor(self.analyzer)
        self.regressor = PerformanceRegressor(baseline_path)
        self.visualizer = LatencyVisualizer()

        self._session_start = time.perf_counter_ns()

    def profile_operation(self, operation: str):
        """
        Context manager for profiling an operation.

        Usage:
            with session.profile_operation('broadcast'):
                accl.broadcast(data, root=0)
        """
        class ProfileContext:
            def __init__(ctx, profiler, op):
                ctx.profiler = profiler
                ctx.op = op
                ctx.op_id = None

            def __enter__(ctx):
                ctx.op_id = ctx.profiler.start_operation(ctx.op)
                return ctx

            def __exit__(ctx, *args):
                ctx.profiler.end_operation(ctx.op_id)
                return False

        return ProfileContext(self.profiler, operation)

    def analyze(self) -> dict:
        """Run full analysis and return results."""
        # Update regressor from monitor
        if self.monitor:
            self.regressor.update_from_monitor(self.monitor)

        return {
            'session_duration_ns': time.perf_counter_ns() - self._session_start,
            'bottlenecks': [b.to_dict() for b in self.analyzer.analyze()],
            'recommendations': [r.to_dict() for r in self.advisor.get_top_recommendations()],
            'regressions': self.regressor.check_regressions(),
        }

    def generate_report(self) -> str:
        """Generate comprehensive text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("ACCL-Q PERFORMANCE PROFILING REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Session info
        duration_s = (time.perf_counter_ns() - self._session_start) / 1e9
        lines.append(f"Session Duration: {duration_s:.2f}s")
        lines.append("")

        # Latency breakdowns
        lines.append("LATENCY BREAKDOWNS")
        lines.append("-" * 70)
        for op in ['broadcast', 'reduce', 'allreduce', 'barrier', 'feedback']:
            breakdown = self.profiler.get_breakdown(op)
            if breakdown.total_ns > 0:
                lines.append(f"\n{op.upper()}:")
                lines.append(self.visualizer.breakdown_bar(breakdown))
        lines.append("")

        # Bottlenecks
        lines.append("IDENTIFIED BOTTLENECKS")
        lines.append("-" * 70)
        bottlenecks = self.analyzer.analyze()
        if bottlenecks:
            for b in sorted(bottlenecks, key=lambda x: x.severity, reverse=True):
                lines.append(f"\n[{b.type.value}] Severity: {b.severity:.2f}")
                lines.append(f"  {b.description}")
                lines.append(f"  Affected: {', '.join(b.affected_operations)}")
        else:
            lines.append("No significant bottlenecks detected.")
        lines.append("")

        # Recommendations
        lines.append("OPTIMIZATION RECOMMENDATIONS")
        lines.append("-" * 70)
        recommendations = self.advisor.get_top_recommendations()
        if recommendations:
            for i, r in enumerate(recommendations, 1):
                lines.append(f"\n{i}. [{r.category.value}] {r.title} (Priority: {r.priority}/5)")
                lines.append(f"   {r.description}")
                lines.append(f"   Expected: {r.expected_improvement}")
                lines.append(f"   Effort: {r.implementation_effort}")
        else:
            lines.append("No recommendations at this time.")
        lines.append("")

        # Regressions
        lines.append("PERFORMANCE REGRESSIONS")
        lines.append("-" * 70)
        regressions = self.regressor.check_regressions()
        if regressions:
            for r in regressions:
                lines.append(f"\n[{r['operation']}] {r['metric']}")
                lines.append(f"  Baseline: {r['baseline_ns']:.1f}ns")
                lines.append(f"  Current:  {r['current_ns']:.1f}ns")
                lines.append(f"  Change:   {r['change_percent']:+.1f}% (threshold: {r['threshold_percent']:.0f}%)")
        else:
            lines.append("No performance regressions detected.")
        lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)
