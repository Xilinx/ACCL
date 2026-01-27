"""
ACCL-Q Measurement Feedback Pipeline

Implements end-to-end measurement-based feedback system for quantum control:
1. Measurement acquisition
2. ACCL distribution/aggregation
3. Conditional operation triggering

Total latency budget: < 500ns
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import threading

from .driver import ACCLQuantum, OperationResult
from .constants import (
    ReduceOp,
    SyncMode,
    QuantumMsgType,
    FEEDBACK_LATENCY_BUDGET_NS,
    CLOCK_PERIOD_NS,
)
from .stats import LatencyMonitor, LatencyProfiler, CollectiveOp


# ============================================================================
# Feedback Pipeline Configuration
# ============================================================================

class FeedbackMode(Enum):
    """Feedback operation modes."""
    SINGLE_QUBIT = 0      # Condition on single qubit measurement
    PARITY = 1            # Condition on parity of multiple qubits
    SYNDROME = 2          # Full QEC syndrome-based feedback
    THRESHOLD = 3         # Threshold-based soft decision


@dataclass
class FeedbackConfig:
    """Configuration for measurement feedback pipeline."""
    latency_budget_ns: float = FEEDBACK_LATENCY_BUDGET_NS
    mode: FeedbackMode = FeedbackMode.SINGLE_QUBIT
    decoder_rank: int = 0
    enable_pipelining: bool = True
    max_pending_operations: int = 4


@dataclass
class FeedbackResult:
    """Result of a feedback operation."""
    success: bool
    measurement: np.ndarray
    decision: Any
    action_taken: bool
    total_latency_ns: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        return self.total_latency_ns <= FEEDBACK_LATENCY_BUDGET_NS


# ============================================================================
# Measurement Feedback Pipeline
# ============================================================================

class MeasurementFeedbackPipeline:
    """
    End-to-end measurement feedback system.

    Implements the complete feedback loop:
    1. Acquire measurement result (local or distributed)
    2. Distribute/aggregate via ACCL collective ops
    3. Make decision (local or at decoder)
    4. Trigger conditional operation

    Timing breakdown target (500ns total):
    - Measurement acquisition: ~100ns
    - ACCL communication: ~300ns
    - Decision + trigger: ~100ns
    """

    def __init__(self, accl: ACCLQuantum,
                 config: Optional[FeedbackConfig] = None):
        """
        Initialize feedback pipeline.

        Args:
            accl: ACCL-Q driver instance
            config: Pipeline configuration
        """
        self.accl = accl
        self.config = config or FeedbackConfig()

        # Pipeline state
        self._is_armed = False
        self._pending_ops: List[Dict] = []

        # Callbacks
        self._action_callbacks: Dict[str, Callable] = {}

        # Latency tracking
        self._latency_history: List[FeedbackResult] = []

        # Pre-allocated buffers for low latency
        self._measurement_buffer = np.zeros(64, dtype=np.uint64)
        self._syndrome_buffer = np.zeros(32, dtype=np.uint64)

    def register_action(self, name: str, callback: Callable) -> None:
        """
        Register a conditional action callback.

        Args:
            name: Action identifier
            callback: Function to call when action is triggered
        """
        self._action_callbacks[name] = callback

    def arm(self) -> None:
        """Arm the feedback pipeline for operation."""
        self._is_armed = True

    def disarm(self) -> None:
        """Disarm the feedback pipeline."""
        self._is_armed = False

    # ========================================================================
    # Single-Qubit Feedback
    # ========================================================================

    def single_qubit_feedback(self, source_rank: int,
                              action_if_one: str,
                              action_if_zero: Optional[str] = None) -> FeedbackResult:
        """
        Perform single-qubit measurement feedback.

        Measures a qubit on source_rank, broadcasts result, and
        triggers conditional action on all ranks.

        Args:
            source_rank: Rank with the qubit to measure
            action_if_one: Action name to execute if measurement = 1
            action_if_zero: Optional action if measurement = 0

        Returns:
            FeedbackResult with timing breakdown
        """
        breakdown = {}
        start_ns = time.perf_counter_ns()

        # Step 1: Get measurement (simulated or from hardware)
        meas_start = time.perf_counter_ns()
        if self.accl.local_rank == source_rank:
            measurement = self._acquire_measurement(1)
        else:
            measurement = np.zeros(1, dtype=np.uint64)
        breakdown['measurement_ns'] = time.perf_counter_ns() - meas_start

        # Step 2: Broadcast measurement to all ranks
        comm_start = time.perf_counter_ns()
        result = self.accl.broadcast(measurement, root=source_rank)
        breakdown['communication_ns'] = time.perf_counter_ns() - comm_start

        if not result.success:
            return FeedbackResult(
                success=False,
                measurement=measurement,
                decision=None,
                action_taken=False,
                total_latency_ns=time.perf_counter_ns() - start_ns,
                breakdown=breakdown
            )

        # Step 3: Make decision and trigger action
        decision_start = time.perf_counter_ns()
        meas_value = result.data[0]
        action_taken = False

        if meas_value == 1 and action_if_one:
            self._trigger_action(action_if_one)
            action_taken = True
        elif meas_value == 0 and action_if_zero:
            self._trigger_action(action_if_zero)
            action_taken = True

        breakdown['decision_ns'] = time.perf_counter_ns() - decision_start

        total_latency = time.perf_counter_ns() - start_ns

        feedback_result = FeedbackResult(
            success=True,
            measurement=result.data,
            decision=meas_value,
            action_taken=action_taken,
            total_latency_ns=total_latency,
            breakdown=breakdown
        )

        self._latency_history.append(feedback_result)
        return feedback_result

    # ========================================================================
    # Parity Feedback
    # ========================================================================

    def parity_feedback(self, qubit_ranks: List[int],
                        action_if_odd: str,
                        action_if_even: Optional[str] = None) -> FeedbackResult:
        """
        Perform parity-based feedback on multiple qubits.

        Measures qubits on specified ranks, computes global parity
        via XOR allreduce, triggers action based on result.

        Args:
            qubit_ranks: Ranks with qubits to measure
            action_if_odd: Action if parity is odd (XOR = 1)
            action_if_even: Optional action if parity is even

        Returns:
            FeedbackResult with timing breakdown
        """
        breakdown = {}
        start_ns = time.perf_counter_ns()

        # Step 1: Get local measurement
        meas_start = time.perf_counter_ns()
        if self.accl.local_rank in qubit_ranks:
            local_meas = self._acquire_measurement(1)
        else:
            local_meas = np.zeros(1, dtype=np.uint64)
        breakdown['measurement_ns'] = time.perf_counter_ns() - meas_start

        # Step 2: Compute global parity via XOR allreduce
        comm_start = time.perf_counter_ns()
        result = self.accl.allreduce(local_meas, op=ReduceOp.XOR)
        breakdown['communication_ns'] = time.perf_counter_ns() - comm_start

        if not result.success:
            return FeedbackResult(
                success=False,
                measurement=local_meas,
                decision=None,
                action_taken=False,
                total_latency_ns=time.perf_counter_ns() - start_ns,
                breakdown=breakdown
            )

        # Step 3: Decision based on parity
        decision_start = time.perf_counter_ns()
        parity = result.data[0] & 1
        action_taken = False

        if parity == 1 and action_if_odd:
            self._trigger_action(action_if_odd)
            action_taken = True
        elif parity == 0 and action_if_even:
            self._trigger_action(action_if_even)
            action_taken = True

        breakdown['decision_ns'] = time.perf_counter_ns() - decision_start

        total_latency = time.perf_counter_ns() - start_ns

        return FeedbackResult(
            success=True,
            measurement=local_meas,
            decision=parity,
            action_taken=action_taken,
            total_latency_ns=total_latency,
            breakdown=breakdown
        )

    # ========================================================================
    # Syndrome-Based Feedback (QEC)
    # ========================================================================

    def syndrome_feedback(self, decoder_callback: Callable[[np.ndarray], np.ndarray]
                         ) -> FeedbackResult:
        """
        Perform full QEC syndrome-based feedback.

        1. Each rank measures local ancillas
        2. Syndromes aggregated via XOR allreduce
        3. Decoder (on decoder_rank) computes corrections
        4. Corrections scattered to all ranks
        5. Corrections applied locally

        Args:
            decoder_callback: Function that takes syndrome and returns corrections

        Returns:
            FeedbackResult with timing breakdown
        """
        breakdown = {}
        start_ns = time.perf_counter_ns()

        # Step 1: Measure local ancillas
        meas_start = time.perf_counter_ns()
        local_syndrome = self._measure_syndrome()
        breakdown['measurement_ns'] = time.perf_counter_ns() - meas_start

        # Step 2: Aggregate global syndrome
        agg_start = time.perf_counter_ns()
        result = self.accl.allreduce(local_syndrome, op=ReduceOp.XOR)
        breakdown['aggregation_ns'] = time.perf_counter_ns() - agg_start

        if not result.success:
            return FeedbackResult(
                success=False,
                measurement=local_syndrome,
                decision=None,
                action_taken=False,
                total_latency_ns=time.perf_counter_ns() - start_ns,
                breakdown=breakdown
            )

        global_syndrome = result.data

        # Step 3: Decode (at decoder rank)
        decode_start = time.perf_counter_ns()
        if self.accl.local_rank == self.config.decoder_rank:
            corrections = decoder_callback(global_syndrome)
            # Prepare corrections for each rank
            corrections_list = [corrections] * self.accl.num_ranks
        else:
            corrections_list = [np.zeros_like(local_syndrome)] * self.accl.num_ranks
        breakdown['decode_ns'] = time.perf_counter_ns() - decode_start

        # Step 4: Scatter corrections
        scatter_start = time.perf_counter_ns()
        correction_result = self.accl.scatter(
            corrections_list, root=self.config.decoder_rank
        )
        breakdown['scatter_ns'] = time.perf_counter_ns() - scatter_start

        # Step 5: Apply corrections
        apply_start = time.perf_counter_ns()
        if correction_result.success:
            self._apply_corrections(correction_result.data)
        breakdown['apply_ns'] = time.perf_counter_ns() - apply_start

        total_latency = time.perf_counter_ns() - start_ns

        return FeedbackResult(
            success=correction_result.success,
            measurement=local_syndrome,
            decision=global_syndrome,
            action_taken=True,
            total_latency_ns=total_latency,
            breakdown=breakdown
        )

    # ========================================================================
    # Pipelined Feedback
    # ========================================================================

    def start_pipelined_feedback(self, source_rank: int,
                                  action: str) -> int:
        """
        Start a pipelined feedback operation (non-blocking).

        Returns immediately, allowing overlap with other operations.

        Args:
            source_rank: Rank with measurement
            action: Action to trigger based on result

        Returns:
            Operation ID for checking completion
        """
        if not self.config.enable_pipelining:
            raise RuntimeError("Pipelining not enabled")

        op_id = len(self._pending_ops)
        self._pending_ops.append({
            'id': op_id,
            'source_rank': source_rank,
            'action': action,
            'status': 'pending',
            'result': None
        })

        # In hardware: would start non-blocking operation
        return op_id

    def check_pipelined_feedback(self, op_id: int) -> Optional[FeedbackResult]:
        """
        Check if pipelined feedback operation is complete.

        Args:
            op_id: Operation ID from start_pipelined_feedback

        Returns:
            FeedbackResult if complete, None if still pending
        """
        if op_id >= len(self._pending_ops):
            return None

        op = self._pending_ops[op_id]
        if op['status'] == 'complete':
            return op['result']

        # In hardware: check completion status
        # Simulate completion
        op['status'] = 'complete'
        op['result'] = FeedbackResult(
            success=True,
            measurement=np.array([1]),
            decision=1,
            action_taken=True,
            total_latency_ns=300
        )
        return op['result']

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _acquire_measurement(self, num_qubits: int) -> np.ndarray:
        """Acquire measurement from hardware (simulated)."""
        # In real implementation: read from FPGA measurement unit
        return np.random.randint(0, 2, num_qubits, dtype=np.uint64)

    def _measure_syndrome(self) -> np.ndarray:
        """Measure QEC syndrome ancillas (simulated)."""
        # In real implementation: measure ancilla qubits
        return np.random.randint(0, 2, 8, dtype=np.uint64)

    def _trigger_action(self, action_name: str) -> None:
        """Trigger a registered action."""
        callback = self._action_callbacks.get(action_name)
        if callback:
            callback()

    def _apply_corrections(self, corrections: np.ndarray) -> None:
        """Apply QEC corrections (simulated)."""
        # In real implementation: send correction pulses to hardware
        pass

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_latency_statistics(self) -> Dict[str, float]:
        """Get latency statistics for feedback operations."""
        if not self._latency_history:
            return {}

        latencies = [r.total_latency_ns for r in self._latency_history]
        within_budget = sum(1 for r in self._latency_history if r.within_budget)

        return {
            'count': len(latencies),
            'mean_ns': np.mean(latencies),
            'std_ns': np.std(latencies),
            'min_ns': np.min(latencies),
            'max_ns': np.max(latencies),
            'within_budget_rate': within_budget / len(latencies),
            'budget_ns': FEEDBACK_LATENCY_BUDGET_NS
        }

    def get_breakdown_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get per-stage latency breakdown statistics."""
        if not self._latency_history:
            return {}

        # Collect all breakdown keys
        all_keys = set()
        for r in self._latency_history:
            all_keys.update(r.breakdown.keys())

        stats = {}
        for key in all_keys:
            values = [r.breakdown.get(key, 0) for r in self._latency_history
                     if key in r.breakdown]
            if values:
                stats[key] = {
                    'mean_ns': np.mean(values),
                    'std_ns': np.std(values),
                    'max_ns': np.max(values)
                }

        return stats

    def clear_history(self) -> None:
        """Clear latency history."""
        self._latency_history.clear()


# ============================================================================
# Feedback Scheduler
# ============================================================================

class FeedbackScheduler:
    """
    Schedules and manages multiple feedback operations.

    Optimizes ordering and timing of feedback operations to
    minimize total latency and maximize throughput.
    """

    def __init__(self, pipeline: MeasurementFeedbackPipeline):
        """
        Initialize feedback scheduler.

        Args:
            pipeline: Feedback pipeline instance
        """
        self.pipeline = pipeline
        self._schedule: List[Dict] = []
        self._lock = threading.Lock()

    def add_feedback(self, feedback_type: FeedbackMode,
                     priority: int = 0, **kwargs) -> int:
        """
        Add feedback operation to schedule.

        Args:
            feedback_type: Type of feedback operation
            priority: Priority (higher = more urgent)
            **kwargs: Operation-specific arguments

        Returns:
            Schedule entry ID
        """
        with self._lock:
            entry_id = len(self._schedule)
            self._schedule.append({
                'id': entry_id,
                'type': feedback_type,
                'priority': priority,
                'kwargs': kwargs,
                'status': 'pending'
            })
            return entry_id

    def execute_schedule(self) -> List[FeedbackResult]:
        """
        Execute all scheduled feedback operations.

        Operations are executed in priority order.

        Returns:
            List of FeedbackResults
        """
        with self._lock:
            # Sort by priority (descending)
            sorted_schedule = sorted(
                self._schedule,
                key=lambda x: x['priority'],
                reverse=True
            )

            results = []
            for entry in sorted_schedule:
                result = self._execute_entry(entry)
                results.append(result)
                entry['status'] = 'complete'
                entry['result'] = result

            return results

    def _execute_entry(self, entry: Dict) -> FeedbackResult:
        """Execute a single schedule entry."""
        feedback_type = entry['type']
        kwargs = entry['kwargs']

        if feedback_type == FeedbackMode.SINGLE_QUBIT:
            return self.pipeline.single_qubit_feedback(**kwargs)
        elif feedback_type == FeedbackMode.PARITY:
            return self.pipeline.parity_feedback(**kwargs)
        elif feedback_type == FeedbackMode.SYNDROME:
            return self.pipeline.syndrome_feedback(**kwargs)
        else:
            raise ValueError(f"Unknown feedback type: {feedback_type}")

    def clear_schedule(self) -> None:
        """Clear the schedule."""
        with self._lock:
            self._schedule.clear()
