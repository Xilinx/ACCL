"""
ACCL-Q: Quantum-Optimized Alveo Collective Communication Library

This package provides Python bindings for ACCL-Q, enabling quantum control
systems to perform low-latency collective communication operations.

Key features:
- Sub-microsecond collective operations (broadcast, reduce, barrier)
- Hardware-synchronized timing with < 10ns jitter
- Integration with QubiC and QICK quantum control frameworks
- Real-time measurement feedback within coherence time budgets

Example usage:
    from accl_quantum import ACCLQuantum, ReduceOp, SyncMode

    # Initialize ACCL-Q
    accl = ACCLQuantum(num_ranks=8, local_rank=0)
    accl.configure(mode=ACCLMode.DETERMINISTIC)
    accl.sync_clocks()

    # Perform collective operations
    result = accl.allreduce(local_syndrome, op=ReduceOp.XOR)
    accl.broadcast(measurement_result, root=decoder_rank)
"""

from .driver import ACCLQuantum
from .constants import (
    ACCLMode,
    ReduceOp,
    SyncMode,
    QuantumMsgType,
    CLOCK_PERIOD_NS,
    TARGET_P2P_LATENCY_NS,
    TARGET_BROADCAST_LATENCY_NS,
    TARGET_REDUCE_LATENCY_NS,
    MAX_JITTER_NS,
    FEEDBACK_LATENCY_BUDGET_NS,
)
from .stats import LatencyStats, LatencyMonitor
from .integrations import QubiCIntegration, QICKIntegration

__version__ = "0.1.0"
__all__ = [
    "ACCLQuantum",
    "ACCLMode",
    "ReduceOp",
    "SyncMode",
    "QuantumMsgType",
    "LatencyStats",
    "LatencyMonitor",
    "QubiCIntegration",
    "QICKIntegration",
    "CLOCK_PERIOD_NS",
    "TARGET_P2P_LATENCY_NS",
    "TARGET_BROADCAST_LATENCY_NS",
    "TARGET_REDUCE_LATENCY_NS",
    "MAX_JITTER_NS",
    "FEEDBACK_LATENCY_BUDGET_NS",
]
