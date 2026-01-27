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

from .driver import ACCLQuantum, OperationResult
from .constants import (
    ACCLMode,
    ACCLConfig,
    ReduceOp,
    SyncMode,
    CollectiveOp,
    OperationStatus,
    QuantumMsgType,
    LatencyBudget,
    CLOCK_PERIOD_NS,
    TARGET_P2P_LATENCY_NS,
    TARGET_BROADCAST_LATENCY_NS,
    TARGET_REDUCE_LATENCY_NS,
    MAX_JITTER_NS,
    FEEDBACK_LATENCY_BUDGET_NS,
)
from .stats import LatencyStats, LatencyMonitor, LatencyProfiler
from .integrations import QubiCIntegration, QICKIntegration, UnifiedQuantumControl
from .feedback import MeasurementFeedbackPipeline, FeedbackScheduler
from .deployment import (
    BoardConfig,
    BoardType,
    DeploymentConfig,
    DeploymentManager,
    DeploymentState,
    NetworkTopology,
    TopologyBuilder,
    BoardDiscovery,
)
from .emulator import (
    RealisticQubitEmulator,
    QubitState,
    NoiseParameters,
    GateType,
    QuantumCircuitValidator,
)
from .profiler import (
    CriticalPathProfiler,
    BottleneckAnalyzer,
    OptimizationAdvisor,
    PerformanceRegressor,
    LatencyVisualizer,
    ProfilingSession,
    LatencyBreakdown,
    Bottleneck,
    Recommendation,
)

__version__ = "0.2.0"
__all__ = [
    # Core driver
    "ACCLQuantum",
    "OperationResult",
    "ACCLConfig",
    # Operation modes and types
    "ACCLMode",
    "ReduceOp",
    "SyncMode",
    "CollectiveOp",
    "OperationStatus",
    "QuantumMsgType",
    "LatencyBudget",
    # Statistics and monitoring
    "LatencyStats",
    "LatencyMonitor",
    "LatencyProfiler",
    # Framework integrations
    "QubiCIntegration",
    "QICKIntegration",
    "UnifiedQuantumControl",
    # Feedback pipeline
    "MeasurementFeedbackPipeline",
    "FeedbackScheduler",
    # Deployment
    "BoardConfig",
    "BoardType",
    "DeploymentConfig",
    "DeploymentManager",
    "DeploymentState",
    "NetworkTopology",
    "TopologyBuilder",
    "BoardDiscovery",
    # Emulation
    "RealisticQubitEmulator",
    "QubitState",
    "NoiseParameters",
    "GateType",
    "QuantumCircuitValidator",
    # Profiling
    "CriticalPathProfiler",
    "BottleneckAnalyzer",
    "OptimizationAdvisor",
    "PerformanceRegressor",
    "LatencyVisualizer",
    "ProfilingSession",
    "LatencyBreakdown",
    "Bottleneck",
    "Recommendation",
    # Constants
    "CLOCK_PERIOD_NS",
    "TARGET_P2P_LATENCY_NS",
    "TARGET_BROADCAST_LATENCY_NS",
    "TARGET_REDUCE_LATENCY_NS",
    "MAX_JITTER_NS",
    "FEEDBACK_LATENCY_BUDGET_NS",
]
