# ACCL-Q API Reference

Complete API documentation for the ACCL-Q (Quantum-Optimized Collective Communication Library).

## Table of Contents

1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [Collective Operations](#collective-operations)
4. [Clock Synchronization](#clock-synchronization)
5. [Quantum-Specific Operations](#quantum-specific-operations)
6. [Statistics and Monitoring](#statistics-and-monitoring)
7. [Constants and Configuration](#constants-and-configuration)

---

## Overview

ACCL-Q provides sub-microsecond collective communication operations optimized for quantum control systems. It supports:

- **Deterministic timing** with hardware synchronization
- **Sub-microsecond collective operations** (<500ns total feedback latency)
- **Clock synchronization** across nodes (<1ns phase error)
- **Integration with QubiC and QICK** quantum control frameworks

### Quick Start

```python
from accl_quantum import ACCLQuantum, ACCLMode, ReduceOp

# Initialize driver
accl = ACCLQuantum(num_ranks=8, local_rank=0)
accl.configure(mode=ACCLMode.DETERMINISTIC)
accl.sync_clocks()

# Broadcast measurement result
result = accl.broadcast(measurement, root=source_rank)

# Compute global syndrome via XOR reduction
syndrome = accl.allreduce(local_syndrome, op=ReduceOp.XOR)
```

---

## Core Classes

### ACCLQuantum

Main driver class for quantum-optimized collective communication.

```python
class ACCLQuantum:
    def __init__(self, num_ranks: int, local_rank: int,
                 config: Optional[ACCLConfig] = None)
```

**Parameters:**
- `num_ranks` (int): Total number of ranks in the system
- `local_rank` (int): This node's rank (0-indexed)
- `config` (ACCLConfig, optional): Configuration object

**Attributes:**
- `num_ranks` (int): Total number of ranks
- `local_rank` (int): This node's rank
- `config` (ACCLConfig): Configuration object

**Context Manager Support:**
```python
with ACCLQuantum(num_ranks=4, local_rank=0) as accl:
    accl.broadcast(data, root=0)
```

---

### ACCLConfig

Configuration dataclass for ACCL-Q.

```python
@dataclass
class ACCLConfig:
    num_ranks: int
    local_rank: int
    timeout_ns: int = 10_000_000  # 10ms default
    enable_latency_monitoring: bool = True
    enable_hardware_sync: bool = True
    max_message_size: int = 4096
    tree_fanout: int = 4
```

**Methods:**
- `validate()`: Validate configuration, raises ValueError if invalid

---

### OperationResult

Result of an ACCL-Q operation.

```python
@dataclass
class OperationResult:
    status: OperationStatus
    data: Optional[np.ndarray] = None
    latency_ns: float = 0.0
    timestamp_ns: int = 0
```

**Properties:**
- `success` (bool): True if operation completed successfully

---

## Collective Operations

### broadcast

Broadcast data from root to all ranks.

```python
def broadcast(self, data: np.ndarray, root: int,
              sync: SyncMode = None) -> OperationResult
```

**Parameters:**
- `data` (np.ndarray): Data to broadcast (at root) or receive buffer (others)
- `root` (int): Rank that sends the data
- `sync` (SyncMode, optional): Synchronization mode override

**Returns:** OperationResult with received data

**Latency Target:** <300ns for 8 ranks

**Example:**
```python
# At rank 0 (root)
measurement = np.array([0, 1, 1, 0], dtype=np.uint8)
result = accl.broadcast(measurement, root=0)

# At other ranks
buffer = np.zeros(4, dtype=np.uint8)
result = accl.broadcast(buffer, root=0)
print(result.data)  # [0, 1, 1, 0]
```

---

### reduce

Reduce data to root using specified operation.

```python
def reduce(self, data: np.ndarray, op: ReduceOp, root: int,
           sync: SyncMode = None) -> OperationResult
```

**Parameters:**
- `data` (np.ndarray): Local data to contribute
- `op` (ReduceOp): Reduction operation (XOR, ADD, MAX, MIN)
- `root` (int): Rank to receive result
- `sync` (SyncMode, optional): Synchronization mode override

**Returns:** OperationResult with reduced data (only at root, None at others)

**Latency Target:** <400ns for 8 ranks

---

### allreduce

Reduce and distribute result to all ranks.

```python
def allreduce(self, data: np.ndarray, op: ReduceOp,
              sync: SyncMode = None) -> OperationResult
```

**Parameters:**
- `data` (np.ndarray): Local data to contribute
- `op` (ReduceOp): Reduction operation
- `sync` (SyncMode, optional): Synchronization mode override

**Returns:** OperationResult with reduced data (at all ranks)

**Example:**
```python
# Compute global parity
local_parity = np.array([measure_qubit(i)], dtype=np.uint8)
result = accl.allreduce(local_parity, op=ReduceOp.XOR)
global_parity = result.data[0]
```

---

### scatter

Scatter different data to each rank from root.

```python
def scatter(self, data: Union[np.ndarray, List[np.ndarray]], root: int,
            sync: SyncMode = None) -> OperationResult
```

**Parameters:**
- `data`: Array of arrays (at root) - one per rank
- `root` (int): Rank that sends the data
- `sync` (SyncMode, optional): Synchronization mode override

**Returns:** OperationResult with this rank's portion

---

### gather

Gather data from all ranks to root.

```python
def gather(self, data: np.ndarray, root: int,
           sync: SyncMode = None) -> OperationResult
```

**Parameters:**
- `data` (np.ndarray): Local data to send
- `root` (int): Rank to receive all data
- `sync` (SyncMode, optional): Synchronization mode override

**Returns:** OperationResult with gathered data (at root only)

---

### allgather

Gather data from all ranks to all ranks.

```python
def allgather(self, data: np.ndarray,
              sync: SyncMode = None) -> OperationResult
```

**Parameters:**
- `data` (np.ndarray): Local data to contribute
- `sync` (SyncMode, optional): Synchronization mode override

**Returns:** OperationResult with all gathered data

---

### barrier

Synchronize all ranks with guaranteed timing.

```python
def barrier(self, timeout_ns: Optional[int] = None) -> OperationResult
```

**Parameters:**
- `timeout_ns` (int, optional): Operation timeout

**Returns:** OperationResult indicating success/failure

**Timing Guarantee:** All ranks release within <2ns of each other

---

## Clock Synchronization

### sync_clocks

Synchronize clocks across all ranks.

```python
def sync_clocks(self, timeout_us: int = SYNC_TIMEOUT_US) -> bool
```

**Parameters:**
- `timeout_us` (int): Timeout for synchronization in microseconds

**Returns:** True if synchronization successful

**Target Accuracy:** <1ns phase error

---

### get_global_counter

Get current synchronized global counter value.

```python
def get_global_counter(self) -> int
```

**Returns:** Global counter value (cycles)

---

### get_sync_status

Get clock synchronization status.

```python
def get_sync_status(self) -> dict
```

**Returns:** Dictionary with:
- `synchronized` (bool): Whether clocks are synchronized
- `counter_offset_cycles` (int): Offset from master
- `phase_error_ns` (float): Phase error in nanoseconds
- `global_counter` (int): Current global counter value

---

## Quantum-Specific Operations

### distribute_measurement

Distribute measurement result to all control boards.

```python
def distribute_measurement(self, measurement: np.ndarray,
                           source_rank: int) -> OperationResult
```

**Parameters:**
- `measurement` (np.ndarray): Measurement outcomes array
- `source_rank` (int): Rank that performed the measurement

**Returns:** OperationResult with measurement data

Optimized for measurement-based feedback where one qubit's measurement determines operations on other qubits.

---

### aggregate_syndrome

Aggregate QEC syndrome data via XOR reduction.

```python
def aggregate_syndrome(self, local_syndrome: np.ndarray) -> OperationResult
```

**Parameters:**
- `local_syndrome` (np.ndarray): Local syndrome bits

**Returns:** OperationResult with global syndrome (at all ranks)

Computes global syndrome for quantum error correction by XORing local syndromes from all ranks.

---

### distribute_correction

Distribute decoder corrections to individual control boards.

```python
def distribute_correction(self, corrections: List[np.ndarray],
                          decoder_rank: int) -> OperationResult
```

**Parameters:**
- `corrections`: Correction data for each rank
- `decoder_rank` (int): Rank running the decoder

**Returns:** OperationResult with this rank's correction

---

### synchronized_trigger

Schedule synchronized trigger at specified global counter value.

```python
def synchronized_trigger(self, trigger_time: int) -> bool
```

**Parameters:**
- `trigger_time` (int): Global counter value for trigger

**Returns:** True if trigger scheduled successfully

All ranks will trigger within <2ns of each other.

---

## Statistics and Monitoring

### LatencyMonitor

Real-time latency monitoring for ACCL-Q operations.

```python
class LatencyMonitor:
    def __init__(self, window_size: int = 1000,
                 enable_alerts: bool = True)
```

**Methods:**

#### record
```python
def record(self, operation: CollectiveOp, latency_ns: float,
           num_ranks: int, root_rank: Optional[int] = None,
           success: bool = True, **metadata) -> None
```

#### get_stats
```python
def get_stats(self, operation: Optional[CollectiveOp] = None
              ) -> Dict[CollectiveOp, LatencyStats]
```

#### get_histogram
```python
def get_histogram(self, operation: CollectiveOp,
                  bin_width_ns: float = 10.0) -> Tuple[np.ndarray, np.ndarray]
```

#### add_alert_callback
```python
def add_alert_callback(self, callback: callable) -> None
```
Callback signature: `callback(operation, latency_ns, target_ns)`

#### summary
```python
def summary(self) -> str
```

---

### LatencyStats

Statistics for latency measurements.

```python
@dataclass
class LatencyStats:
    count: int
    mean_ns: float
    std_ns: float
    min_ns: float
    max_ns: float
    p50_ns: float
    p95_ns: float
    p99_ns: float
```

**Methods:**
- `from_samples(samples: List[float]) -> LatencyStats`: Create from samples
- `meets_target(target_ns, jitter_target_ns) -> bool`: Check if targets met

---

### ACCLQuantum Statistics Methods

#### get_latency_stats
```python
def get_latency_stats(self, operation: Optional[CollectiveOp] = None) -> dict
```

#### get_monitor
```python
def get_monitor(self) -> Optional[LatencyMonitor]
```

#### validate_timing
```python
def validate_timing(self) -> dict
```
Returns validation results with pass/fail for each operation.

---

## Constants and Configuration

### Enums

#### ACCLMode
```python
class ACCLMode(Enum):
    STANDARD = "standard"           # Standard latency-optimized
    DETERMINISTIC = "deterministic" # Deterministic timing
    LOW_LATENCY = "low_latency"     # Minimum latency
```

#### SyncMode
```python
class SyncMode(Enum):
    NONE = "none"           # No synchronization
    SOFTWARE = "software"   # Software barrier
    HARDWARE = "hardware"   # Hardware-synchronized
```

#### ReduceOp
```python
class ReduceOp(Enum):
    XOR = "xor"   # Bitwise XOR (for syndrome aggregation)
    ADD = "add"   # Addition
    MAX = "max"   # Maximum
    MIN = "min"   # Minimum
```

#### CollectiveOp
```python
class CollectiveOp(Enum):
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    ALLREDUCE = "allreduce"
    SCATTER = "scatter"
    GATHER = "gather"
    ALLGATHER = "allgather"
    BARRIER = "barrier"
```

#### OperationStatus
```python
class OperationStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SYNC_FAILED = "sync_failed"
```

---

### Timing Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CLOCK_PERIOD_NS` | 4.069 | Clock period at 245.76 MHz |
| `TARGET_P2P_LATENCY_NS` | 200 | Point-to-point latency target |
| `TARGET_BROADCAST_LATENCY_NS` | 300 | Broadcast latency target |
| `TARGET_REDUCE_LATENCY_NS` | 400 | Reduce latency target |
| `MAX_JITTER_NS` | 10 | Maximum allowed jitter |
| `FEEDBACK_LATENCY_BUDGET_NS` | 500 | Total feedback budget |
| `SYNC_TIMEOUT_US` | 1000 | Clock sync timeout |
| `MAX_RANKS` | 64 | Maximum supported ranks |

---

## Error Handling

All operations return `OperationResult` with status indicating success or failure:

```python
result = accl.broadcast(data, root=0)
if not result.success:
    if result.status == OperationStatus.TIMEOUT:
        print("Operation timed out")
    elif result.status == OperationStatus.SYNC_FAILED:
        print("Clock synchronization failed")
    else:
        print(f"Operation failed: {result.status}")
```

---

## Thread Safety

All `ACCLQuantum` methods are thread-safe and can be called concurrently from multiple threads. Internal state is protected by reentrant locks.

---

## See Also

- [Integration Guide](integration_guide.md) - QubiC and QICK integration
- [Performance Tuning](performance_tuning.md) - Optimization guide
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
