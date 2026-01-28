# ACCL-Q Performance Tuning Guide

This guide covers performance optimization strategies for achieving optimal latency in ACCL-Q operations.

## Table of Contents

1. [Latency Targets](#latency-targets)
2. [Profiling Your System](#profiling-your-system)
3. [Topology Optimization](#topology-optimization)
4. [Clock Synchronization](#clock-synchronization)
5. [Buffer Management](#buffer-management)
6. [Operation-Specific Tuning](#operation-specific-tuning)
7. [Hardware Considerations](#hardware-considerations)

---

## Latency Targets

### Default Targets

| Operation | Target | Jitter |
|-----------|--------|--------|
| Point-to-Point | <200ns | <10ns |
| Broadcast (8 ranks) | <300ns | <10ns |
| Reduce (8 ranks) | <400ns | <10ns |
| AllReduce (8 ranks) | <450ns | <10ns |
| Barrier | <100ns | <2ns |
| **Total Feedback** | **<500ns** | - |

### Quantum Requirements Context

These targets are derived from qubit coherence constraints:

- **T1 (relaxation)**: 50-100 μs typical
- **T2 (dephasing)**: 20-70 μs typical
- **QEC cycle budget**: T2 / 100 ≈ 200ns - 700ns

Feedback operations must complete within ~1% of coherence time to maintain error correction effectiveness.

---

## Profiling Your System

### Using the Profiler

```python
from accl_quantum import ACCLQuantum
from accl_quantum.profiler import ProfilingSession

# Create profiling session
accl = ACCLQuantum(num_ranks=8, local_rank=0)
session = ProfilingSession(monitor=accl.get_monitor())

# Profile operations
for i in range(100):
    with session.profile_operation('broadcast'):
        accl.broadcast(data, root=0)

    with session.profile_operation('allreduce'):
        accl.allreduce(syndrome, op=ReduceOp.XOR)

# Generate report
print(session.generate_report())
```

### Understanding the Report

```
LATENCY BREAKDOWNS
------------------

BROADCAST:
Total: 287.3ns
============================================================
tree_down    |################################          | 180.2ns (62.7%)
serialize    |########                                  |  52.1ns (18.1%)
deserialize  |######                                    |  41.5ns (14.4%)
overhead     |..                                        |  13.5ns ( 4.7%)

IDENTIFIED BOTTLENECKS
----------------------

[network_latency] Severity: 0.63
  Network communication dominates broadcast latency
  Affected: broadcast

OPTIMIZATION RECOMMENDATIONS
----------------------------

1. [topology] Optimize tree fanout (Priority: 5/5)
   Increase tree fanout to reduce depth and hops.
   Expected: 10-30% latency reduction
   Effort: low
```

### Key Metrics to Monitor

1. **Mean Latency**: Average operation time
2. **P99 Latency**: Worst-case for 99% of operations
3. **Jitter (std)**: Timing variability
4. **Violation Rate**: Percentage exceeding target

```python
stats = accl.get_latency_stats()
for op, s in stats.items():
    print(f"{op}: mean={s.mean_ns:.1f}ns, p99={s.p99_ns:.1f}ns, "
          f"jitter={s.std_ns:.1f}ns")
```

---

## Topology Optimization

### Tree Fanout Selection

The tree fanout determines how many children each node has in collective operations.

| Fanout | Depth (8 ranks) | Latency Characteristics |
|--------|-----------------|------------------------|
| 2 | 3 | Higher latency, lower per-node load |
| 4 | 2 | **Balanced (recommended)** |
| 8 | 1 | Lowest latency, highest root load |

```python
# Configure tree fanout
config = ACCLConfig(
    num_ranks=8,
    local_rank=0,
    tree_fanout=4  # Adjust based on profiling
)
accl = ACCLQuantum(config=config)
```

### Choosing Root Rank

For rooted operations (broadcast, reduce, scatter, gather), choose the root strategically:

```python
# For measurement distribution, use the measuring board as root
result = accl.distribute_measurement(measurement, source_rank=measuring_board)

# For QEC, use the decoder board as root
result = accl.distribute_correction(corrections, decoder_rank=decoder_board)
```

### Link Utilization

Balance traffic across Aurora links:

```python
from accl_quantum.deployment import TopologyBuilder, DeploymentConfig

# Build optimized topology
config = DeploymentConfig(
    name="optimized",
    num_boards=8,
    topology=NetworkTopology.TREE
)

# Use all available Aurora ports
config.links = TopologyBuilder.build_tree(
    boards,
    root_rank=0,
    fanout=4  # Utilizes 4 ports per node
)
```

---

## Clock Synchronization

### Achieving Sub-Nanosecond Sync

1. **Use Hardware Sync Mode**
```python
accl.configure(
    mode=ACCLMode.DETERMINISTIC,
    sync_mode=SyncMode.HARDWARE
)
```

2. **Verify Sync Accuracy**
```python
status = accl.get_sync_status()
print(f"Phase error: {status['phase_error_ns']:.2f}ns")

if abs(status['phase_error_ns']) > 1.0:
    # Re-synchronize
    accl.sync_clocks()
```

3. **Periodic Re-sync**
```python
import threading
import time

def periodic_sync(accl, interval_s=60):
    """Re-sync clocks periodically to counter drift."""
    while True:
        time.sleep(interval_s)
        accl.sync_clocks()

sync_thread = threading.Thread(
    target=periodic_sync,
    args=(accl,),
    daemon=True
)
sync_thread.start()
```

### Clock Distribution Best Practices

- Use matched-length cables for clock distribution
- Terminate clock signals properly
- Keep clock traces away from high-speed digital signals
- Use dedicated clock buffer ICs

---

## Buffer Management

### Pre-allocation

```python
# Pre-allocate all buffers at initialization
class ACCLBufferPool:
    def __init__(self, num_ranks, max_message_size=4096):
        self.send_buffer = np.zeros(max_message_size, dtype=np.uint8)
        self.recv_buffer = np.zeros(max_message_size, dtype=np.uint8)
        self.gather_buffer = np.zeros(
            (num_ranks, max_message_size), dtype=np.uint8
        )

    def get_send_buffer(self, size):
        return self.send_buffer[:size]

    def get_recv_buffer(self, size):
        return self.recv_buffer[:size]

# Use in operations
pool = ACCLBufferPool(num_ranks=8)

# Reuse buffers
for cycle in range(1000):
    send_buf = pool.get_send_buffer(syndrome_size)
    np.copyto(send_buf, local_syndrome)
    result = accl.allreduce(send_buf, op=ReduceOp.XOR)
```

### Memory Alignment

```python
import numpy as np

# Align to cache line (64 bytes typical)
def aligned_array(size, dtype=np.uint8, alignment=64):
    """Create cache-line aligned array."""
    extra = alignment // np.dtype(dtype).itemsize
    arr = np.zeros(size + extra, dtype=dtype)
    offset = (alignment - arr.ctypes.data % alignment) // np.dtype(dtype).itemsize
    return arr[offset:offset + size]

# Use aligned buffers
syndrome_buffer = aligned_array(64, dtype=np.uint8)
```

### Zero-Copy Operations

For maximum performance, use memory-mapped buffers that can be DMA'd directly:

```python
# Map FPGA buffer to user space (hardware-specific)
fpga_buffer = mmap_fpga_buffer(address=0x40000000, size=4096)

# Use directly in operations (zero-copy)
result = accl.broadcast(fpga_buffer, root=0)
```

---

## Operation-Specific Tuning

### Broadcast Optimization

```python
# For small messages (<64 bytes), use eager protocol
if message_size < 64:
    # Message fits in single packet
    result = accl.broadcast(small_data, root=0)
else:
    # Use rendezvous for large messages
    result = accl.broadcast(large_data, root=0)
```

### Reduce Optimization

```python
# For XOR reduction (syndrome aggregation), ensure data is byte-aligned
syndrome = np.array(syndrome_bits, dtype=np.uint8)

# Use native XOR which is hardware-accelerated
result = accl.allreduce(syndrome, op=ReduceOp.XOR)
```

### Barrier Optimization

```python
# Hardware barrier is fastest but requires sync
accl.barrier()  # Uses SyncMode.HARDWARE by default

# For debugging, use software barrier
accl.barrier(sync=SyncMode.SOFTWARE)  # Higher latency, more flexible
```

---

## Hardware Considerations

### Aurora Link Configuration

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Line Rate | 10.3125 Gbps | Per lane |
| Lanes | 4 | Bonded for bandwidth |
| Encoding | 64B/66B | Low overhead |
| Scrambling | Enabled | EMI reduction |

### FPGA Resource Usage

```
Resource          Used    Available   Utilization
--------------------------------------------------
LUTs              45,000  345,000     13%
FFs               52,000  690,000     8%
BRAMs             128     650         20%
DSPs              0       2,760       0%
Aurora Cores      4       4           100%
```

### Reducing FPGA Latency

1. **Pipeline Depth**: Reduce pipeline stages where possible
2. **Clock Domain Crossings**: Minimize CDC delays
3. **Memory Access**: Use distributed RAM for small FIFOs
4. **Routing**: Constrain critical paths

---

## Benchmarking

### Standard Benchmark Suite

```python
from accl_quantum import ACCLQuantum
import numpy as np
import time

def benchmark_operation(accl, operation, iterations=1000):
    """Benchmark a collective operation."""
    data = np.random.randint(0, 256, size=64, dtype=np.uint8)
    latencies = []

    # Warmup
    for _ in range(100):
        operation(data)

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter_ns()
        operation(data)
        latencies.append(time.perf_counter_ns() - start)

    arr = np.array(latencies)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'p50': np.percentile(arr, 50),
        'p99': np.percentile(arr, 99),
    }

# Run benchmarks
results = {}
results['broadcast'] = benchmark_operation(
    accl, lambda d: accl.broadcast(d, root=0)
)
results['allreduce'] = benchmark_operation(
    accl, lambda d: accl.allreduce(d, op=ReduceOp.XOR)
)
results['barrier'] = benchmark_operation(
    accl, lambda d: accl.barrier()
)

# Print results
for op, stats in results.items():
    print(f"{op}: mean={stats['mean']:.1f}ns, "
          f"p99={stats['p99']:.1f}ns, "
          f"jitter={stats['std']:.1f}ns")
```

### Expected Results

On properly configured hardware:

```
broadcast: mean=285.3ns, p99=312.1ns, jitter=8.2ns   [PASS]
allreduce: mean=378.5ns, p99=421.8ns, jitter=9.1ns   [PASS]
barrier:   mean=89.2ns,  p99=98.4ns,  jitter=1.8ns   [PASS]
```

---

## Troubleshooting Performance Issues

### High Latency

1. Check clock synchronization: `accl.get_sync_status()`
2. Verify topology is optimal
3. Look for network congestion
4. Check for thermal throttling

### High Jitter

1. Verify hardware sync mode is enabled
2. Check for interrupt interference
3. Isolate CPU cores for ACCL-Q threads
4. Review OS scheduler settings

### Inconsistent Results

1. Increase warmup iterations
2. Check for background processes
3. Verify consistent clock frequencies
4. Monitor for memory pressure

---

## See Also

- [API Reference](api_reference.md) - Complete API documentation
- [Integration Guide](integration_guide.md) - Framework integration
- [Troubleshooting](troubleshooting.md) - Common issues
