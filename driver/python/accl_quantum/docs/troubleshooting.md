# ACCL-Q Troubleshooting Guide

This guide covers common issues and their solutions when working with ACCL-Q.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Connection Issues](#connection-issues)
3. [Clock Synchronization Issues](#clock-synchronization-issues)
4. [Latency Issues](#latency-issues)
5. [Operation Failures](#operation-failures)
6. [Framework Integration Issues](#framework-integration-issues)
7. [Hardware Issues](#hardware-issues)
8. [Logging and Debugging](#logging-and-debugging)

---

## Quick Diagnostics

Run this diagnostic script to identify common issues:

```python
from accl_quantum import ACCLQuantum, ACCLMode, SyncMode, ReduceOp
import numpy as np

def diagnose_accl(accl):
    """Run diagnostic checks on ACCL-Q instance."""
    issues = []

    # Check configuration
    print("Configuration Check...")
    print(f"  Ranks: {accl.num_ranks}")
    print(f"  Local Rank: {accl.local_rank}")
    print(f"  Mode: {accl._mode}")
    print(f"  Sync Mode: {accl._sync_mode}")

    # Check clock sync
    print("\nClock Sync Check...")
    sync_status = accl.get_sync_status()
    print(f"  Synchronized: {sync_status['synchronized']}")
    print(f"  Phase Error: {sync_status['phase_error_ns']:.2f}ns")

    if not sync_status['synchronized']:
        issues.append("Clock not synchronized - run accl.sync_clocks()")
    elif abs(sync_status['phase_error_ns']) > 2.0:
        issues.append(f"High phase error ({sync_status['phase_error_ns']:.2f}ns)")

    # Test basic operations
    print("\nOperation Tests...")
    test_data = np.array([1, 2, 3, 4], dtype=np.uint8)

    # Broadcast
    result = accl.broadcast(test_data, root=0)
    print(f"  Broadcast: {result.status.value} ({result.latency_ns:.1f}ns)")
    if not result.success:
        issues.append(f"Broadcast failed: {result.status}")

    # Barrier
    result = accl.barrier()
    print(f"  Barrier: {result.status.value} ({result.latency_ns:.1f}ns)")
    if not result.success:
        issues.append(f"Barrier failed: {result.status}")

    # AllReduce
    result = accl.allreduce(test_data, op=ReduceOp.XOR)
    print(f"  AllReduce: {result.status.value} ({result.latency_ns:.1f}ns)")
    if not result.success:
        issues.append(f"AllReduce failed: {result.status}")

    # Latency validation
    print("\nLatency Validation...")
    validation = accl.validate_timing()
    for op, v in validation.items():
        status = "PASS" if v['overall_pass'] else "FAIL"
        print(f"  {op}: {status} (mean={v['mean_ns']:.1f}ns, target={v['target_ns']}ns)")
        if not v['overall_pass']:
            issues.append(f"{op} exceeds latency target")

    # Summary
    print("\n" + "=" * 50)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All checks passed!")

    return issues

# Run diagnostics
accl = ACCLQuantum(num_ranks=8, local_rank=0)
accl.configure(mode=ACCLMode.DETERMINISTIC)
diagnose_accl(accl)
```

---

## Connection Issues

### Problem: Board Discovery Fails

**Symptoms:**
- `discover_boards()` returns fewer boards than expected
- Timeout during discovery

**Solutions:**

1. **Check Network Connectivity**
```bash
# Ping all board IPs
for i in {0..7}; do
  ping -c 1 192.168.1.10$i
done
```

2. **Verify Multicast**
```bash
# Check multicast routing
ip maddr show
netstat -g

# Enable multicast on interface
sudo ip link set eth0 multicast on
```

3. **Check Firewall**
```bash
# Allow discovery port
sudo ufw allow 5099/udp
sudo ufw allow 5000:5010/tcp
```

4. **Increase Discovery Timeout**
```python
from accl_quantum.deployment import BoardDiscovery

discovery = BoardDiscovery(timeout_s=10.0)  # Increase from 5s default
boards = discovery.discover(expected_boards=8)
```

### Problem: Aurora Links Not Established

**Symptoms:**
- Operations timeout
- `link.is_active` returns False

**Solutions:**

1. **Check Aurora Status**
```python
# In hardware diagnostics
from accl_quantum.deployment import DeploymentManager

manager = DeploymentManager(config)
status = manager.get_status()
for rank, board in status['boards'].items():
    print(f"Board {rank}: {'online' if board['online'] else 'OFFLINE'}")
```

2. **Verify Bitstream**
```python
# Ensure correct bitstream is loaded
manager.load_bitstreams()
```

3. **Check SFP Modules**
- Verify SFP+ modules are properly seated
- Check for link LED indicators
- Try swapping SFP modules between ports

---

## Clock Synchronization Issues

### Problem: sync_clocks() Returns False

**Symptoms:**
- `accl.sync_clocks()` returns False
- `get_sync_status()` shows `synchronized: False`

**Solutions:**

1. **Increase Sync Timeout**
```python
success = accl.sync_clocks(timeout_us=5000)  # 5ms instead of 1ms
```

2. **Check Master Board**
```python
# Verify master board is online
status = accl.get_sync_status()
if not status['synchronized']:
    # Try re-initializing sync
    accl.configure(mode=ACCLMode.DETERMINISTIC)
    accl.sync_clocks()
```

3. **Verify Reference Clock**
- Check external clock source if using one
- Verify clock frequency is correct (245.76 MHz)

### Problem: High Phase Error

**Symptoms:**
- `phase_error_ns` > 2.0ns
- Inconsistent barrier release times

**Solutions:**

1. **Re-synchronize More Frequently**
```python
# Add periodic re-sync
import threading

def resync_task(accl):
    while True:
        time.sleep(30)  # Every 30 seconds
        accl.sync_clocks()

threading.Thread(target=resync_task, args=(accl,), daemon=True).start()
```

2. **Check Cable Lengths**
- Use matched-length cables for clock distribution
- Minimize cable length differences

3. **Use Hardware Sync Mode**
```python
accl.configure(
    mode=ACCLMode.DETERMINISTIC,
    sync_mode=SyncMode.HARDWARE  # Not SOFTWARE
)
```

---

## Latency Issues

### Problem: Operations Exceed Latency Targets

**Symptoms:**
- `validate_timing()` shows failures
- Feedback operations exceed 500ns

**Diagnosis:**

```python
from accl_quantum.profiler import ProfilingSession

session = ProfilingSession(monitor=accl.get_monitor())

# Profile operations
for _ in range(100):
    with session.profile_operation('broadcast'):
        accl.broadcast(data, root=0)

# Identify bottleneck
print(session.generate_report())
```

**Solutions Based on Bottleneck:**

1. **Network Latency Dominant**
```python
# Increase tree fanout to reduce hops
config.tree_fanout = 8  # Instead of 4
```

2. **Serialization Overhead**
```python
# Use smaller data types
syndrome = np.array(bits, dtype=np.uint8)  # Not int64

# Pre-allocate buffers
buffer = np.zeros(64, dtype=np.uint8)
```

3. **High Jitter**
```python
# Isolate ACCL threads from OS scheduler
import os
os.sched_setaffinity(0, {4, 5, 6, 7})  # Dedicate cores 4-7
```

### Problem: Intermittent High Latency Spikes

**Symptoms:**
- Mean latency is good, but p99 is high
- Occasional operation timeouts

**Solutions:**

1. **Disable CPU Power Management**
```bash
# Disable frequency scaling
sudo cpupower frequency-set --governor performance
```

2. **Increase Priority**
```python
import os
os.nice(-20)  # Requires root
```

3. **Check for Thermal Throttling**
```bash
# Monitor CPU temperature
watch -n 1 'sensors | grep Core'
```

---

## Operation Failures

### Problem: Timeout Status

**Symptoms:**
- `result.status == OperationStatus.TIMEOUT`

**Solutions:**

1. **Increase Timeout**
```python
accl.set_timeout(timeout_ns=100_000_000)  # 100ms

# Or per-operation
result = accl.barrier(timeout_ns=10_000_000)
```

2. **Check for Deadlock**
```python
# Ensure all ranks call the same collective
# Wrong: only some ranks call barrier
if local_rank == 0:
    accl.barrier()  # Deadlock!

# Correct: all ranks call barrier
accl.barrier()  # All ranks must call
```

3. **Verify Rank Configuration**
```python
# All ranks must have consistent num_ranks
assert accl.num_ranks == expected_num_ranks
```

### Problem: SYNC_FAILED Status

**Symptoms:**
- `result.status == OperationStatus.SYNC_FAILED`

**Solutions:**

1. **Re-sync Clocks**
```python
accl.sync_clocks()
result = accl.barrier()  # Retry
```

2. **Fall Back to Software Sync**
```python
result = accl.barrier(sync=SyncMode.SOFTWARE)
```

### Problem: Data Corruption

**Symptoms:**
- Received data doesn't match sent data
- XOR reduction gives wrong result

**Solutions:**

1. **Verify Data Types**
```python
# Ensure consistent dtypes
local_data = np.array(data, dtype=np.uint8)  # Explicit dtype
```

2. **Check Buffer Sizes**
```python
# Ensure sufficient buffer size
recv_buffer = np.zeros(len(send_data), dtype=send_data.dtype)
```

3. **Enable Debug Logging**
```python
import logging
logging.getLogger('accl_quantum').setLevel(logging.DEBUG)
```

---

## Framework Integration Issues

### QubiC Integration

**Problem: Instruction Handler Not Called**

```python
# Ensure handler is registered before use
@qubic.instruction_handler('DIST_MEAS')
def handle_dist_meas(qubit_id, source_board):
    ...

# Verify registration
assert 'DIST_MEAS' in qubic.get_handlers()
```

**Problem: Timing Mismatch with QubiC**

```python
# Sync ACCL-Q clock with QubiC reference
accl.sync_clocks()
qubic_time = qubic.get_current_time()
accl_counter = accl.get_global_counter()

# Verify alignment
print(f"QubiC time: {qubic_time}, ACCL counter: {accl_counter}")
```

### QICK Integration

**Problem: tProcessor Instruction Fails**

```python
# Verify tProcessor is initialized
assert qick.tproc is not None

# Check instruction registration
assert 'accl_broadcast' in qick.get_instructions()
```

**Problem: Pulse Timing Drift**

```python
# Re-sync before critical sequences
accl.sync_clocks()
qick.sync_all()  # QICK's internal sync

# Use synchronized trigger for precise timing
trigger_time = accl.get_global_counter() + offset
accl.synchronized_trigger(trigger_time)
```

---

## Hardware Issues

### Problem: FPGA Not Responding

**Solutions:**

1. **Check Board Power**
- Verify power LEDs
- Check power supply voltage

2. **Reload Bitstream**
```python
manager = DeploymentManager(config)
manager.load_bitstreams()
```

3. **Reset Board**
```python
# Board-specific reset (example)
sock.send(b'{"command": "reset"}')
```

### Problem: Aurora Link Errors

**Diagnosis:**
```python
# Check Aurora status registers
aurora_status = read_aurora_status()
print(f"Soft errors: {aurora_status['soft_err_count']}")
print(f"Hard errors: {aurora_status['hard_err_count']}")
print(f"Channel up: {aurora_status['channel_up']}")
```

**Solutions:**
1. Check fiber/cable connections
2. Clean optical connectors
3. Replace suspect SFP modules
4. Check for electrical interference

---

## Logging and Debugging

### Enable Verbose Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)

# ACCL-Q specific
logger = logging.getLogger('accl_quantum')
logger.setLevel(logging.DEBUG)

# Now operations will log details
accl.broadcast(data, root=0)
# DEBUG: Starting broadcast, root=0, size=64
# DEBUG: Tree depth=2, fanout=4
# DEBUG: Broadcast complete, latency=285.3ns
```

### Capture Operation History

```python
# Enable history capture
monitor = accl.get_monitor()
history = monitor.export_history()

# Save for analysis
import json
with open('accl_history.json', 'w') as f:
    json.dump(history, f, indent=2)
```

### Debug Mode

```python
# Enable debug assertions
import accl_quantum
accl_quantum.DEBUG = True

# Now additional checks are enabled
accl = ACCLQuantum(num_ranks=8, local_rank=0)
# Will raise AssertionError on invalid operations
```

### Remote Debugging

```python
# Connect debugger to specific board
import pdb
import socket

def remote_debug(board_ip, port=4444):
    """Connect pdb to remote board."""
    sock = socket.socket()
    sock.connect((board_ip, port))
    pdb.Pdb(stdin=sock.makefile('r'), stdout=sock.makefile('w')).set_trace()
```

---

## Getting Help

If you can't resolve your issue:

1. **Collect Diagnostics**
```python
diagnostics = {
    'config': accl.config.__dict__,
    'sync_status': accl.get_sync_status(),
    'latency_stats': accl.get_latency_stats(),
    'timing_validation': accl.validate_timing(),
}
```

2. **Include System Information**
```python
import platform
system_info = {
    'platform': platform.platform(),
    'python': platform.python_version(),
    'numpy': np.__version__,
}
```

3. **Report Issue**
- Include diagnostic output
- Describe steps to reproduce
- Attach relevant logs

---

## See Also

- [API Reference](api_reference.md) - Complete API documentation
- [Integration Guide](integration_guide.md) - Framework integration
- [Performance Tuning](performance_tuning.md) - Optimization guide
