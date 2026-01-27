# ACCL-Q Integration Guide

This guide covers integration with QubiC (LBNL) and QICK (Fermilab) quantum control frameworks.

## Table of Contents

1. [Overview](#overview)
2. [QubiC Integration](#qubic-integration)
3. [QICK Integration](#qick-integration)
4. [Unified API](#unified-api)
5. [Measurement Feedback Pipeline](#measurement-feedback-pipeline)
6. [Best Practices](#best-practices)

---

## Overview

ACCL-Q provides native integration with two major quantum control frameworks:

- **QubiC** (Lawrence Berkeley National Laboratory): Instruction-based quantum control with compiler infrastructure
- **QICK** (Fermilab): tProcessor-based pulse sequencing for RFSoC platforms

Both integrations provide:
- Direct ACCL-Q operation mapping to framework primitives
- Automatic timing coordination
- Measurement feedback within coherence budgets

---

## QubiC Integration

### Setup

```python
from accl_quantum import ACCLQuantum
from accl_quantum.integrations import QubiCIntegration

# Initialize ACCL-Q
accl = ACCLQuantum(num_ranks=8, local_rank=rank_id)

# Create QubiC integration
qubic = QubiCIntegration(accl)
```

### Instruction Handlers

QubiC integration provides custom instructions for collective operations:

#### DIST_MEAS - Distribute Measurement

```python
# Register instruction handler
@qubic.instruction_handler('DIST_MEAS')
def handle_dist_meas(qubit_id, source_board):
    """Distribute measurement from source to all boards."""
    measurement = read_measurement_register(qubit_id)
    result = accl.distribute_measurement(measurement, source_board)
    return result.data

# Usage in QubiC program
program.add_instruction('DIST_MEAS', qubit=0, source=2)
```

#### SYNC_BARRIER - Synchronized Barrier

```python
@qubic.instruction_handler('SYNC_BARRIER')
def handle_sync_barrier():
    """Hardware-synchronized barrier."""
    result = accl.barrier()
    return result.success
```

#### XOR_SYNDROME - Syndrome Aggregation

```python
@qubic.instruction_handler('XOR_SYNDROME')
def handle_xor_syndrome(syndrome_bits):
    """Aggregate syndrome via XOR reduction."""
    local_syndrome = np.array(syndrome_bits, dtype=np.uint8)
    result = accl.aggregate_syndrome(local_syndrome)
    return result.data
```

### Measurement Callback Integration

```python
def measurement_callback(qubit_id: int, result: int, context: dict):
    """Called when measurement completes on this board."""
    # Get source board for this qubit
    source_board = context.get('source_board', accl.local_rank)

    # Distribute to all boards
    measurement = np.array([result], dtype=np.uint8)
    dist_result = accl.distribute_measurement(measurement, source_board)

    # Apply conditional operation based on measurement
    if dist_result.data[0] == 1:
        apply_correction(context['target_qubit'])

    return dist_result.latency_ns

# Register callback
qubic.register_measurement_callback(measurement_callback)
```

### Timing Integration

QubiC timing can be coordinated with ACCL-Q clock synchronization:

```python
# Synchronize ACCL-Q clocks
accl.sync_clocks()

# Get synchronized trigger time
trigger_time = accl.get_global_counter() + delay_cycles

# Schedule synchronized operations across all boards
accl.synchronized_trigger(trigger_time)

# QubiC operations will execute at the trigger
program.schedule_at_trigger(trigger_time)
```

### Complete QubiC Example

```python
from accl_quantum import ACCLQuantum, ACCLMode
from accl_quantum.integrations import QubiCIntegration
import numpy as np

# Setup
accl = ACCLQuantum(num_ranks=4, local_rank=0)
accl.configure(mode=ACCLMode.DETERMINISTIC)
accl.sync_clocks()

qubic = QubiCIntegration(accl)

# Define QEC cycle
def qec_cycle():
    # 1. Measure ancilla qubits (local)
    syndromes = []
    for ancilla in range(4):
        syndromes.append(qubic.measure(ancilla))

    local_syndrome = np.array(syndromes, dtype=np.uint8)

    # 2. Aggregate syndromes across all boards
    global_syndrome = accl.aggregate_syndrome(local_syndrome)

    # 3. Decode (at decoder board)
    if accl.local_rank == 0:
        corrections = decode_syndrome(global_syndrome.data)
        # 4. Distribute corrections
        accl.distribute_correction(corrections, decoder_rank=0)
    else:
        result = accl.scatter(None, root=0)
        apply_correction(result.data)

# Run QEC
for cycle in range(100):
    qec_cycle()
```

---

## QICK Integration

### Setup

```python
from accl_quantum import ACCLQuantum
from accl_quantum.integrations import QICKIntegration

# Initialize ACCL-Q
accl = ACCLQuantum(num_ranks=8, local_rank=rank_id)

# Create QICK integration with tProcessor reference
qick = QICKIntegration(accl, tproc=soc.tproc)
```

### tProcessor Extensions

QICK integration adds ACCL-Q operations as tProcessor instructions:

#### accl_broadcast

```python
# In tProcessor ASM
accl_broadcast r0, r1  # Broadcast r0 from rank r1
```

```python
# Python equivalent
@qick.tproc_instruction('accl_broadcast')
def accl_broadcast(data_reg, root_reg):
    data = tproc.read_reg(data_reg)
    root = tproc.read_reg(root_reg)
    result = accl.broadcast(np.array([data]), root)
    tproc.write_reg(data_reg, result.data[0])
```

#### accl_xor_reduce

```python
# In tProcessor ASM
accl_xor_reduce r0  # XOR reduce r0 across all ranks
```

```python
@qick.tproc_instruction('accl_xor_reduce')
def accl_xor_reduce(data_reg):
    data = tproc.read_reg(data_reg)
    result = accl.allreduce(np.array([data]), ReduceOp.XOR)
    tproc.write_reg(data_reg, result.data[0])
```

#### accl_barrier

```python
# In tProcessor ASM
accl_barrier  # Synchronized barrier
```

```python
@qick.tproc_instruction('accl_barrier')
def accl_barrier():
    accl.barrier()
```

### RAveragerProgram Integration

```python
from qick import RAveragerProgram

class ACCLAveragerProgram(RAveragerProgram):
    """RAveragerProgram with ACCL-Q collective operations."""

    def __init__(self, soccfg, cfg, accl):
        super().__init__(soccfg, cfg)
        self.accl = accl
        self.qick_int = QICKIntegration(accl, self.tproc)

    def body(self):
        # Standard QICK operations
        self.pulse(ch=self.cfg['qubit_ch'], name='X90')
        self.sync_all()

        # Measure
        self.measure(pulse_ch=self.cfg['res_ch'],
                    adcs=[self.cfg['adc_ch']],
                    adc_trig_offset=self.cfg['adc_trig_offset'],
                    wait=True)

        # Distribute measurement via ACCL-Q
        self.qick_int.sync_and_distribute_measurement(
            source_rank=self.accl.local_rank
        )

        # Apply conditional correction
        self.qick_int.conditional_pulse_if_one(
            ch=self.cfg['qubit_ch'],
            name='Z'
        )
```

### Pulse Timing Coordination

```python
# Coordinate pulse timing with ACCL-Q sync
def synchronized_pulse_sequence(qick_int, pulse_times):
    """Execute pulses at synchronized times across boards."""

    # Sync ACCL-Q clocks
    qick_int.accl.sync_clocks()

    # Get common reference time
    ref_time = qick_int.accl.get_global_counter()

    for pulse_time, pulse_config in pulse_times:
        # Calculate absolute trigger time
        trigger = ref_time + pulse_time

        # Schedule synchronized trigger
        qick_int.accl.synchronized_trigger(trigger)

        # Program pulse at trigger
        qick_int.program_pulse_at_trigger(trigger, pulse_config)
```

### Complete QICK Example

```python
from accl_quantum import ACCLQuantum, ACCLMode
from accl_quantum.integrations import QICKIntegration
from qick import QickSoc
import numpy as np

# Initialize hardware
soc = QickSoc()

# Initialize ACCL-Q
accl = ACCLQuantum(num_ranks=4, local_rank=0)
accl.configure(mode=ACCLMode.DETERMINISTIC)

# Create QICK integration
qick = QICKIntegration(accl, tproc=soc.tproc)

# Teleportation protocol
def teleportation():
    # 1. Alice prepares state and measures
    soc.tproc.pulse(ch=0, name='H')  # Hadamard
    soc.tproc.pulse(ch=0, name='CNOT', target=1)  # Entangle

    # 2. Alice measures qubits 0 and 1
    m0 = soc.tproc.measure(ch=0)
    m1 = soc.tproc.measure(ch=1)

    # 3. Distribute measurements via ACCL-Q
    measurements = np.array([m0, m1], dtype=np.uint8)
    result = accl.broadcast(measurements, root=0)

    # 4. Bob applies corrections based on measurements
    if accl.local_rank == 1:  # Bob's board
        m0, m1 = result.data
        if m1 == 1:
            soc.tproc.pulse(ch=2, name='X')
        if m0 == 1:
            soc.tproc.pulse(ch=2, name='Z')

teleportation()
```

---

## Unified API

For framework-agnostic code, use `UnifiedQuantumControl`:

```python
from accl_quantum.integrations import UnifiedQuantumControl

# Create unified controller
controller = UnifiedQuantumControl(accl, backend='qubic')
# or
controller = UnifiedQuantumControl(accl, backend='qick', tproc=soc.tproc)

# Framework-agnostic operations
controller.sync_clocks()
controller.barrier()
controller.distribute_measurement(measurement, source=0)
controller.aggregate_syndrome(syndrome)

# Get backend-specific interface if needed
if controller.backend == 'qubic':
    qubic = controller.get_integration()
    qubic.custom_instruction(...)
```

---

## Measurement Feedback Pipeline

### MeasurementFeedbackPipeline

Provides end-to-end feedback with timing guarantees:

```python
from accl_quantum.feedback import MeasurementFeedbackPipeline

# Create pipeline
pipeline = MeasurementFeedbackPipeline(accl, latency_budget_ns=500)

# Single-qubit feedback
async def feedback_x_if_one(measurement, target_qubit):
    result = await pipeline.single_qubit_feedback(
        measurement=measurement,
        source_rank=0,
        target_rank=1,
        correction_fn=lambda m: 'X' if m == 1 else 'I'
    )
    return result

# Parity-based feedback
async def parity_feedback(measurements, target_qubit):
    result = await pipeline.parity_feedback(
        measurements=measurements,
        sources=[0, 1, 2],
        target_rank=3,
        correction_fn=lambda parity: 'Z' if parity == 1 else 'I'
    )
    return result

# Full syndrome feedback
async def qec_feedback(syndromes):
    result = await pipeline.syndrome_feedback(
        syndromes=syndromes,
        decoder_rank=0,
        decoder_fn=minimum_weight_decoder
    )
    return result
```

### FeedbackScheduler

Schedule feedback operations within timing budget:

```python
from accl_quantum.feedback import FeedbackScheduler

scheduler = FeedbackScheduler(accl, coherence_time_us=50)

# Schedule feedback with deadline
scheduler.schedule(
    feedback_operation,
    deadline_ns=400,  # Must complete within 400ns
    priority=1
)

# Run scheduled operations
scheduler.run()

# Check if deadlines were met
stats = scheduler.get_timing_stats()
print(f"On-time: {stats['on_time_percent']}%")
```

---

## Best Practices

### 1. Initialize Early

```python
# Initialize ACCL-Q before quantum operations
accl = ACCLQuantum(num_ranks=8, local_rank=rank_id)
accl.configure(mode=ACCLMode.DETERMINISTIC)
accl.sync_clocks()  # Sync before any timed operations
```

### 2. Monitor Latency

```python
# Enable monitoring
config = ACCLConfig(
    num_ranks=8,
    local_rank=0,
    enable_latency_monitoring=True
)
accl = ACCLQuantum(config=config)

# Check after operations
stats = accl.get_latency_stats()
validation = accl.validate_timing()
if not all(v['overall_pass'] for v in validation.values()):
    print("Warning: Timing targets not met")
```

### 3. Use Appropriate Sync Mode

```python
# For measurement feedback (strict timing)
accl.broadcast(data, root=0, sync=SyncMode.HARDWARE)

# For non-critical operations (lower overhead)
accl.broadcast(data, root=0, sync=SyncMode.SOFTWARE)
```

### 4. Pre-allocate Buffers

```python
# Pre-allocate receive buffers
recv_buffer = np.zeros(syndrome_size, dtype=np.uint8)

# Reuse for multiple operations
for cycle in range(num_cycles):
    result = accl.aggregate_syndrome(local_syndrome)
    np.copyto(recv_buffer, result.data)
```

### 5. Handle Errors

```python
result = accl.broadcast(data, root=0)
if not result.success:
    if result.status == OperationStatus.TIMEOUT:
        # Re-sync clocks and retry
        accl.sync_clocks()
        result = accl.broadcast(data, root=0)
    else:
        raise RuntimeError(f"ACCL-Q error: {result.status}")
```

---

## See Also

- [API Reference](api_reference.md) - Complete API documentation
- [Performance Tuning](performance_tuning.md) - Optimization guide
- [Troubleshooting](troubleshooting.md) - Common issues
