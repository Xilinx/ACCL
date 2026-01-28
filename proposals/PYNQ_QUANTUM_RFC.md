# RFC: PYNQ-Quantum - Quantum Computing Support for RFSoC Platforms

**Author:** ACCL-Q Team
**Status:** Draft
**Created:** 2026-01-27
**Target Repository:** [Xilinx/RFSoC-PYNQ](https://github.com/Xilinx/RFSoC-PYNQ)

---

## Executive Summary

This RFC proposes adding native quantum computing support to PYNQ for RFSoC platforms. The goal is to provide Python-native APIs for qubit control, measurement feedback, and multi-board synchronization—enabling researchers to develop quantum control systems with the same ease that PYNQ brings to traditional FPGA development.

### Key Deliverables

| Component | Description |
|-----------|-------------|
| `pynq.quantum` | Core Python package for quantum control |
| Quantum Base Overlay | Pre-built bitstreams for ZCU111/ZCU216/RFSoC4x2 |
| QICK Integration | Native support for Fermilab's QICK firmware |
| QubiC Integration | Support for LBNL's QubiC control system |
| ACCL-Q Collective Ops | Sub-microsecond multi-board communication |
| Jupyter Notebooks | Interactive tutorials and examples |

---

## Motivation

### The Problem

Quantum computing researchers using Xilinx RFSoC face significant barriers:

1. **Fragmented Ecosystem**: QICK, QubiC, and custom solutions exist independently
2. **Steep Learning Curve**: Requires Vivado, HLS, and low-level driver expertise
3. **No Standard APIs**: Each lab develops proprietary control software
4. **Limited Multi-Board Support**: Distributed quantum systems need synchronized FPGAs

### The Opportunity

RFSoC platforms are becoming the standard for quantum control:

- **[QICK](https://github.com/openquantumhardware/qick)** (Fermilab) - 900+ GitHub stars, 100+ labs worldwide
- **[QubiC](https://arxiv.org/abs/2303.03816)** (LBNL) - Production use at AQT/LBNL
- **[SpinQICK](https://github.com/HRL-Laboratories/spinqick)** (HRL) - Spin qubit extension
- **Academic Adoption** - Stanford, MIT, IBM, Google using RFSoC for control

### Why PYNQ?

PYNQ's mission—"Python Productivity for Zynq"—aligns perfectly with quantum computing needs:

| PYNQ Strength | Quantum Application |
|---------------|---------------------|
| Python-native APIs | Intuitive qubit control |
| Overlay system | Swappable quantum firmware |
| Jupyter integration | Interactive calibration |
| Driver abstractions | Hardware-agnostic control |
| Community ecosystem | Shared quantum overlays |

---

## Technical Architecture

### Package Structure

```
pynq/
├── quantum/
│   ├── __init__.py           # Public API exports
│   ├── core.py               # QuantumOverlay base class
│   ├── control.py            # Qubit control primitives
│   ├── measurement.py        # Readout and feedback
│   ├── timing.py             # Clock synchronization
│   ├── collective.py         # Multi-board operations (ACCL-Q)
│   ├── calibration.py        # Auto-calibration routines
│   │
│   ├── backends/
│   │   ├── qick.py           # QICK firmware backend
│   │   ├── qubic.py          # QubiC firmware backend
│   │   └── generic.py        # Custom firmware interface
│   │
│   ├── pulses/
│   │   ├── library.py        # Standard pulse shapes
│   │   ├── compiler.py       # Pulse sequence compiler
│   │   └── optimizer.py      # Gate optimization
│   │
│   └── qec/
│       ├── syndrome.py       # Syndrome extraction
│       ├── decoders.py       # Error decoders
│       └── feedback.py       # Real-time correction
│
boards/
├── ZCU111/
│   └── quantum/
│       ├── quantum.bit       # Pre-built bitstream
│       ├── quantum.hwh       # Hardware handoff
│       └── quantum.xsa       # Exported hardware
├── ZCU216/
│   └── quantum/
│       └── ...
└── RFSoC4x2/
    └── quantum/
        └── ...
```

### Class Hierarchy

```
pynq.Overlay
    └── pynq.quantum.QuantumOverlay
            ├── pynq.quantum.QICKOverlay      # QICK-compatible
            ├── pynq.quantum.QubiCOverlay     # QubiC-compatible
            └── pynq.quantum.GenericOverlay   # Custom firmware
```

### Core APIs

#### 1. Overlay Initialization

```python
from pynq.quantum import QuantumOverlay

# Load quantum overlay (auto-detects board)
qo = QuantumOverlay()

# Or specify backend explicitly
qo = QuantumOverlay(backend='qick', bitfile='custom.bit')

# Access hardware info
print(f"Board: {qo.board}")
print(f"DACs: {qo.num_dacs}, ADCs: {qo.num_adcs}")
print(f"Qubits configured: {qo.num_qubits}")
```

#### 2. Qubit Control

```python
from pynq.quantum import QubitController
from pynq.quantum.pulses import GaussianPulse, DRAGPulse

# Initialize controller
ctrl = QubitController(qo, num_qubits=4)

# Configure qubit frequencies
ctrl.set_qubit_frequency(0, 5.123e9)  # Hz
ctrl.set_readout_frequency(0, 7.456e9)

# Define pulses
x90 = GaussianPulse(duration=20e-9, sigma=5e-9, amplitude=0.5)
x180 = DRAGPulse(duration=40e-9, sigma=10e-9, amplitude=1.0, drag_coef=0.5)

# Execute gate sequence
ctrl.pulse(0, x90)           # X90 on qubit 0
ctrl.pulse(1, x180)          # X180 on qubit 1
ctrl.cz(0, 1)                # CZ gate
ctrl.measure([0, 1])         # Measure both
results = ctrl.run(shots=1000)
```

#### 3. Measurement Feedback

```python
from pynq.quantum import FeedbackController
from pynq.quantum.qec import SyndromeDecoder

# Real-time feedback (sub-microsecond)
fb = FeedbackController(qo, latency_budget_ns=500)

# Simple conditional
fb.measure_and_apply(
    qubit=0,
    condition=lambda m: m == 1,
    action=lambda: ctrl.pulse(1, x180)
)

# QEC syndrome feedback
decoder = SyndromeDecoder(code='surface_17')
fb.syndrome_feedback(
    ancilla_qubits=[4, 5, 6, 7],
    decoder=decoder,
    correction_map={...}
)
```

#### 4. Multi-Board Synchronization (ACCL-Q Integration)

```python
from pynq.quantum import QuantumCluster
from pynq.quantum.collective import broadcast, allreduce

# Create synchronized cluster
cluster = QuantumCluster(
    boards=['192.168.1.10', '192.168.1.11', '192.168.1.12'],
    sync_method='hardware'  # Sub-nanosecond sync
)

# Verify synchronization
status = cluster.sync_status()
assert status['phase_error_ns'] < 1.0

# Distributed operations
measurements = cluster.local_measure([0, 1, 2, 3])
global_syndrome = allreduce(measurements, op='XOR')  # <400ns

# Broadcast correction
correction = decoder.decode(global_syndrome)
broadcast(correction, root=0)  # <300ns
```

#### 5. Calibration Tools

```python
from pynq.quantum.calibration import AutoCalibrator

cal = AutoCalibrator(ctrl)

# Run calibration routines
cal.find_qubit_frequency(0, search_range=(5.0e9, 5.5e9))
cal.calibrate_pi_pulse(0)
cal.calibrate_readout(0)
cal.measure_t1(0)
cal.measure_t2_ramsey(0)
cal.measure_t2_echo(0)

# Save calibration
cal.save('calibration_2026_01_27.json')
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (8 weeks)

| Task | Description | Deliverable |
|------|-------------|-------------|
| Package scaffold | Create `pynq.quantum` package structure | Python package |
| QuantumOverlay base | Extend `pynq.Overlay` for quantum | `core.py` |
| Hardware detection | Auto-detect RFSoC board and capabilities | Board configs |
| Basic drivers | RF-DAC/ADC control via existing xrfdc | Driver wrappers |
| Unit tests | pytest suite with simulation backend | Test framework |

### Phase 2: QICK Integration (6 weeks)

| Task | Description | Deliverable |
|------|-------------|-------------|
| QICK backend | Wrap QICK firmware and drivers | `backends/qick.py` |
| Pulse compiler | Translate pulses to QICK format | `pulses/compiler.py` |
| tProcessor interface | Program execution and readout | Control interface |
| Loopback tests | Validate DAC→ADC signal path | Integration tests |
| QICK examples | Jupyter notebooks from QICK demos | Notebooks |

### Phase 3: Measurement & Feedback (6 weeks)

| Task | Description | Deliverable |
|------|-------------|-------------|
| Readout pipeline | IQ demodulation, thresholding | `measurement.py` |
| Feedback controller | Real-time conditional operations | `measurement.py` |
| Latency profiling | Measure and optimize feedback latency | Profiler tools |
| Syndrome extraction | Multi-qubit parity measurements | `qec/syndrome.py` |
| Decoder interface | Pluggable decoder backends | `qec/decoders.py` |

### Phase 4: Multi-Board / ACCL-Q (8 weeks)

| Task | Description | Deliverable |
|------|-------------|-------------|
| Clock synchronization | Hardware-level multi-board sync | `timing.py` |
| ACCL-Q integration | Import from accl-quantum package | `collective.py` |
| Collective operations | broadcast, reduce, allreduce, barrier | Collective APIs |
| Distributed QEC | Multi-node syndrome aggregation | QEC examples |
| Cluster management | Board discovery, health monitoring | `QuantumCluster` |

### Phase 5: Documentation & Community (4 weeks)

| Task | Description | Deliverable |
|------|-------------|-------------|
| API documentation | Sphinx autodoc for all modules | docs.pynq.io |
| Tutorial notebooks | Step-by-step quantum control guides | Jupyter notebooks |
| Example gallery | Common use cases and patterns | Examples repo |
| Video tutorials | YouTube walkthrough series | Video content |
| Community outreach | QICK/QubiC community engagement | Forum posts |

---

## Hardware Requirements

### Supported Boards

| Board | Status | DACs | ADCs | Max Qubits* |
|-------|--------|------|------|-------------|
| ZCU111 | Primary | 8 | 8 | 8 |
| ZCU216 | Primary | 16 | 16 | 16 |
| RFSoC4x2 | Primary | 2 | 4 | 4 |
| ZCU208 | Planned | 8 | 8 | 8 |

*Assumes 1 DAC + 1 ADC per qubit for control + readout

### Minimum Firmware Resources

| Resource | Requirement |
|----------|-------------|
| LUTs | ~50,000 (base overlay) |
| BRAMs | ~100 (pulse memory) |
| DSP48s | ~200 (NCOs, mixers) |
| PL Clock | 500 MHz |
| PS-PL Interface | AXI4 @ 256-bit |

---

## Compatibility Matrix

### Framework Interoperability

| Framework | Integration Level | Notes |
|-----------|-------------------|-------|
| [QICK](https://github.com/openquantumhardware/qick) | Native backend | Full API compatibility |
| [QubiC](https://github.com/lbnl-science-it/qubic) | Native backend | Requires QubiC firmware |
| [Qiskit](https://qiskit.org/) | Provider plugin | `qiskit-pynq-provider` |
| [Cirq](https://quantumai.google/cirq) | Sampler backend | `cirq-pynq` |
| [ACCL](https://github.com/Xilinx/ACCL) | Collective ops | Via `accl-quantum` package |
| [OpenPulse](https://arxiv.org/abs/1809.03452) | Pulse format | Import/export support |

### Python Version Support

- Python 3.8+ (matching PYNQ requirements)
- NumPy 1.20+
- Tested on PYNQ v3.0, v3.1

---

## Testing Strategy

### Test Levels

```
┌─────────────────────────────────────────────────────┐
│                  Hardware Tests                      │
│    (Requires physical RFSoC board)                  │
├─────────────────────────────────────────────────────┤
│              Integration Tests                       │
│    (Simulation backend + emulated hardware)         │
├─────────────────────────────────────────────────────┤
│                 Unit Tests                          │
│    (Pure Python, no hardware)                       │
└─────────────────────────────────────────────────────┘
```

### Test Coverage Targets

| Module | Unit | Integration | Hardware |
|--------|------|-------------|----------|
| `core.py` | 90% | 80% | 70% |
| `control.py` | 85% | 75% | 60% |
| `measurement.py` | 85% | 70% | 50% |
| `collective.py` | 90% | 80% | 40% |
| `backends/*` | 80% | 70% | 60% |

### CI/CD Pipeline

```yaml
# .github/workflows/quantum-tests.yml
- Unit tests: Every PR (no hardware)
- Integration tests: Nightly (simulation)
- Hardware tests: Weekly (ZCU111 in CI farm)
```

---

## Performance Targets

### Latency Requirements

| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Single pulse | <100 ns | Oscilloscope |
| Readout + threshold | <500 ns | Loopback test |
| Feedback decision | <200 ns | Internal counter |
| Broadcast (8 nodes) | <300 ns | ACCL-Q monitor |
| Allreduce (8 nodes) | <400 ns | ACCL-Q monitor |

### Jitter Requirements

| Operation | Max Jitter | Notes |
|-----------|------------|-------|
| Pulse timing | <2 ns | Critical for gates |
| Multi-board sync | <1 ns | Phase-locked |
| Feedback trigger | <10 ns | QEC compatible |

---

## Security Considerations

### Network Security

- Multi-board communication over isolated network
- Optional TLS for remote Jupyter access
- No credential storage in notebooks

### Firmware Integrity

- Bitstream signature verification (when available)
- Checksum validation for downloaded overlays

---

## Community Engagement Plan

### Target Communities

1. **QICK Users** - Fermilab mailing list, GitHub discussions
2. **QubiC Users** - LBNL quantum computing group
3. **PYNQ Community** - discuss.pynq.io forum
4. **Academic Labs** - arXiv announcements, conference workshops
5. **Industry** - IBM, Google, IonQ, Rigetti (potential adopters)

### Outreach Activities

| Activity | Timeline | Audience |
|----------|----------|----------|
| RFC announcement | Week 1 | PYNQ forum |
| QICK community RFC | Week 2 | QICK GitHub |
| APS March Meeting poster | March 2026 | Physicists |
| Xilinx Developer Forum talk | Q2 2026 | FPGA developers |
| Tutorial workshop | Q3 2026 | New users |

---

## Alternatives Considered

### Alternative 1: Standalone Package (Not in PYNQ)

**Pros:** Faster iteration, independent releases
**Cons:** No overlay integration, duplicate driver code, fragmented ecosystem

**Decision:** Rejected. PYNQ integration provides overlay management and driver reuse.

### Alternative 2: QICK-Only Support

**Pros:** Simpler implementation, proven firmware
**Cons:** Excludes QubiC users, limits flexibility

**Decision:** Rejected. Multi-backend support enables broader adoption.

### Alternative 3: Kernel-Space Implementation

**Pros:** Lower latency potential
**Cons:** Complex development, limited Python integration

**Decision:** Rejected. User-space with MMIO achieves required latency (<500 ns).

---

## Dependencies

### Required Packages

```
pynq >= 3.0
numpy >= 1.20
scipy >= 1.7  # For signal processing
accl-quantum >= 0.2.0  # For collective operations
```

### Optional Packages

```
qick >= 0.2  # For QICK backend
qiskit >= 0.45  # For Qiskit integration
matplotlib >= 3.5  # For visualization
```

---

## Appendix A: Example Notebooks

### Notebook 1: Getting Started

```python
# 01_getting_started.ipynb
"""
PYNQ-Quantum: Your First Qubit Control
=======================================
This notebook walks through:
1. Loading the quantum overlay
2. Configuring a qubit
3. Running a simple experiment
4. Visualizing results
"""
```

### Notebook 2: Rabi Oscillation

```python
# 02_rabi_oscillation.ipynb
"""
Measuring Rabi Oscillations
===========================
Calibrate pulse amplitude by sweeping drive power
and measuring excited state population.
"""
```

### Notebook 3: T1/T2 Characterization

```python
# 03_coherence_times.ipynb
"""
Qubit Coherence Measurements
============================
- T1 (energy relaxation)
- T2* (Ramsey dephasing)
- T2 (Echo dephasing)
"""
```

### Notebook 4: Multi-Board QEC

```python
# 04_distributed_qec.ipynb
"""
Distributed Quantum Error Correction
====================================
Using ACCL-Q for multi-board syndrome aggregation
with sub-microsecond feedback.
"""
```

---

## Appendix B: Comparison with Existing Solutions

| Feature | PYNQ-Quantum | QICK | QubiC | Qiskit-Metal |
|---------|--------------|------|-------|--------------|
| Python-native | Yes | Yes | Yes | Yes |
| Multi-backend | Yes | No | No | No |
| Multi-board sync | Yes (ACCL-Q) | Limited | Limited | No |
| Sub-μs feedback | Yes | Yes | Yes | No |
| Overlay management | Yes (PYNQ) | Manual | Manual | N/A |
| Qiskit integration | Yes | Community | No | Native |
| Open source | BSD-3 | BSD-3 | Apache-2 | Apache-2 |

---

## References

1. [QICK: Quantum Instrumentation Control Kit](https://github.com/openquantumhardware/qick)
2. [QubiC: Quantum Control System](https://arxiv.org/abs/2303.03816)
3. [PYNQ: Python Productivity for Zynq](https://github.com/Xilinx/PYNQ)
4. [RFSoC-PYNQ](https://github.com/Xilinx/RFSoC-PYNQ)
5. [ACCL: Accelerated Collective Communication Library](https://github.com/Xilinx/ACCL)
6. [ACCL-Q: Quantum-Optimized ACCL](https://github.com/Xilinx/ACCL/pull/216)
7. [SpinQICK: Spin Qubit Control](https://github.com/HRL-Laboratories/spinqick)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-01-27 | Initial RFC draft |

---

## Feedback

Please provide feedback via:

- **GitHub Issue:** [Xilinx/RFSoC-PYNQ/issues](https://github.com/Xilinx/RFSoC-PYNQ/issues)
- **PYNQ Forum:** [discuss.pynq.io](https://discuss.pynq.io)
- **Email:** [quantum-rfc@example.com]

---

*This RFC is submitted under BSD-3-Clause license, consistent with PYNQ licensing.*

Signed-off-by: ACCL-Q Team <accl-q@example.com>
