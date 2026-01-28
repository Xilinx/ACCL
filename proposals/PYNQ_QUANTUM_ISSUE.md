# [RFC] PYNQ-Quantum: Native Quantum Computing Support for RFSoC

## Summary

We propose adding a `pynq.quantum` package to provide Python-native quantum computing support for RFSoC platforms. This would unify the fragmented quantum control ecosystem (QICK, QubiC, custom solutions) under PYNQ's overlay architecture.

## Motivation

RFSoC platforms have become the de facto standard for quantum control:

- **[QICK](https://github.com/openquantumhardware/qick)** (Fermilab) - 900+ stars, used by 100+ labs
- **[QubiC](https://github.com/lbnl-science-it/qubic)** (LBNL) - Production at AQT/LBNL
- **[SpinQICK](https://github.com/HRL-Laboratories/spinqick)** (HRL) - Spin qubit control

However, researchers face barriers:
1. No standard Python APIs for quantum control
2. Steep learning curve (Vivado, HLS expertise required)
3. Limited multi-board synchronization support
4. Each lab reinvents drivers and calibration tools

PYNQ's overlay system and Python-first approach could solve these problems.

## Proposed Features

### Core Package (`pynq.quantum`)

```python
from pynq.quantum import QuantumOverlay, QubitController

# Load overlay (auto-detects board)
qo = QuantumOverlay(backend='qick')

# Control qubits
ctrl = QubitController(qo, num_qubits=4)
ctrl.set_qubit_frequency(0, 5.123e9)
ctrl.x90(0)
ctrl.measure([0])
results = ctrl.run(shots=1000)
```

### Multi-Backend Support

| Backend | Firmware | Status |
|---------|----------|--------|
| QICK | Fermilab QICK | Proposed |
| QubiC | LBNL QubiC | Proposed |
| Generic | Custom HLS | Proposed |

### Multi-Board Synchronization (via [ACCL-Q](https://github.com/Xilinx/ACCL/pull/216))

```python
from pynq.quantum import QuantumCluster
from pynq.quantum.collective import allreduce

cluster = QuantumCluster(['192.168.1.10', '192.168.1.11'])
measurements = cluster.local_measure([0, 1, 2, 3])
syndrome = allreduce(measurements, op='XOR')  # <400ns latency
```

### Pre-built Overlays

- ZCU111 quantum base overlay
- ZCU216 quantum base overlay
- RFSoC4x2 quantum base overlay

## Questions for Discussion

1. **Scope:** Should this live in `RFSoC-PYNQ` or the main `PYNQ` repo?
2. **Backend priority:** Start with QICK, QubiC, or generic?
3. **Overlay distribution:** Ship pre-built bitstreams or build-from-source?
4. **Community interest:** Would QICK/QubiC maintainers collaborate?

## Full RFC

See the complete RFC with implementation phases, API design, and testing strategy:
📄 [PYNQ_QUANTUM_RFC.md](./PYNQ_QUANTUM_RFC.md)

## Related Work

- [ACCL-Q PR #216](https://github.com/Xilinx/ACCL/pull/216) - Quantum collective operations
- [strath-sdr/rfsoc_qpsk](https://github.com/strath-sdr/rfsoc_qpsk) - RFSoC signal processing example
- [PYNQ_RFSOC_Workshop](https://github.com/Xilinx/PYNQ_RFSOC_Workshop) - Existing RFSoC tutorials

## Call for Collaborators

We're seeking:
- PYNQ maintainers for architecture guidance
- QICK/QubiC developers for backend integration
- Quantum researchers for requirements and testing
- FPGA engineers for overlay optimization

---

**Signed-off-by:** ACCL-Q Team
