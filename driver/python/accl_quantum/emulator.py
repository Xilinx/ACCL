"""
ACCL-Q Realistic Qubit Emulator

Provides comprehensive qubit emulation with realistic noise models
for thorough validation testing of quantum control operations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import time
import threading
from collections import deque


class GateType(Enum):
    """Quantum gate types."""
    I = "I"      # Identity
    X = "X"      # Pauli-X (NOT)
    Y = "Y"      # Pauli-Y
    Z = "Z"      # Pauli-Z
    H = "H"      # Hadamard
    S = "S"      # Phase gate
    T = "T"      # T gate
    RX = "RX"    # Rotation around X
    RY = "RY"    # Rotation around Y
    RZ = "RZ"    # Rotation around Z
    CNOT = "CNOT"    # Controlled-NOT
    CZ = "CZ"        # Controlled-Z
    SWAP = "SWAP"    # SWAP gate
    MEASURE = "MEASURE"


@dataclass
class NoiseParameters:
    """
    Comprehensive noise model parameters for qubit simulation.

    Based on typical superconducting qubit characteristics.
    """
    # Coherence times (microseconds)
    t1_us: float = 50.0          # Energy relaxation time
    t2_us: float = 70.0          # Dephasing time (T2 <= 2*T1)
    t2_echo_us: float = 90.0     # T2 with echo (T2* < T2_echo)

    # Gate errors
    single_qubit_gate_error: float = 0.001      # 0.1% single-qubit gate error
    two_qubit_gate_error: float = 0.01          # 1% two-qubit gate error

    # Gate times (nanoseconds)
    single_qubit_gate_time_ns: float = 25.0     # Single-qubit gate duration
    two_qubit_gate_time_ns: float = 200.0       # Two-qubit gate duration

    # Measurement
    measurement_time_ns: float = 500.0          # Measurement duration
    readout_error_0: float = 0.02               # P(1|0) - false positive
    readout_error_1: float = 0.05               # P(0|1) - false negative

    # Crosstalk
    crosstalk_strength: float = 0.02            # Crosstalk coefficient
    crosstalk_range: int = 2                    # Crosstalk affects this many neighbors

    # Leakage
    leakage_rate: float = 0.001                 # Rate of leakage to non-computational states

    # Thermal
    thermal_population: float = 0.01            # Residual excited state population

    # Frequency
    qubit_frequency_ghz: float = 5.0            # Qubit transition frequency
    frequency_drift_mhz_per_hour: float = 0.1   # Frequency drift rate

    def validate(self) -> List[str]:
        """Validate parameters are physically reasonable."""
        errors = []

        if self.t2_us > 2 * self.t1_us:
            errors.append(f"T2 ({self.t2_us}us) cannot exceed 2*T1 ({2*self.t1_us}us)")

        if not 0 <= self.single_qubit_gate_error <= 1:
            errors.append("Single-qubit gate error must be in [0, 1]")

        if not 0 <= self.two_qubit_gate_error <= 1:
            errors.append("Two-qubit gate error must be in [0, 1]")

        if not 0 <= self.readout_error_0 <= 1:
            errors.append("Readout error P(1|0) must be in [0, 1]")

        if not 0 <= self.readout_error_1 <= 1:
            errors.append("Readout error P(0|1) must be in [0, 1]")

        return errors


@dataclass
class QubitState:
    """
    State of a single qubit with noise tracking.

    Uses density matrix representation for mixed states.
    """
    # Density matrix (2x2 complex)
    rho: np.ndarray = field(default_factory=lambda: np.array([[1, 0], [0, 0]], dtype=complex))

    # Time tracking for decoherence
    last_operation_time_ns: int = 0
    creation_time_ns: int = 0

    # Accumulated errors
    accumulated_error: float = 0.0
    gate_count: int = 0

    # Leakage tracking (probability in non-computational subspace)
    leakage_population: float = 0.0

    @property
    def population_0(self) -> float:
        """Ground state population."""
        return float(np.real(self.rho[0, 0]))

    @property
    def population_1(self) -> float:
        """Excited state population."""
        return float(np.real(self.rho[1, 1]))

    @property
    def coherence(self) -> float:
        """Off-diagonal coherence magnitude."""
        return float(np.abs(self.rho[0, 1]))

    @property
    def purity(self) -> float:
        """State purity: Tr(rho^2)."""
        return float(np.real(np.trace(self.rho @ self.rho)))

    def bloch_vector(self) -> Tuple[float, float, float]:
        """Get Bloch sphere coordinates (x, y, z)."""
        x = 2 * np.real(self.rho[0, 1])
        y = 2 * np.imag(self.rho[0, 1])
        z = np.real(self.rho[0, 0] - self.rho[1, 1])
        return (float(x), float(y), float(z))

    def reset(self) -> None:
        """Reset to ground state."""
        self.rho = np.array([[1, 0], [0, 0]], dtype=complex)
        self.accumulated_error = 0.0
        self.gate_count = 0
        self.leakage_population = 0.0


class RealisticQubitEmulator:
    """
    High-fidelity qubit emulator with comprehensive noise modeling.

    Features:
    - T1/T2 decoherence with continuous evolution
    - Gate errors with depolarizing noise
    - Measurement errors (readout fidelity)
    - Crosstalk between neighboring qubits
    - Leakage to non-computational states
    - Thermal excitation
    - Frequency drift

    Example:
        emulator = RealisticQubitEmulator(num_qubits=8)
        emulator.apply_gate(0, GateType.H)
        emulator.apply_gate([0, 1], GateType.CNOT)
        result = emulator.measure(0)
    """

    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Common gates
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    def __init__(self, num_qubits: int,
                 noise_params: Optional[NoiseParameters] = None,
                 seed: Optional[int] = None):
        """
        Initialize qubit emulator.

        Args:
            num_qubits: Number of qubits to simulate
            noise_params: Noise model parameters
            seed: Random seed for reproducibility
        """
        self.num_qubits = num_qubits
        self.noise = noise_params or NoiseParameters()

        # Validate noise parameters
        errors = self.noise.validate()
        if errors:
            raise ValueError(f"Invalid noise parameters: {errors}")

        # Initialize RNG
        self._rng = np.random.default_rng(seed)

        # Initialize qubit states
        self._states: Dict[int, QubitState] = {}
        self._init_time_ns = time.perf_counter_ns()

        for i in range(num_qubits):
            self._states[i] = QubitState(
                creation_time_ns=self._init_time_ns,
                last_operation_time_ns=self._init_time_ns
            )

        # Crosstalk matrix
        self._crosstalk_matrix = self._build_crosstalk_matrix()

        # Operation history for debugging
        self._history: deque = deque(maxlen=1000)

        # Statistics
        self._stats = {
            'total_gates': 0,
            'total_measurements': 0,
            'decoherence_events': 0,
            'leakage_events': 0,
            'crosstalk_events': 0,
        }

        # Thread safety
        self._lock = threading.RLock()

    def _build_crosstalk_matrix(self) -> np.ndarray:
        """Build crosstalk coupling matrix."""
        n = self.num_qubits
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = abs(i - j)
                    if distance <= self.noise.crosstalk_range:
                        # Crosstalk decays with distance
                        matrix[i, j] = self.noise.crosstalk_strength / distance

        return matrix

    def _current_time_ns(self) -> int:
        """Get current simulation time."""
        return time.perf_counter_ns()

    def _apply_decoherence(self, qubit: int) -> None:
        """
        Apply T1/T2 decoherence to qubit based on elapsed time.

        T1 decay: |1> -> |0> with rate 1/T1
        T2 decay: Coherence decay with rate 1/T2
        """
        state = self._states[qubit]
        current_time = self._current_time_ns()

        # Calculate elapsed time in microseconds
        elapsed_ns = current_time - state.last_operation_time_ns
        elapsed_us = elapsed_ns / 1000.0

        if elapsed_us <= 0:
            return

        # T1 decay (amplitude damping)
        gamma1 = 1.0 - np.exp(-elapsed_us / self.noise.t1_us)

        # T2 decay (phase damping) - T2* from dephasing
        gamma2 = 1.0 - np.exp(-elapsed_us / self.noise.t2_us)

        # Apply amplitude damping (T1)
        # Kraus operators: K0 = [[1, 0], [0, sqrt(1-gamma)]], K1 = [[0, sqrt(gamma)], [0, 0]]
        if gamma1 > 0:
            p1 = state.population_1
            decay_prob = p1 * gamma1

            # Update populations
            state.rho[0, 0] += decay_prob
            state.rho[1, 1] -= decay_prob

            # Update coherence
            coherence_factor = np.sqrt(1 - gamma1)
            state.rho[0, 1] *= coherence_factor
            state.rho[1, 0] *= coherence_factor

            if self._rng.random() < decay_prob:
                self._stats['decoherence_events'] += 1

        # Apply phase damping (T2 beyond T1 contribution)
        if gamma2 > gamma1 / 2:  # T2 contribution beyond T1
            phase_decay = np.exp(-elapsed_us / self.noise.t2_us)
            state.rho[0, 1] *= phase_decay
            state.rho[1, 0] *= phase_decay

        # Apply thermal excitation
        if self.noise.thermal_population > 0 and state.population_0 > 0:
            thermal_excitation = state.population_0 * self.noise.thermal_population * gamma1
            state.rho[0, 0] -= thermal_excitation
            state.rho[1, 1] += thermal_excitation

        state.last_operation_time_ns = current_time

    def _apply_gate_error(self, qubit: int, gate_error: float) -> None:
        """
        Apply depolarizing noise after gate.

        Depolarizing channel: rho -> (1-p)*rho + p*I/2
        """
        if gate_error <= 0:
            return

        state = self._states[qubit]

        # Depolarizing channel
        if self._rng.random() < gate_error:
            # Apply random Pauli error
            error_type = self._rng.choice(['X', 'Y', 'Z'])
            if error_type == 'X':
                state.rho = self.X @ state.rho @ self.X
            elif error_type == 'Y':
                state.rho = self.Y @ state.rho @ self.Y
            else:
                state.rho = self.Z @ state.rho @ self.Z

            state.accumulated_error += gate_error

    def _apply_crosstalk(self, target_qubit: int) -> None:
        """Apply crosstalk effects from target qubit to neighbors."""
        if self.noise.crosstalk_strength <= 0:
            return

        for neighbor in range(self.num_qubits):
            coupling = self._crosstalk_matrix[target_qubit, neighbor]
            if coupling > 0 and self._rng.random() < coupling:
                # Small Z rotation on neighbor
                angle = self._rng.normal(0, 0.01)  # Small random rotation
                self._apply_rz(neighbor, angle, apply_noise=False)
                self._stats['crosstalk_events'] += 1

    def _apply_leakage(self, qubit: int) -> None:
        """Apply leakage to non-computational states."""
        if self.noise.leakage_rate <= 0:
            return

        state = self._states[qubit]

        if self._rng.random() < self.noise.leakage_rate:
            # Transfer some population to leakage
            leaked = state.population_1 * self.noise.leakage_rate
            state.rho[1, 1] -= leaked
            state.leakage_population += leaked
            self._stats['leakage_events'] += 1

    def _rotation_matrix(self, axis: str, angle: float) -> np.ndarray:
        """Generate rotation matrix for given axis and angle."""
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)

        if axis == 'X':
            return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
        elif axis == 'Y':
            return np.array([[c, -s], [s, c]], dtype=complex)
        elif axis == 'Z':
            return np.array([[np.exp(-1j*angle/2), 0], [0, np.exp(1j*angle/2)]], dtype=complex)
        else:
            raise ValueError(f"Unknown axis: {axis}")

    def _apply_single_qubit_gate(self, qubit: int, gate: np.ndarray,
                                  apply_noise: bool = True) -> None:
        """Apply single-qubit gate to density matrix."""
        state = self._states[qubit]

        # Apply decoherence from idle time
        if apply_noise:
            self._apply_decoherence(qubit)

        # Apply gate: rho -> U * rho * U†
        state.rho = gate @ state.rho @ gate.conj().T
        state.gate_count += 1

        if apply_noise:
            # Apply gate error
            self._apply_gate_error(qubit, self.noise.single_qubit_gate_error)

            # Apply crosstalk
            self._apply_crosstalk(qubit)

            # Apply leakage
            self._apply_leakage(qubit)

            # Update time (gate takes finite time)
            state.last_operation_time_ns += int(self.noise.single_qubit_gate_time_ns)

    def _apply_rx(self, qubit: int, angle: float, apply_noise: bool = True) -> None:
        """Apply RX rotation."""
        gate = self._rotation_matrix('X', angle)
        self._apply_single_qubit_gate(qubit, gate, apply_noise)

    def _apply_ry(self, qubit: int, angle: float, apply_noise: bool = True) -> None:
        """Apply RY rotation."""
        gate = self._rotation_matrix('Y', angle)
        self._apply_single_qubit_gate(qubit, gate, apply_noise)

    def _apply_rz(self, qubit: int, angle: float, apply_noise: bool = True) -> None:
        """Apply RZ rotation."""
        gate = self._rotation_matrix('Z', angle)
        self._apply_single_qubit_gate(qubit, gate, apply_noise)

    def apply_gate(self, qubits, gate_type: GateType,
                   angle: float = 0.0) -> None:
        """
        Apply quantum gate to qubit(s).

        Args:
            qubits: Single qubit index or list of qubits for multi-qubit gates
            gate_type: Type of gate to apply
            angle: Rotation angle for parameterized gates (radians)
        """
        with self._lock:
            self._stats['total_gates'] += 1

            if isinstance(qubits, int):
                qubits = [qubits]

            # Single-qubit gates
            if gate_type == GateType.I:
                pass  # Identity, but still evolve decoherence
            elif gate_type == GateType.X:
                self._apply_single_qubit_gate(qubits[0], self.X)
            elif gate_type == GateType.Y:
                self._apply_single_qubit_gate(qubits[0], self.Y)
            elif gate_type == GateType.Z:
                self._apply_single_qubit_gate(qubits[0], self.Z)
            elif gate_type == GateType.H:
                self._apply_single_qubit_gate(qubits[0], self.H)
            elif gate_type == GateType.S:
                self._apply_single_qubit_gate(qubits[0], self.S)
            elif gate_type == GateType.T:
                self._apply_single_qubit_gate(qubits[0], self.T)
            elif gate_type == GateType.RX:
                self._apply_rx(qubits[0], angle)
            elif gate_type == GateType.RY:
                self._apply_ry(qubits[0], angle)
            elif gate_type == GateType.RZ:
                self._apply_rz(qubits[0], angle)

            # Two-qubit gates
            elif gate_type == GateType.CNOT:
                self._apply_cnot(qubits[0], qubits[1])
            elif gate_type == GateType.CZ:
                self._apply_cz(qubits[0], qubits[1])
            elif gate_type == GateType.SWAP:
                self._apply_swap(qubits[0], qubits[1])

            else:
                raise ValueError(f"Unknown gate type: {gate_type}")

            # Record operation
            self._history.append({
                'time_ns': self._current_time_ns(),
                'gate': gate_type.value,
                'qubits': qubits,
                'angle': angle,
            })

    def _apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate (simplified two-qubit implementation)."""
        # Apply decoherence
        self._apply_decoherence(control)
        self._apply_decoherence(target)

        control_state = self._states[control]
        target_state = self._states[target]

        # Simplified: if control is in |1>, flip target
        # This is an approximation for separable states
        p1_control = control_state.population_1

        # Apply X to target with probability based on control |1> population
        if p1_control > 0.5:
            target_state.rho = self.X @ target_state.rho @ self.X

        # Apply two-qubit gate error
        self._apply_gate_error(control, self.noise.two_qubit_gate_error / 2)
        self._apply_gate_error(target, self.noise.two_qubit_gate_error / 2)

        # Update times
        control_state.last_operation_time_ns += int(self.noise.two_qubit_gate_time_ns)
        target_state.last_operation_time_ns += int(self.noise.two_qubit_gate_time_ns)
        control_state.gate_count += 1
        target_state.gate_count += 1

    def _apply_cz(self, qubit1: int, qubit2: int) -> None:
        """Apply CZ gate."""
        self._apply_decoherence(qubit1)
        self._apply_decoherence(qubit2)

        state1 = self._states[qubit1]
        state2 = self._states[qubit2]

        # CZ applies -1 phase when both qubits are |1>
        # Simplified implementation for separable states
        p11 = state1.population_1 * state2.population_1

        if p11 > 0.25:
            # Apply Z to both with correlation
            state1.rho[0, 1] *= -1
            state1.rho[1, 0] *= -1
            state2.rho[0, 1] *= -1
            state2.rho[1, 0] *= -1

        self._apply_gate_error(qubit1, self.noise.two_qubit_gate_error / 2)
        self._apply_gate_error(qubit2, self.noise.two_qubit_gate_error / 2)

        state1.last_operation_time_ns += int(self.noise.two_qubit_gate_time_ns)
        state2.last_operation_time_ns += int(self.noise.two_qubit_gate_time_ns)

    def _apply_swap(self, qubit1: int, qubit2: int) -> None:
        """Apply SWAP gate."""
        self._apply_decoherence(qubit1)
        self._apply_decoherence(qubit2)

        # Swap the density matrices
        self._states[qubit1].rho, self._states[qubit2].rho = \
            self._states[qubit2].rho.copy(), self._states[qubit1].rho.copy()

        self._apply_gate_error(qubit1, self.noise.two_qubit_gate_error)
        self._apply_gate_error(qubit2, self.noise.two_qubit_gate_error)

    def measure(self, qubit: int, basis: str = 'Z') -> int:
        """
        Measure qubit in specified basis.

        Args:
            qubit: Qubit index to measure
            basis: Measurement basis ('X', 'Y', 'Z')

        Returns:
            Measurement outcome (0 or 1)
        """
        with self._lock:
            self._stats['total_measurements'] += 1

            # Apply decoherence up to measurement
            self._apply_decoherence(qubit)

            state = self._states[qubit]

            # Rotate to measurement basis if not Z
            if basis == 'X':
                self._apply_single_qubit_gate(qubit, self.H, apply_noise=False)
            elif basis == 'Y':
                self._apply_single_qubit_gate(qubit, self.S.conj().T, apply_noise=False)
                self._apply_single_qubit_gate(qubit, self.H, apply_noise=False)

            # Get ideal outcome probabilities
            p0 = float(np.real(state.rho[0, 0]))
            p1 = float(np.real(state.rho[1, 1]))

            # Normalize (accounting for leakage)
            total = p0 + p1 + state.leakage_population
            if total > 0:
                p0 /= total
                p1 /= total

            # Sample ideal outcome
            ideal_outcome = 0 if self._rng.random() < p0 else 1

            # Apply readout error
            actual_outcome = ideal_outcome
            if ideal_outcome == 0:
                if self._rng.random() < self.noise.readout_error_0:
                    actual_outcome = 1
            else:
                if self._rng.random() < self.noise.readout_error_1:
                    actual_outcome = 0

            # Collapse state
            if actual_outcome == 0:
                state.rho = np.array([[1, 0], [0, 0]], dtype=complex)
            else:
                state.rho = np.array([[0, 0], [0, 1]], dtype=complex)

            # Measurement takes time
            state.last_operation_time_ns += int(self.noise.measurement_time_ns)

            # Record
            self._history.append({
                'time_ns': self._current_time_ns(),
                'gate': 'MEASURE',
                'qubits': [qubit],
                'basis': basis,
                'outcome': actual_outcome,
            })

            return actual_outcome

    def measure_all(self, basis: str = 'Z') -> List[int]:
        """Measure all qubits."""
        return [self.measure(i, basis) for i in range(self.num_qubits)]

    def reset(self, qubit: Optional[int] = None) -> None:
        """
        Reset qubit(s) to ground state.

        Args:
            qubit: Specific qubit to reset, or None for all
        """
        with self._lock:
            if qubit is not None:
                self._states[qubit].reset()
                self._states[qubit].last_operation_time_ns = self._current_time_ns()
            else:
                for state in self._states.values():
                    state.reset()
                    state.last_operation_time_ns = self._current_time_ns()

    def get_state(self, qubit: int) -> QubitState:
        """Get qubit state (for debugging/analysis)."""
        with self._lock:
            self._apply_decoherence(qubit)
            return self._states[qubit]

    def get_density_matrix(self, qubit: int) -> np.ndarray:
        """Get qubit density matrix."""
        return self.get_state(qubit).rho.copy()

    def get_bloch_vector(self, qubit: int) -> Tuple[float, float, float]:
        """Get qubit Bloch vector."""
        return self.get_state(qubit).bloch_vector()

    def get_fidelity(self, qubit: int, target_state: np.ndarray) -> float:
        """
        Calculate fidelity with target pure state.

        Args:
            qubit: Qubit index
            target_state: Target state vector [alpha, beta]

        Returns:
            Fidelity F = <psi|rho|psi>
        """
        state = self.get_state(qubit)
        target = np.array(target_state).reshape(-1, 1)
        target_dm = target @ target.conj().T
        return float(np.real(np.trace(state.rho @ target_dm)))

    def get_statistics(self) -> dict:
        """Get emulation statistics."""
        with self._lock:
            stats = self._stats.copy()

            # Add per-qubit stats
            stats['qubit_stats'] = {}
            for i, state in self._states.items():
                stats['qubit_stats'][i] = {
                    'purity': state.purity,
                    'population_0': state.population_0,
                    'population_1': state.population_1,
                    'coherence': state.coherence,
                    'accumulated_error': state.accumulated_error,
                    'gate_count': state.gate_count,
                    'leakage': state.leakage_population,
                }

            return stats

    def get_history(self) -> List[dict]:
        """Get operation history."""
        return list(self._history)

    def simulate_idle(self, duration_us: float) -> None:
        """
        Simulate idle evolution (decoherence only).

        Args:
            duration_us: Idle duration in microseconds
        """
        with self._lock:
            # Advance time
            duration_ns = int(duration_us * 1000)
            for state in self._states.values():
                state.last_operation_time_ns -= duration_ns

            # Apply decoherence
            for qubit in range(self.num_qubits):
                self._apply_decoherence(qubit)


class QuantumCircuitValidator:
    """
    Validates quantum operations meet timing and fidelity requirements.

    Integrates with RealisticQubitEmulator to verify ACCL-Q operations
    complete within coherence budgets.
    """

    def __init__(self, emulator: RealisticQubitEmulator,
                 feedback_budget_ns: float = 500.0):
        """
        Initialize validator.

        Args:
            emulator: Qubit emulator instance
            feedback_budget_ns: Maximum allowed feedback latency
        """
        self.emulator = emulator
        self.feedback_budget_ns = feedback_budget_ns

        # Validation results
        self._results: List[dict] = []

    def validate_feedback_timing(self, source_qubit: int, target_qubit: int,
                                 feedback_latency_ns: float) -> dict:
        """
        Validate that feedback operation completes within coherence time.

        Args:
            source_qubit: Qubit being measured
            target_qubit: Qubit receiving feedback
            feedback_latency_ns: Measured feedback latency

        Returns:
            Validation result dictionary
        """
        # Get target qubit coherence parameters
        t2_ns = self.emulator.noise.t2_us * 1000

        # Calculate decoherence during feedback
        decoherence_factor = np.exp(-feedback_latency_ns / t2_ns)

        # Estimate fidelity loss
        fidelity_loss = 1 - decoherence_factor

        result = {
            'source_qubit': source_qubit,
            'target_qubit': target_qubit,
            'feedback_latency_ns': feedback_latency_ns,
            'budget_ns': self.feedback_budget_ns,
            'within_budget': feedback_latency_ns <= self.feedback_budget_ns,
            't2_ns': t2_ns,
            'decoherence_factor': decoherence_factor,
            'estimated_fidelity_loss': fidelity_loss,
            'acceptable_fidelity': fidelity_loss < 0.01,  # <1% fidelity loss
        }

        self._results.append(result)
        return result

    def validate_qec_cycle(self, syndrome_latency_ns: float,
                           correction_latency_ns: float,
                           num_data_qubits: int) -> dict:
        """
        Validate QEC cycle timing.

        Args:
            syndrome_latency_ns: Time to collect and aggregate syndrome
            correction_latency_ns: Time to apply corrections
            num_data_qubits: Number of data qubits in code

        Returns:
            Validation result dictionary
        """
        total_latency = syndrome_latency_ns + correction_latency_ns

        # QEC cycle time should be << T2
        t2_ns = self.emulator.noise.t2_us * 1000

        # Estimate logical error rate improvement
        # (simplified - real calculation depends on code and noise model)
        physical_error = self.emulator.noise.single_qubit_gate_error

        # Decoherence during cycle
        cycle_decoherence = 1 - np.exp(-total_latency / t2_ns)

        result = {
            'syndrome_latency_ns': syndrome_latency_ns,
            'correction_latency_ns': correction_latency_ns,
            'total_cycle_ns': total_latency,
            't2_ns': t2_ns,
            'cycle_fraction_of_t2': total_latency / t2_ns,
            'cycle_decoherence': cycle_decoherence,
            'physical_error_rate': physical_error,
            'num_data_qubits': num_data_qubits,
            'qec_effective': total_latency < t2_ns / 10,  # Cycle should be < T2/10
        }

        self._results.append(result)
        return result

    def get_validation_summary(self) -> dict:
        """Get summary of all validation results."""
        if not self._results:
            return {'num_validations': 0}

        timing_results = [r for r in self._results if 'within_budget' in r]
        qec_results = [r for r in self._results if 'qec_effective' in r]

        return {
            'num_validations': len(self._results),
            'timing_validations': {
                'total': len(timing_results),
                'passed': sum(1 for r in timing_results if r['within_budget']),
                'avg_latency_ns': np.mean([r['feedback_latency_ns'] for r in timing_results]) if timing_results else 0,
            },
            'qec_validations': {
                'total': len(qec_results),
                'passed': sum(1 for r in qec_results if r['qec_effective']),
                'avg_cycle_ns': np.mean([r['total_cycle_ns'] for r in qec_results]) if qec_results else 0,
            },
        }
