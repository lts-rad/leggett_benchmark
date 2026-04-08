"""
Leggett inequality test circuit construction and correlation analysis.

Core library for building quantum circuits that test the Leggett inequality
(arXiv:0801.2241v2, Branciard et al.) on various quantum hardware platforms.

Provides:
- Optimized measurement basis rotations (single-gate where possible)
- Circuit builders for different qubit counts and Bell state variants
- Correlation extraction from measurement counts
- Leggett parameter L₃ calculation with violation detection

Measurement basis rotation optimization:
- Z-axis: no rotation needed
- XZ plane (y≈0): single Ry with angle = -arctan2(x, z)
- YZ plane (x≈0): single Rx with angle = +arctan2(y, z)
- General/XY plane: Rz then Ry
"""

import numpy as np
from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# Measurement basis helpers
# ---------------------------------------------------------------------------

def get_measurement_gates(vec):
    """
    Get the optimal gate sequence to rotate the computational basis
    so that a Z-measurement effectively measures along `vec`.

    Returns list of (gate_name, angle) tuples.
    """
    x, y, z = vec

    if abs(x) < 1e-10 and abs(y) < 1e-10:
        return []

    if abs(y) < 1e-10:  # XZ plane
        angle = -np.arctan2(x, z)
        return [('ry', angle)] if abs(angle) > 1e-10 else []

    if abs(x) < 1e-10:  # YZ plane
        angle = np.arctan2(y, z)
        return [('rx', angle)] if abs(angle) > 1e-10 else []

    # General case
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    gates = []
    if abs(phi) > 1e-10:
        gates.append(('rz', -phi))
    if abs(theta) > 1e-10:
        gates.append(('ry', -theta))
    return gates


def apply_measurement_rotation(qc, qubit, vec):
    """Apply optimal measurement rotation to measure along `vec`."""
    for gate_name, angle in get_measurement_gates(vec):
        getattr(qc, gate_name)(angle, qubit)


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to (theta, phi) spherical angles."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def measure_polarization(qc, qubit, theta, phi_angle):
    """Measure polarization at angle (theta, phi) via Rz(-phi) Ry(-theta)."""
    qc.rz(-phi_angle, qubit)
    qc.ry(-theta, qubit)


# ---------------------------------------------------------------------------
# Measurement directions
# ---------------------------------------------------------------------------

def _alice_directions():
    """Alice's three orthogonal measurement directions."""
    return (
        np.array([1, 0, 0]),  # a₁ = X
        np.array([0, 1, 0]),  # a₂ = Y
        np.array([0, 0, 1]),  # a₃ = Z
    )


def _bob_directions(phi_rad):
    """Bob's six measurement directions for a given phi (b and b' for each a)."""
    c = np.cos(phi_rad / 2)
    s = np.sin(phi_rad / 2)
    return [
        np.array([c,  s, 0]),   # b₁
        np.array([c, -s, 0]),   # b₁'
        np.array([0,  c, s]),   # b₂
        np.array([0,  c, -s]),  # b₂'
        np.array([s,  0, c]),   # b₃
        np.array([-s, 0, c]),   # b₃'
    ]


def _measurement_pairs_12(phi_rad):
    """12 (Alice, Bob) measurement pairs: 6 for +phi, 6 for -phi."""
    a1, a2, a3 = _alice_directions()
    b_pos = _bob_directions(phi_rad)
    b_neg = _bob_directions(-phi_rad)
    return [
        (a1, b_pos[0]), (a1, b_pos[1]),
        (a2, b_pos[2]), (a2, b_pos[3]),
        (a3, b_pos[4]), (a3, b_pos[5]),
        (a1, b_neg[0]), (a1, b_neg[1]),
        (a2, b_neg[2]), (a2, b_neg[3]),
        (a3, b_neg[4]), (a3, b_neg[5]),
    ]


def _measurement_pairs_6(phi_rad):
    """6 (Alice, Bob) measurement pairs for +phi only."""
    a1, a2, a3 = _alice_directions()
    b = _bob_directions(phi_rad)
    return [
        (a1, b[0]), (a1, b[1]),
        (a2, b[2]), (a2, b[3]),
        (a3, b[4]), (a3, b[5]),
    ]


# ---------------------------------------------------------------------------
# Singlet state preparation helpers
# ---------------------------------------------------------------------------

def _prepare_singlet(qc, qubit_a, qubit_b):
    """|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2"""
    qc.x(qubit_b)
    qc.h(qubit_a)
    qc.cx(qubit_a, qubit_b)
    qc.z(qubit_b)


def _prepare_phi_plus(qc, qubit_a, qubit_b):
    """|Φ⁺⟩ = (|00⟩ + |11⟩)/√2"""
    qc.h(qubit_a)
    qc.cx(qubit_a, qubit_b)


def _prepare_phi_minus(qc, qubit_a, qubit_b):
    """|Φ⁻⟩ = (|00⟩ - |11⟩)/√2"""
    qc.h(qubit_a)
    qc.z(qubit_a)
    qc.cx(qubit_a, qubit_b)


def _prepare_psi_plus(qc, qubit_a, qubit_b):
    """|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2"""
    qc.x(qubit_b)
    qc.h(qubit_a)
    qc.cx(qubit_a, qubit_b)


_BELL_STATE_PREP = {
    'singlet':    _prepare_singlet,
    'psi_minus':  _prepare_singlet,
    'phi_plus':   _prepare_phi_plus,
    'phi_minus':  _prepare_phi_minus,
    'psi_plus':   _prepare_psi_plus,
}


# ---------------------------------------------------------------------------
# Generic circuit builder
# ---------------------------------------------------------------------------

def _build_leggett_circuit(phi_rad, pairs, bell_state='singlet',
                           use_barrier=False, use_reset=False):
    """
    Build a Leggett test circuit for the given measurement pairs.

    Args:
        phi_rad: Angle φ in radians (informational, pairs already computed)
        pairs: List of (alice_vec, bob_vec) measurement direction tuples
        bell_state: Which Bell state to prepare
        use_barrier: Insert barriers between pairs
        use_reset: Reset qubits before each pair (crosstalk mitigation)

    Returns:
        QuantumCircuit
    """
    num_qubits = len(pairs) * 2
    qc = QuantumCircuit(num_qubits)
    prep_fn = _BELL_STATE_PREP[bell_state]

    for i, (a_vec, b_vec) in enumerate(pairs):
        qa = 2 * i
        qb = 2 * i + 1

        if use_reset:
            qc.reset(qa)
            qc.reset(qb)

        prep_fn(qc, qa, qb)
        apply_measurement_rotation(qc, qa, a_vec)
        apply_measurement_rotation(qc, qb, b_vec)

        if use_barrier:
            qc.barrier()

    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Circuit constructors — 24 qubit (12 pairs, +φ and -φ)
# ---------------------------------------------------------------------------

def create_leggett_circuit_for_angle(phi_rad):
    """24-qubit circuit: 12 singlet pairs (6 for +φ, 6 for -φ)."""
    return _build_leggett_circuit(phi_rad, _measurement_pairs_12(phi_rad))


def create_leggett_circuit_for_angle_barrier(phi_rad):
    """24-qubit circuit with barriers and resets between pairs."""
    return _build_leggett_circuit(
        phi_rad, _measurement_pairs_12(phi_rad),
        use_barrier=True, use_reset=True,
    )


# ---------------------------------------------------------------------------
# Circuit constructors — 12 qubit (6 pairs, +φ only)
# ---------------------------------------------------------------------------

def create_leggett_circuit_for_angle_six(phi_rad, bell_state='singlet'):
    """12-qubit circuit: 6 singlet pairs for +φ only."""
    return _build_leggett_circuit(
        phi_rad, _measurement_pairs_6(phi_rad), bell_state=bell_state,
    )


def create_leggett_circuit_for_angle_0011_six(phi_rad):
    """12-qubit circuit using |Φ⁺⟩ = (|00⟩+|11⟩)/√2."""
    return create_leggett_circuit_for_angle_six(phi_rad, bell_state='phi_plus')


def create_leggett_circuit_for_angle_00n11_six(phi_rad):
    """12-qubit circuit using |Φ⁻⟩ = (|00⟩-|11⟩)/√2."""
    return create_leggett_circuit_for_angle_six(phi_rad, bell_state='phi_minus')


def create_leggett_circuit_for_angle_01p10_six(phi_rad):
    """12-qubit circuit using |Ψ⁺⟩ = (|01⟩+|10⟩)/√2."""
    return create_leggett_circuit_for_angle_six(phi_rad, bell_state='psi_plus')


def create_leggett_circuit_for_angle_six_barrier(phi_rad):
    """12-qubit circuit with barriers between pairs."""
    return _build_leggett_circuit(
        phi_rad, _measurement_pairs_6(phi_rad),
        use_barrier=True, use_reset=True,
    )


def create_leggett_circuit_for_angle_six_barrier_NOP(phi_rad):
    """13-qubit circuit with barriers and a NOP ancilla qubit."""
    pairs = _measurement_pairs_6(phi_rad)
    num_qubits = len(pairs) * 2 + 1  # +1 for NOP qubit
    qc = QuantumCircuit(num_qubits)

    for i, (a_vec, b_vec) in enumerate(pairs):
        qa = 2 * i
        qb = 2 * i + 1

        _prepare_singlet(qc, qa, qb)
        apply_measurement_rotation(qc, qa, a_vec)
        apply_measurement_rotation(qc, qb, b_vec)
        qc.barrier()
        qc.barrier()

    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Circuit constructors — large scale (48qb redundant, 64 pairs, 70 pairs)
# ---------------------------------------------------------------------------

def create_leggett_circuit_for_angle_redundant(phi_rad):
    """48-qubit circuit: 24 pairs (each of 12 correlations measured twice)."""
    pairs = _measurement_pairs_12(phi_rad) * 2  # duplicate
    return _build_leggett_circuit(phi_rad, pairs)


def create_leggett_circuit_twelve(phi_rad):
    """24-qubit circuit: 12 pairs for +φ only (two copies of 6 correlations)."""
    pairs = _measurement_pairs_6(phi_rad) * 2
    return _build_leggett_circuit(phi_rad, pairs)


def _create_leggett_circuit_n_pairs(phi_rad, num_pairs):
    """Generic N-pair circuit cycling through 12 correlation types."""
    corr_types = _measurement_pairs_12(phi_rad)
    pairs = [corr_types[i % 12] for i in range(num_pairs)]
    return _build_leggett_circuit(phi_rad, pairs)


def create_leggett_circuit_64_pairs(phi_rad):
    """128-qubit circuit: 64 pairs (~5.3x redundancy per correlation)."""
    return _create_leggett_circuit_n_pairs(phi_rad, 64)


def create_leggett_circuit_70_pairs(phi_rad):
    """140-qubit circuit: 70 pairs (~5.8x redundancy per correlation)."""
    return _create_leggett_circuit_n_pairs(phi_rad, 70)


# ---------------------------------------------------------------------------
# Correlation extraction
# ---------------------------------------------------------------------------

def extract_correlation_from_counts(counts, num_shots):
    """Extract correlation from 2-qubit measurement counts."""
    correlation = 0.0
    for bitstring, count in counts.items():
        alice_bit = int(bitstring[1])
        bob_bit = int(bitstring[0])
        if alice_bit == bob_bit:
            correlation += count
        else:
            correlation -= count
    return correlation / num_shots


def _extract_pair_correlations(counts, num_shots, num_pairs):
    """Extract per-pair correlations from an N-pair circuit."""
    num_qubits = num_pairs * 2
    correlations = [0.0] * num_pairs

    for bitstring, count in counts.items():
        for pair_idx in range(num_pairs):
            alice_bit = int(bitstring[num_qubits - 1 - 2 * pair_idx])
            bob_bit = int(bitstring[num_qubits - 2 - 2 * pair_idx])
            if alice_bit == bob_bit:
                correlations[pair_idx] += count
            else:
                correlations[pair_idx] -= count

    return [c / num_shots for c in correlations]


def extract_correlations_from_counts(counts, num_shots):
    """
    Extract 12 correlations from 24-qubit circuit.

    Returns (correlations_pos, correlations_neg) — 6 values each.
    """
    correlations = _extract_pair_correlations(counts, num_shots, 12)
    return correlations[:6], correlations[6:]


def extract_correlations_from_counts_six(counts, num_shots, bell_state='singlet'):
    """
    Extract 6 correlations from 12-qubit circuit.

    Returns list of 6 correlation values for +φ.
    """
    return _extract_pair_correlations(counts, num_shots, 6)


def extract_correlations_from_counts_twelve(counts, num_shots):
    """
    Extract 12 correlations from 24-qubit circuit (two copies of 6).

    Returns averaged 6 correlation values.
    """
    raw = _extract_pair_correlations(counts, num_shots, 12)
    return [(raw[i] + raw[i + 6]) / 2.0 for i in range(6)]


def extract_correlations_from_counts_redundant(counts, num_shots):
    """
    Extract and average from 48-qubit redundant circuit.

    Returns (correlations_pos, correlations_neg) averaged across copies.
    """
    raw = _extract_pair_correlations(counts, num_shots, 24)
    correlations_pos = [(raw[i] + raw[i + 12]) / 2.0 for i in range(6)]
    correlations_neg = [(raw[i] + raw[i + 12]) / 2.0 for i in range(6, 12)]
    return correlations_pos, correlations_neg


def _extract_n_pair_averaged(counts, num_shots, num_pairs):
    """Extract and average correlations from N-pair circuit cycling 12 types."""
    num_qubits = num_pairs * 2
    correlation_sums = [0.0] * 12
    correlation_counts = [0] * 12

    for bitstring, count in counts.items():
        for pair_idx in range(num_pairs):
            corr_idx = pair_idx % 12
            alice_bit = int(bitstring[num_qubits - 1 - 2 * pair_idx])
            bob_bit = int(bitstring[num_qubits - 2 - 2 * pair_idx])
            if alice_bit == bob_bit:
                correlation_sums[corr_idx] += count
            else:
                correlation_sums[corr_idx] -= count
            correlation_counts[corr_idx] += count

    correlations = [
        s / c if c > 0 else 0.0
        for s, c in zip(correlation_sums, correlation_counts)
    ]
    return correlations[:6], correlations[6:]


def extract_correlations_from_counts_64_pairs(counts, num_shots):
    """Extract averaged correlations from 64-pair (128-qubit) circuit."""
    return _extract_n_pair_averaged(counts, num_shots, 64)


def extract_correlations_from_counts_70_pairs(counts, num_shots):
    """Extract averaged correlations from 70-pair (140-qubit) circuit."""
    return _extract_n_pair_averaged(counts, num_shots, 70)


# ---------------------------------------------------------------------------
# Leggett parameter calculation
# ---------------------------------------------------------------------------

def calc_leggett_for_angle(correlations, phi_rad, bell_state='singlet'):
    """
    Calculate Leggett inequality parameter L₃ for given correlations.

    Args:
        correlations: [C(a₁,b₁), C(a₁,b₁'), C(a₂,b₂), C(a₂,b₂'), C(a₃,b₃), C(a₃,b₃')]
        phi_rad: Angle φ in radians
        bell_state: 'singlet', 'psi_plus', 'phi_plus', or 'phi_minus'

    Returns:
        dict with L3, L3_theory, bound, violated, correlations, correlations_theory
    """
    C_a1b1, C_a1b1p, C_a2b2, C_a2b2p, C_a3b3, C_a3b3p = correlations

    a1, a2, a3 = _alice_directions()
    b = _bob_directions(phi_rad)
    b1, b1p, b2, b2p, b3, b3p = b

    # Theoretical correlations depend on Bell state
    if bell_state in ('singlet', 'psi_minus'):
        # |Ψ⁻⟩: C(a,b) = -a·b
        sign = -1
        th = [sign * np.dot(a, bv) for a, bv in
              [(a1, b1), (a1, b1p), (a2, b2), (a2, b2p), (a3, b3), (a3, b3p)]]
    elif bell_state == 'psi_plus':
        # |Ψ⁺⟩: XX and YY correlations flip sign relative to singlet
        th = [
            np.dot(a1, b1), np.dot(a1, b1p),
            np.dot(a2, b2), np.dot(a2, b2p),
            -np.dot(a3, b3), -np.dot(a3, b3p),
        ]
    elif bell_state == 'phi_plus':
        # |Φ⁺⟩: C(a,b) = +a·b
        th = [np.dot(a, bv) for a, bv in
              [(a1, b1), (a1, b1p), (a2, b2), (a2, b2p), (a3, b3), (a3, b3p)]]
    elif bell_state == 'phi_minus':
        # |Φ⁻⟩: same pattern as singlet
        th = [-np.dot(a, bv) for a, bv in
              [(a1, b1), (a1, b1p), (a2, b2), (a2, b2p), (a3, b3), (a3, b3p)]]
    else:
        raise ValueError(f"Unknown bell_state: {bell_state}")

    # L₃ parameter
    L3 = (1 / 3) * (abs(C_a1b1 + C_a1b1p) +
                     abs(C_a2b2 + C_a2b2p) +
                     abs(C_a3b3 + C_a3b3p))

    L3_th = (1 / 3) * (abs(th[0] + th[1]) +
                        abs(th[2] + th[3]) +
                        abs(th[4] + th[5]))

    L_bound = 2 - (2 / 3) * abs(np.sin(phi_rad / 2))

    return {
        'correlations': correlations,
        'correlations_theory': th,
        'L3': L3,
        'L3_theory': L3_th,
        'bound': L_bound,
        'violated': L3 > L_bound,
    }


# ---------------------------------------------------------------------------
# Debug utility
# ---------------------------------------------------------------------------

def print_gate_summary(phi_deg):
    """Print the gate sequences for each measurement direction at a given angle."""
    phi_rad = np.radians(phi_deg)
    a1, a2, a3 = _alice_directions()
    b = _bob_directions(phi_rad)

    directions = {
        'a₁ (X)': a1, 'a₂ (Y)': a2, 'a₃ (Z)': a3,
        'b₁': b[0], "b₁'": b[1],
        'b₂': b[2], "b₂'": b[3],
        'b₃': b[4], "b₃'": b[5],
    }

    print(f"\nGate sequences for φ={phi_deg}°:")
    print("=" * 50)
    for name, vec in directions.items():
        gates = get_measurement_gates(vec)
        if not gates:
            gate_str = "none (Z-basis)"
        else:
            gate_str = " → ".join(
                f"{g}({np.degrees(a):.1f}°)" for g, a in gates
            )
        print(f"  {name:8s}: {gate_str}")


if __name__ == "__main__":
    print_gate_summary(30)

    qc = create_leggett_circuit_for_angle(np.radians(30))
    print(f"\nCircuit: {qc.num_qubits} qubits, depth {qc.depth()}")
    print(f"Gate counts: {qc.count_ops()}")
