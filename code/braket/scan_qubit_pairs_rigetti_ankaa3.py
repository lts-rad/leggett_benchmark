#!/usr/bin/env python3
"""
Scan qubit pairs for Rigetti Ankaa-3 (84-qubit square lattice) using exact Leggett test logic.
Test C(a2, b2) and C(a2, b2') correlations with φ = 30°.

Rigetti Ankaa-3 specifications:
- 84 qubits (superconducting transmon)
- Square lattice topology (approximate 9x10 grid)
- Native gates: rx, rz, iswap
- Typical fidelities: ~99.8% 1Q, ~97.5% 2Q, ~98% readout

This version simulates per-qubit noise variation to model what real hardware
would show. On real Rigetti hardware, different qubit pairs would have different
error rates based on their calibration state.
"""

import numpy as np
import json
from datetime import datetime

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.circuits import gates
from braket.circuits.noises import Depolarizing, TwoQubitDepolarizing, BitFlip


def get_rigetti_ankaa3_coupling_map():
    """
    Generate the coupling map for Rigetti Ankaa-3's 84-qubit grid.

    The topology is approximately a 9x10 grid with nearest-neighbor connectivity.
    Qubit numbering: row-major order, q[i] = row * 10 + col (for first 80)
    Plus 4 additional qubits at the edges.
    """
    rows, cols = 8, 10  # 80 qubits in main grid
    edges = []

    for row in range(rows):
        for col in range(cols):
            qubit = row * cols + col

            # Connect to right neighbor
            if col + 1 < cols:
                right = row * cols + (col + 1)
                edges.append((qubit, right))

            # Connect to bottom neighbor
            if row + 1 < rows:
                bottom = (row + 1) * cols + col
                edges.append((qubit, bottom))

    # Add 4 extra qubits (80-83) at edges
    # Typical Ankaa-3 has some edge connections
    edges.append((79, 80))  # Edge qubit connections
    edges.append((80, 81))
    edges.append((81, 82))
    edges.append((82, 83))

    return edges


def generate_qubit_error_rates(num_qubits=84, seed=42):
    """
    Generate realistic per-qubit error rates based on typical Rigetti calibration data.

    Returns dictionaries of:
    - 1Q gate errors (per qubit)
    - 2Q gate errors (per pair)
    - Readout errors (per qubit)
    """
    np.random.seed(seed)

    # Base error rates (Rigetti Ankaa-3 typical - slightly worse than IQM)
    base_q1_error = 0.002   # 0.2% 1Q error
    base_q2_error = 0.025   # 2.5% 2Q error
    base_readout = 0.02     # 2% readout error

    # Generate variation (log-normal distribution)
    q1_errors = {}
    readout_errors = {}

    for q in range(num_qubits):
        # 1Q errors vary by ~2-3x around the mean
        q1_errors[q] = base_q1_error * np.exp(np.random.normal(0, 0.5))
        # Readout errors vary by ~1.5-2x
        readout_errors[q] = base_readout * np.exp(np.random.normal(0, 0.3))

    # 2Q errors for each coupled pair
    q2_errors = {}
    edges = get_rigetti_ankaa3_coupling_map()

    for q1, q2 in edges:
        pair = tuple(sorted([q1, q2]))
        if pair not in q2_errors:
            # 2Q errors vary by ~2-5x, some pairs are much worse
            q2_errors[pair] = base_q2_error * np.exp(np.random.normal(0, 0.7))

    return q1_errors, q2_errors, readout_errors


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def create_single_correlation_circuit(a_vec, b_vec):
    """
    Create Braket circuit for ONE correlation measurement C(a, b).
    Always uses qubits 0 (Alice) and 1 (Bob).
    """
    circ = Circuit()

    q_alice, q_bob = 0, 1

    # Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    circ.x(q_bob)
    circ.h(q_alice)
    circ.cnot(q_alice, q_bob)
    circ.z(q_bob)

    # Apply measurement rotations
    theta_a, phi_a = poincare_to_angles(a_vec)
    theta_b, phi_b = poincare_to_angles(b_vec)

    circ.rz(q_alice, -phi_a)
    circ.ry(q_alice, -theta_a)
    circ.rz(q_bob, -phi_b)
    circ.ry(q_bob, -theta_b)

    return circ


def add_rigetti_noise_model(circ, q1_error, q2_error, readout_error):
    """Add Rigetti-style noise to a circuit."""
    noisy_circ = circ.copy()

    noisy_circ.apply_gate_noise(
        Depolarizing(probability=q1_error),
        target_gates=[gates.H, gates.X, gates.Y, gates.Z, gates.Ry, gates.Rz, gates.Rx]
    )

    noisy_circ.apply_gate_noise(
        TwoQubitDepolarizing(probability=q2_error),
        target_gates=[gates.CNot]
    )

    noisy_circ.apply_readout_noise(BitFlip(probability=readout_error))

    return noisy_circ


def test_qubit_pair(q1, q2, q1_errors, q2_errors, readout_errors, use_noise=True, phi_deg=30, num_shots=500):
    """Test a qubit pair by measuring C(a2, b2) and C(a2, b2') with φ = phi_deg."""
    phi_rad = np.radians(phi_deg)

    # Alice always on y-axis
    a2 = np.array([0, 1, 0])

    # Bob at ±φ/2 around y-axis
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])

    # Get pair-specific error rates
    pair = tuple(sorted([q1, q2]))
    avg_q1_error = (q1_errors[q1] + q1_errors[q2]) / 2
    q2_error = q2_errors[pair]
    avg_readout = (readout_errors[q1] + readout_errors[q2]) / 2

    errors = []
    correlations = []

    simulator = LocalSimulator('braket_dm' if use_noise else 'braket_sv')

    for b_vec, label in [(b2, 'b2'), (b2_prime, "b2'")]:
        circ = create_single_correlation_circuit(a2, b_vec)

        if use_noise:
            circ = add_rigetti_noise_model(circ, avg_q1_error, q2_error, avg_readout)

        task = simulator.run(circ, shots=num_shots)
        result = task.result()
        counts = result.measurement_counts

        # Calculate correlation
        correlation = 0.0
        for bitstring, count in counts.items():
            padded = bitstring.zfill(2)
            alice_bit = int(padded[1])
            bob_bit = int(padded[0])

            if alice_bit == bob_bit:
                correlation += count
            else:
                correlation -= count

        correlation /= num_shots

        # Theoretical: C(a, b) = -a·b for singlet
        correlation_theory = -np.dot(a2, b_vec)
        error = abs(correlation - correlation_theory)

        errors.append(error)
        correlations.append(correlation)

    avg_error = np.mean(errors)
    return avg_error, correlations, errors, {'q1_error': avg_q1_error, 'q2_error': q2_error, 'readout_error': avg_readout}


def scan_coupled_pairs(use_noise=True, num_shots=500, phi_deg=30, max_pairs=None):
    """Scan coupled qubit pairs on Rigetti Ankaa-3 topology."""
    edges = get_rigetti_ankaa3_coupling_map()

    # Get unique pairs
    unique_pairs = []
    seen = set()
    for q1, q2 in edges:
        pair = tuple(sorted([q1, q2]))
        if pair not in seen:
            unique_pairs.append((q1, q2))
            seen.add(pair)

    if max_pairs:
        unique_pairs = unique_pairs[:max_pairs]

    # Generate per-qubit/pair error rates
    q1_errors, q2_errors, readout_errors = generate_qubit_error_rates()

    print(f"Testing {len(unique_pairs)} qubit pairs on Rigetti Ankaa-3 (84-qubit grid)")
    print(f"Noise model: {'enabled with per-pair variation' if use_noise else 'disabled'}")
    print(f"φ = {phi_deg}°, φ/2 = {phi_deg/2}°")

    # Calculate theoretical correlations
    a2 = np.array([0, 1, 0])
    b2 = np.array([0, np.cos(np.radians(phi_deg)/2), np.sin(np.radians(phi_deg)/2)])
    b2_prime = np.array([0, np.cos(np.radians(phi_deg)/2), -np.sin(np.radians(phi_deg)/2)])

    corr_b2_theory = -np.dot(a2, b2)
    corr_b2p_theory = -np.dot(a2, b2_prime)

    print(f"Theoretical C(a2, b2) = {corr_b2_theory:.4f}")
    print(f"Theoretical C(a2, b2') = {corr_b2p_theory:.4f}")
    print()

    results = []

    for i, (q1, q2) in enumerate(unique_pairs):
        print(f"Testing {i+1}/{len(unique_pairs)}: ({q1},{q2})... ", end='', flush=True)

        try:
            avg_error, correlations, errors, noise_params = test_qubit_pair(
                q1, q2, q1_errors, q2_errors, readout_errors, use_noise, phi_deg, num_shots
            )
            results.append({
                'qubits': [q1, q2],
                'avg_error': avg_error,
                'correlations': correlations,
                'errors': errors,
                'phi_deg': phi_deg,
                'noise_params': noise_params
            })
            print(f"avg_err={avg_error:.4f}, C(b2)={correlations[0]:+.3f}, C(b2')={correlations[1]:+.3f}, 2Q_err={noise_params['q2_error']*100:.2f}%")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    return results


if __name__ == "__main__":
    import sys

    num_shots = 500
    phi_deg = 30
    use_noise = '--noise' in sys.argv
    max_pairs = None

    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--phi' and i + 1 < len(sys.argv):
            phi_deg = float(sys.argv[i + 1])
        elif arg == '--max-pairs' and i + 1 < len(sys.argv):
            max_pairs = int(sys.argv[i + 1])

    print("="*70)
    print("SCANNING QUBIT PAIRS - RIGETTI ANKAA-3 (84-qubit grid)")
    print("="*70)
    print(f"Shots per pair: {num_shots}")
    print(f"Noise model: {'enabled with per-pair variation' if use_noise else 'disabled'}")
    print()

    results = scan_coupled_pairs(use_noise, num_shots, phi_deg, max_pairs)

    # Sort by error
    results.sort(key=lambda x: x['avg_error'])

    print("\n" + "="*70)
    print(f"TOP 20 BEST QUBIT PAIRS")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Avg Err':>10} {'C(a2,b2)':>11} {'C(a2,b2p)':>11} {'2Q Err%':>10}")
    print("-"*70)

    for i, r in enumerate(results[:20]):
        q1, q2 = r['qubits']
        q2_err = r['noise_params']['q2_error'] * 100
        print(f"{i+1:>4} ({q1:3d},{q2:3d})     {r['avg_error']:>10.4f} {r['correlations'][0]:>+11.4f} {r['correlations'][1]:>+11.4f} {q2_err:>10.2f}")

    print("\n" + "="*70)
    print("WORST 10 QUBIT PAIRS")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Avg Err':>10} {'C(a2,b2)':>11} {'C(a2,b2p)':>11} {'2Q Err%':>10}")
    print("-"*70)

    for i, r in enumerate(results[-10:]):
        q1, q2 = r['qubits']
        rank = len(results) - 10 + i + 1
        q2_err = r['noise_params']['q2_error'] * 100
        print(f"{rank:>4} ({q1:3d},{q2:3d})     {r['avg_error']:>10.4f} {r['correlations'][0]:>+11.4f} {r['correlations'][1]:>+11.4f} {q2_err:>10.2f}")

    noise_suffix = "_NOISE" if use_noise else ""
    output_file = f'qubit_pair_scan_rigetti_ankaa3_phi{int(phi_deg)}{noise_suffix}.json'

    with open(output_file, 'w') as f:
        json.dump({
            'device': 'Rigetti Ankaa-3 (emulator)',
            'topology': '84-qubit grid (8x10 + edges)',
            'noise_model': use_noise,
            'noise_heterogeneity': 'per-pair variation simulated',
            'phi_deg': phi_deg,
            'shots_per_pair': num_shots,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Saved to {output_file}")
    print(f"{'='*70}")

    print("\nRECOMMENDED LAYOUT FOR 12-QUBIT CIRCUIT (top 6 non-overlapping pairs):")
    layout = []
    used_qubits = set()
    selected_pairs = []

    for r in results:
        q1, q2 = r['qubits']
        if q1 not in used_qubits and q2 not in used_qubits:
            layout.extend([q1, q2])
            used_qubits.add(q1)
            used_qubits.add(q2)
            selected_pairs.append(r)
            if len(selected_pairs) >= 6:
                break

    print(f"  physical_pairs = {[(r['qubits'][0], r['qubits'][1]) for r in selected_pairs]}")

    print("\nBest non-overlapping pairs:")
    for i, r in enumerate(selected_pairs):
        q1, q2 = r['qubits']
        q2_err = r['noise_params']['q2_error'] * 100
        print(f"  Pair {i+1}: ({q1},{q2}) - avg error: {r['avg_error']:.4f}, 2Q error: {q2_err:.2f}%")
