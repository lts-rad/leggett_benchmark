import os
#!/usr/bin/env python3
"""
Scan qubit pairs using exact Leggett test logic.
Test C(a2, b2) and C(a2, b2') correlations with φ = 30°.
"""

import numpy as np
import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi

def measure_polarization(qc, qubit, theta, phi_angle):
    """Measure polarization at angle specified by Poincaré sphere vector."""
    qc.rz(-phi_angle, qubit)
    qc.ry(-theta, qubit)

def create_single_correlation_circuit(a_vec, b_vec):
    """
    Create circuit for ONE correlation measurement C(a, b).
    Uses exact leggett.py logic for singlet + Poincaré rotations.
    """
    qc = QuantumCircuit(2, 2)

    # Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(1)

    # Apply measurement rotations
    theta_a, phi_a = poincare_to_angles(a_vec)
    theta_b, phi_b = poincare_to_angles(b_vec)

    measure_polarization(qc, 0, theta_a, phi_a)
    measure_polarization(qc, 1, theta_b, phi_b)

    qc.measure([0, 1], [0, 1])
    return qc

def test_qubit_pair(q1, q2, backend, noise_model, phi_deg=30, num_shots=500):
    """
    Test a qubit pair by measuring C(a2, b2) and C(a2, b2') with φ = phi_deg.

    Returns average error compared to ideal QM correlation.
    """
    phi_rad = np.radians(phi_deg)

    # Alice always on y-axis
    a2 = np.array([0, 1, 0])

    # Bob at ±φ/2 around y-axis
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])

    errors = []
    correlations = []

    for b_vec, label in [(b2, 'b2'), (b2_prime, "b2'")]:
        qc = create_single_correlation_circuit(a2, b_vec)

        # Transpile
        qc_transpiled = transpile(
            qc,
            backend=backend,
            optimization_level=2,
            initial_layout=[q1, q2]
        )

        # Run
        simulator = AerSimulator(noise_model=noise_model)
        result = simulator.run(qc_transpiled, shots=num_shots).result()
        counts = result.get_counts()

        # Calculate correlation using Leggett method
        correlation = 0.0
        for bitstring, count in counts.items():
            alice_bit = int(bitstring[1])  # q0
            bob_bit = int(bitstring[0])    # q1

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

    # Return average error
    avg_error = np.mean(errors)
    return avg_error, correlations, errors

def scan_coupled_pairs(backend, noise_model, num_shots=500, blacklist=None, phi_deg=30):
    """Scan ALL coupled qubit pairs."""
    if blacklist is None:
        blacklist = set()
    else:
        blacklist = set(blacklist)

    coupling_map = backend.coupling_map
    edges = coupling_map.get_edges()

    valid_edges = [
        (q1, q2) for q1, q2 in edges
        if q1 not in blacklist and q2 not in blacklist
    ]

    print(f"Testing ALL {len(valid_edges)} qubit pairs")
    print(f"Blacklist: {sorted(blacklist)}")
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

    for i, (q1, q2) in enumerate(valid_edges):
        print(f"Testing {i+1}/{len(valid_edges)}: ({q1},{q2})... ", end='', flush=True)

        try:
            avg_error, correlations, errors = test_qubit_pair(
                q1, q2, backend, noise_model, phi_deg, num_shots
            )
            results.append({
                'qubits': [q1, q2],
                'avg_error': avg_error,
                'correlations': correlations,
                'errors': errors,
                'phi_deg': phi_deg
            })
            print(f"avg_err={avg_error:.4f}, C(b2)={correlations[0]:+.3f}, C(b2')={correlations[1]:+.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    return results

if __name__ == "__main__":
    import sys

    num_shots = 500
    blacklist = []
    phi_deg = 30

    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--blacklist' and i + 1 < len(sys.argv):
            blacklist = [int(x) for x in sys.argv[i + 1].split(',')]
        elif arg == '--phi' and i + 1 < len(sys.argv):
            phi_deg = float(sys.argv[i + 1])

    print("="*70)
    print("SCANNING QUBIT PAIRS - EXACT LEGGETT METHOD")
    print("="*70)
    print(f"Shots per pair: {num_shots}")
    print()

    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)

    results = scan_coupled_pairs(backend, noise_model, num_shots, blacklist, phi_deg)

    # Sort by error
    results.sort(key=lambda x: x['avg_error'])

    print("\n" + "="*70)
    print(f"TOP 20 BEST QUBIT PAIRS")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Avg Err':>10} {'C(a2,b2)':>11} {'C(a2,b2p)':>11}")
    print("-"*70)

    for i, r in enumerate(results[:20]):
        q1, q2 = r['qubits']
        print(f"{i+1:>4} ({q1:3d},{q2:3d})     {r['avg_error']:>10.4f} {r['correlations'][0]:>+11.4f} {r['correlations'][1]:>+11.4f}")

    print("\n" + "="*70)
    print("WORST 10 QUBIT PAIRS")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Avg Err':>10} {'C(a2,b2)':>11} {'C(a2,b2p)':>11}")
    print("-"*70)

    for i, r in enumerate(results[-10:]):
        q1, q2 = r['qubits']
        rank = len(results) - 10 + i + 1
        print(f"{rank:>4} ({q1:3d},{q2:3d})     {r['avg_error']:>10.4f} {r['correlations'][0]:>+11.4f} {r['correlations'][1]:>+11.4f}")

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'qubit_pair_scan_phi{int(phi_deg)}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Saved to {output_file}")
    print(f"{'='*70}")

    print("\nRECOMMENDED LAYOUT FOR 12-QUBIT CIRCUIT (top 6 pairs):")
    layout = []
    for r in results[:6]:
        layout.extend(r['qubits'])
    print(f"  initial_layout = {layout}")

    print("\nBest pairs:")
    for i, r in enumerate(results[:6]):
        q1, q2 = r['qubits']
        print(f"  Pair {i+1}: ({q1},{q2}) - avg error: {r['avg_error']:.4f}")
