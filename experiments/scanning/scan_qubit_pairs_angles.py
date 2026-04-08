import os
#!/usr/bin/env python3
"""
Scan individual qubit pairs with actual measurement angles (φ = ±30°).
Tests how close each pair gets to ideal QM correlation -a·b.
"""

import numpy as np
import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

def create_singlet_pair_with_angles(phi_a, phi_b):
    """
    Create 2-qubit singlet with measurement angles.

    Circuit creates |ψ⁻⟩ = (|01⟩ - |10⟩)/√2 then measures at angles φ_a, φ_b.
    """
    qc = QuantumCircuit(2, 2)

    # Create singlet state
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)

    # Rotate measurement bases
    qc.ry(-2*phi_a, 0)  # Alice measures at angle φ_a
    qc.ry(-2*phi_b, 1)  # Bob measures at angle φ_b

    qc.measure([0, 1], [0, 1])
    return qc

def test_qubit_pair_angles(q1, q2, backend, noise_model, phi_deg=30, num_shots=1000):
    """
    Test a qubit pair with angles φ = +phi_deg and φ = -phi_deg.

    Returns average error compared to ideal -cos(2φ).
    """
    phi_rad = np.radians(phi_deg)

    # Test both (+φ, 0) and (-φ, 0)
    # For singlet: C(φ_a, φ_b) = -cos(φ_a - φ_b)
    # C(+φ, 0) = -cos(φ)
    # C(-φ, 0) = -cos(-φ) = -cos(φ)

    errors = []
    correlations = []

    for angle in [phi_rad, -phi_rad]:
        qc = create_singlet_pair_with_angles(angle, 0)

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
        # C = (N_same - N_diff) / N_total
        correlation = 0.0
        for bitstring, count in counts.items():
            alice_bit = int(bitstring[1])  # q0
            bob_bit = int(bitstring[0])    # q1

            if alice_bit == bob_bit:
                correlation += count
            else:
                correlation -= count

        correlation /= num_shots

        # Theoretical: C(angle, 0) = -cos(angle)
        correlation_theory = -np.cos(angle)
        error = abs(correlation - correlation_theory)

        errors.append(error)
        correlations.append(correlation)

    # Return average error
    avg_error = np.mean(errors)
    return avg_error, correlations, errors

def scan_coupled_pairs(backend, noise_model, num_shots=1000, blacklist=None, phi_deg=30):
    """
    Scan ALL coupled qubit pairs with angle measurements.
    """
    if blacklist is None:
        blacklist = set()
    else:
        blacklist = set(blacklist)

    # Get coupling map edges
    coupling_map = backend.coupling_map
    edges = coupling_map.get_edges()

    # Filter out blacklisted qubits
    valid_edges = [
        (q1, q2) for q1, q2 in edges
        if q1 not in blacklist and q2 not in blacklist
    ]

    print(f"Testing ALL {len(valid_edges)} qubit pairs with φ = ±{phi_deg}°...")
    print(f"Blacklist: {sorted(blacklist)}")
    print(f"Theoretical correlation: -cos({phi_deg}°) = {-np.cos(np.radians(phi_deg)):.4f}")
    print()

    results = []

    for i, (q1, q2) in enumerate(valid_edges):
        print(f"Testing pair {i+1}/{len(valid_edges)}: qubits ({q1},{q2})... ", end='', flush=True)

        try:
            avg_error, correlations, errors = test_qubit_pair_angles(
                q1, q2, backend, noise_model, phi_deg, num_shots
            )
            results.append({
                'qubits': [q1, q2],
                'avg_error': avg_error,
                'correlations': correlations,
                'errors': errors,
                'phi_deg': phi_deg
            })
            print(f"avg_err={avg_error:.4f}, corr=[{correlations[0]:+.3f}, {correlations[1]:+.3f}]")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    return results

if __name__ == "__main__":
    import sys

    # Parse arguments
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
    print("SCANNING QUBIT PAIRS WITH MEASUREMENT ANGLES")
    print("="*70)
    print(f"Shots per pair: {num_shots}")
    print(f"Angle: φ = ±{phi_deg}°")
    print()

    # Setup
    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)

    # Scan ALL pairs
    results = scan_coupled_pairs(backend, noise_model, num_shots, blacklist, phi_deg)

    # Sort by error (lower is better)
    results.sort(key=lambda x: x['avg_error'])

    # Print top results
    print("\n" + "="*70)
    print(f"TOP 20 BEST QUBIT PAIRS (by avg error at φ = ±{phi_deg}°)")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Avg Error':>12} {'Corr(+φ)':>11} {'Corr(-φ)':>11}")
    print("-"*70)

    for i, result in enumerate(results[:20]):
        q1, q2 = result['qubits']
        print(f"{i+1:>4} ({q1:3d},{q2:3d})     {result['avg_error']:>12.4f} {result['correlations'][0]:>+11.4f} {result['correlations'][1]:>+11.4f}")

    # Print worst results
    print("\n" + "="*70)
    print("WORST 10 QUBIT PAIRS")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Avg Error':>12} {'Corr(+φ)':>11} {'Corr(-φ)':>11}")
    print("-"*70)

    for i, result in enumerate(results[-10:]):
        q1, q2 = result['qubits']
        rank = len(results) - 10 + i + 1
        print(f"{rank:>4} ({q1:3d},{q2:3d})     {result['avg_error']:>12.4f} {result['correlations'][0]:>+11.4f} {result['correlations'][1]:>+11.4f}")

    # Save results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'qubit_pair_scan_phi{int(phi_deg)}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Saved all results to {output_file}")
    print(f"{'='*70}")

    # Print recommended pairs for 12-qubit circuit
    print("\nRECOMMENDED LAYOUT FOR 12-QUBIT CIRCUIT (top 6 pairs):")
    best_6_pairs = results[:6]
    layout = []
    for pair in best_6_pairs:
        layout.extend(pair['qubits'])
    print(f"  initial_layout = {layout}")

    print("\nBest pairs summary:")
    for i, result in enumerate(best_6_pairs):
        q1, q2 = result['qubits']
        print(f"  Pair {i+1}: ({q1},{q2}) - avg error: {result['avg_error']:.4f}")
