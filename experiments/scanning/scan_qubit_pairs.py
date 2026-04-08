import os
#!/usr/bin/env python3
"""
Scan individual qubit pairs to find the best ones for singlet correlations.
Just tests 2-qubit Bell pairs, much more efficient.
"""

import numpy as np
import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

def create_singlet_pair():
    """Create a simple 2-qubit Bell state."""
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def test_qubit_pair(q1, q2, backend, noise_model, num_shots=1000):
    """
    Test a specific qubit pair for singlet correlation.

    Returns correlation value (should be close to -1.0 for good qubits).
    """
    qc = create_singlet_pair()

    # Transpile with this specific pair
    qc_transpiled = transpile(
        qc,
        backend=backend,
        optimization_level=2,
        initial_layout=[q1, q2]
    )

    # Run simulation with noise
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc_transpiled, shots=num_shots).result()
    counts = result.get_counts()

    # Calculate correlation: C = P(00) + P(11) - P(01) - P(10)
    total = num_shots
    p00 = counts.get('00', 0) / total
    p11 = counts.get('11', 0) / total
    p01 = counts.get('01', 0) / total
    p10 = counts.get('10', 0) / total

    correlation = p00 + p11 - p01 - p10

    # For |Φ⁻⟩ = (|00⟩ - |11⟩)/√2, theoretical correlation is +1.0
    error = abs(correlation - (+1.0))

    return correlation, error

def scan_coupled_pairs(backend, noise_model, num_shots=1000, blacklist=None):
    """
    Scan ALL coupled qubit pairs from the backend.

    Args:
        backend: IBM backend
        noise_model: Noise model
        num_shots: Shots per pair test
        blacklist: Set of qubits to skip
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

    print(f"Testing ALL {len(valid_edges)} qubit pairs...")
    print(f"Blacklist: {sorted(blacklist)}")
    print()

    results = []

    for i, (q1, q2) in enumerate(valid_edges):
        print(f"Testing pair {i+1}/{len(valid_edges)}: qubits ({q1},{q2})... ", end='', flush=True)

        try:
            correlation, error = test_qubit_pair(q1, q2, backend, noise_model, num_shots)
            results.append({
                'qubits': [q1, q2],
                'correlation': correlation,
                'error': error
            })
            print(f"C={correlation:+.4f}, error={error:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    return results

if __name__ == "__main__":
    import sys

    # Parse arguments
    num_shots = 500
    blacklist = []

    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--blacklist' and i + 1 < len(sys.argv):
            blacklist = [int(x) for x in sys.argv[i + 1].split(',')]

    print("="*70)
    print("SCANNING ALL QUBIT PAIRS FOR SINGLET CORRELATION QUALITY")
    print("="*70)
    print(f"Shots per pair: {num_shots}")
    print()

    # Setup
    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)

    # Scan ALL pairs
    results = scan_coupled_pairs(backend, noise_model, num_shots, blacklist)

    # Sort by error (lower is better)
    results.sort(key=lambda x: x['error'])

    # Print top results
    print("\n" + "="*70)
    print("TOP 20 BEST QUBIT PAIRS (by error from ideal -1.0)")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Correlation':>13} {'Error':>10}")
    print("-"*70)

    for i, result in enumerate(results[:20]):
        q1, q2 = result['qubits']
        print(f"{i+1:>4} ({q1:3d},{q2:3d})     {result['correlation']:+13.4f} {result['error']:>10.4f}")

    # Print worst results
    print("\n" + "="*70)
    print("WORST 10 QUBIT PAIRS")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Correlation':>13} {'Error':>10}")
    print("-"*70)

    for i, result in enumerate(results[-10:]):
        q1, q2 = result['qubits']
        rank = len(results) - 10 + i + 1
        print(f"{rank:>4} ({q1:3d},{q2:3d})     {result['correlation']:+13.4f} {result['error']:>10.4f}")

    # Save results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'qubit_pair_scan.json')
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
