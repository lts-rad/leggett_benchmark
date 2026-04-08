import os
#!/usr/bin/env python3
"""
Efficiently search for the best 6 qubit pairs on IBM Pittsburgh.
Tests multiple layouts and scores them by correlation errors.
"""

import numpy as np
import json
import random
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle_six, extract_correlations_from_counts_six, calc_leggett_for_angle

def score_layout(correlations, correlations_theory):
    """
    Score a layout by its errors. Lower is better.

    Returns:
        max_error: Maximum absolute error across all correlations
        mean_error: Mean absolute error
        std_error: Standard deviation of errors
    """
    errors = [abs(exp - th) for exp, th in zip(correlations, correlations_theory)]
    return max(errors), np.mean(errors), np.std(errors)

def test_layout(initial_layout, backend, noise_model, phi_rad=np.radians(15), num_shots=2000, verbose=False):
    """
    Test a specific qubit layout.

    Args:
        initial_layout: List of 12 physical qubit indices
        backend: IBM backend
        noise_model: Noise model from backend
        phi_rad: Test angle in radians
        num_shots: Number of shots
        verbose: Print details

    Returns:
        dict with layout, correlations, errors, and score
    """
    qc = create_leggett_circuit_for_angle_six(phi_rad)

    # Transpile with this layout
    qc_transpiled = transpile(
        qc,
        backend=backend,
        optimization_level=2,
        initial_layout=initial_layout
    )

    # Get actual physical qubits used
    layout = qc_transpiled.layout
    actual_qubits = []
    for logical_idx in range(qc.num_qubits):
        physical_qubit = layout.initial_layout._v2p[qc.qubits[logical_idx]]
        actual_qubits.append(physical_qubit)

    # Group into pairs
    qubit_pairs = [[actual_qubits[i], actual_qubits[i+1]] for i in range(0, 12, 2)]

    # Run simulation
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc_transpiled, shots=num_shots).result()
    counts = result.get_counts()

    # Extract correlations
    correlations = extract_correlations_from_counts_six(counts, num_shots)
    result_data = calc_leggett_for_angle(correlations, phi_rad)

    # Score
    max_err, mean_err, std_err = score_layout(
        result_data['correlations'],
        result_data['correlations_theory']
    )

    if verbose:
        print(f"  Pairs: {qubit_pairs}")
        print(f"  Max error: {max_err:.4f}, Mean error: {mean_err:.4f}")

    return {
        'initial_layout': initial_layout,
        'qubit_pairs': qubit_pairs,
        'correlations': result_data['correlations'],
        'correlations_theory': result_data['correlations_theory'],
        'max_error': max_err,
        'mean_error': mean_err,
        'std_error': std_err,
        'L3': result_data['L3'],
        'L3_theory': result_data['L3_theory']
    }

def get_coupled_qubits(backend, qubit):
    """Get qubits coupled to a given qubit."""
    coupling_map = backend.coupling_map
    edges = coupling_map.get_edges()
    coupled = []
    for edge in edges:
        if edge[0] == qubit:
            coupled.append(edge[1])
        elif edge[1] == qubit:
            coupled.append(edge[0])
    return coupled

def generate_random_layout(backend, blacklist=None):
    """
    Generate a random valid layout using coupled qubits.

    Returns list of 12 physical qubits forming 6 pairs.
    """
    if blacklist is None:
        blacklist = set()
    else:
        blacklist = set(blacklist)

    coupling_map = backend.coupling_map
    edges = coupling_map.get_edges()

    # Filter edges to remove blacklisted qubits
    valid_edges = [
        edge for edge in edges
        if edge[0] not in blacklist and edge[1] not in blacklist
    ]

    if len(valid_edges) < 6:
        return None

    # Randomly select 6 edges (pairs)
    selected_pairs = random.sample(valid_edges, 6)

    # Flatten to layout
    layout = []
    for q1, q2 in selected_pairs:
        layout.extend([q1, q2])

    return layout

def search_best_layouts(num_trials=100, num_shots=2000, blacklist=None, top_n=5):
    """
    Search for best layouts by random sampling.

    Args:
        num_trials: Number of random layouts to test
        num_shots: Shots per test
        blacklist: Qubits to avoid
        top_n: Number of top results to return
    """
    print("="*70)
    print(f"SEARCHING FOR BEST QUBIT LAYOUTS")
    print("="*70)
    print(f"Trials: {num_trials}")
    print(f"Shots per trial: {num_shots}")
    print(f"Blacklist: {blacklist}")
    print()

    # Setup
    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)
    phi_rad = np.radians(15)

    results = []

    for trial in range(num_trials):
        # Generate random layout
        layout = generate_random_layout(backend, blacklist=blacklist)

        if layout is None:
            print(f"Trial {trial+1}/{num_trials}: Could not generate valid layout")
            continue

        print(f"Trial {trial+1}/{num_trials}: Testing layout {layout[:6]}... ", end='', flush=True)

        try:
            result = test_layout(layout, backend, noise_model, phi_rad, num_shots, verbose=False)
            results.append(result)
            print(f"max_err={result['max_error']:.4f}, mean_err={result['mean_error']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Sort by max error (lower is better)
    results.sort(key=lambda x: x['max_error'])

    # Print summary
    print("\n" + "="*70)
    print(f"TOP {top_n} LAYOUTS (by max error)")
    print("="*70)

    corr_names = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]

    for i, result in enumerate(results[:top_n]):
        print(f"\n=== RANK {i+1} ===")
        print(f"Max error: {result['max_error']:.4f}")
        print(f"Mean error: {result['mean_error']:.4f}")
        print(f"Std error: {result['std_error']:.4f}")
        print(f"L₃: {result['L3']:.4f} (theory: {result['L3_theory']:.4f})")
        print(f"\nQubit pairs:")
        for j, (name, pair) in enumerate(zip(corr_names, result['qubit_pairs'])):
            err = abs(result['correlations'][j] - result['correlations_theory'][j])
            print(f"  {name:12s}: qubits {pair[0]:3d},{pair[1]:3d}  error={err:+.4f}")
        print(f"\nInitial layout: {result['initial_layout']}")

    return results[:top_n]

if __name__ == "__main__":
    import sys

    # Parse arguments
    num_trials = 100
    num_shots = 2000
    blacklist = []
    top_n = 10

    for i, arg in enumerate(sys.argv):
        if arg == '--trials' and i + 1 < len(sys.argv):
            num_trials = int(sys.argv[i + 1])
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--blacklist' and i + 1 < len(sys.argv):
            # Parse comma-separated list
            blacklist = [int(x) for x in sys.argv[i + 1].split(',')]
        elif arg == '--top' and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])

    # Run search
    top_results = search_best_layouts(
        num_trials=num_trials,
        num_shots=num_shots,
        blacklist=blacklist,
        top_n=top_n
    )

    # Save results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'best_qubit_layouts.json')
    with open(output_file, 'w') as f:
        json.dump(top_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Saved top {top_n} results to {output_file}")
    print(f"{'='*70}")
