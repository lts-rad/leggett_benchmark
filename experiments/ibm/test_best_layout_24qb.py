import os
#!/usr/bin/env python3
"""
Test the best layout found from iteration with a single 24-qubit circuit.
"""

import numpy as np
import json
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle, extract_correlations_from_counts, calc_leggett_for_angle

# Best layout from iteration
BEST_LAYOUT = [67, 68, 29, 30, 54, 55, 93, 92, 109, 110, 2, 3,
               13, 14, 112, 113, 87, 86, 20, 21, 94, 95, 107, 97]

def main():
    print("="*70)
    print("TESTING BEST LAYOUT WITH 24-QUBIT CIRCUIT")
    print("="*70)
    print(f"\nLayout: {BEST_LAYOUT}")
    print(f"\nPairs:")
    for i in range(0, 24, 2):
        pair_num = i // 2
        phi = "+30°" if pair_num < 6 else "-30°"
        print(f"  Pair {pair_num:2d} ({phi}): ({BEST_LAYOUT[i]:3d}, {BEST_LAYOUT[i+1]:3d})")

    # Setup backend and noise model
    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)

    # Run test
    phi_deg = 30
    phi_rad = np.radians(phi_deg)
    num_shots = 1000

    print(f"\nRunning 24-qubit simulation with {num_shots} shots...")

    # Create single 24-qubit circuit (measures both +phi and -phi simultaneously)
    qc = create_leggett_circuit_for_angle(phi_rad)
    print(f"  Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    qc_transpiled = transpile(
        qc,
        backend=backend,
        optimization_level=2,
        initial_layout=BEST_LAYOUT
    )
    print(f"  Transpiled: depth {qc_transpiled.depth()}")

    # Run simulation
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc_transpiled, shots=num_shots).result()
    counts = result.get_counts()

    # Extract correlations for both +phi and -phi (24-qubit circuit measures both simultaneously)
    correlations_pos, correlations_neg = extract_correlations_from_counts(counts, num_shots)

    # Calculate L3 for both +phi and -phi
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
    result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    labels = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]

    print(f"\nφ = +30°:")
    print(f"  L₃ = {result_pos['L3']:.4f}")
    print(f"  Correlations (measured vs theory):")
    errors_pos = []
    for i, (label, meas, theory) in enumerate(zip(labels, result_pos['correlations'], result_pos['correlations_theory'])):
        error = meas - theory
        errors_pos.append(abs(error))
        print(f"    {label:12s}: {meas:+.4f} vs {theory:+.4f}  (err: {error:+.4f})")
    print(f"  Max error: {max(errors_pos):.4f}")
    print(f"  Mean error: {np.mean(errors_pos):.4f}")

    print(f"\nφ = -30°:")
    print(f"  L₃ = {result_neg['L3']:.4f}")
    print(f"  Correlations (measured vs theory):")
    errors_neg = []
    for i, (label, meas, theory) in enumerate(zip(labels, result_neg['correlations'], result_neg['correlations_theory'])):
        error = meas - theory
        errors_neg.append(abs(error))
        print(f"    {label:12s}: {meas:+.4f} vs {theory:+.4f}  (err: {error:+.4f})")
    print(f"  Max error: {max(errors_neg):.4f}")
    print(f"  Mean error: {np.mean(errors_neg):.4f}")

    print(f"\n{'='*70}")
    print(f"OVERALL: Max error = {max(max(errors_pos), max(errors_neg)):.4f}")
    print(f"         Mean error = {(np.mean(errors_pos) + np.mean(errors_neg))/2:.4f}")
    print(f"{'='*70}")

    # Save results
    output = {
        'layout': BEST_LAYOUT,
        'phi_deg': phi_deg,
        'result_pos': result_pos,
        'result_neg': result_neg,
        'errors_pos': errors_pos,
        'errors_neg': errors_neg,
        'max_error': max(max(errors_pos), max(errors_neg)),
        'mean_error': (np.mean(errors_pos) + np.mean(errors_neg))/2,
    }

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'best_layout_24qb_result.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved results to {output_file}")

if __name__ == "__main__":
    main()
