#!/usr/bin/env python3
"""
Test different combinations of top pairs to find optimal 6 for Leggett test.
Uses noise model to evaluate full-circuit performance.
"""

import numpy as np
import json
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle_six, extract_correlations_from_counts_six, calc_leggett_for_angle

# Load scan data
with open('qubit_pair_scan_phi30.json', 'r') as f:
    scan_data = json.load(f)

# Sort by avg_error to get top pairs
scan_data_sorted = sorted(scan_data, key=lambda x: x['avg_error'])

# Define test combinations
print("="*70)
print("TESTING TOP PAIR COMBINATIONS")
print("="*70)
print("\nTop 20 pairs from scan:")
for i, entry in enumerate(scan_data_sorted[:20]):
    q = entry['qubits']
    print(f"{i+1:2d}. ({q[0]:3d},{q[1]:3d})  error={entry['avg_error']:.6f}  corr={entry['correlations']}")

# Test combinations
combinations = [
    ("Top 1-6", [(30,29), (86,87), (87,86), (110,111), (13,14), (112,113)]),
    ("Top 1-6 alt", [(30,29), (86,87), (110,111), (13,14), (112,113), (153,152)]),  # Current best
    ("Top diversified", [(30,29), (86,87), (110,111), (126,127), (2,3), (54,55)]),
    ("Top no-overlap", [(30,29), (86,87), (13,14), (112,113), (126,127), (2,3)]),
    ("Top 7-12", [(126,127), (2,3), (125,126), (54,55), (17,27), (68,67)]),
]

# Get IBM backend and noise model
print("\n" + "="*70)
print("Loading IBM Pittsburgh noise model...")
print("="*70)
service = QiskitRuntimeService()
backend = service.backend("ibm_pittsburgh")
noise_model = NoiseModel.from_backend(backend)

# Test each combination
phi_rad = np.radians(30)
num_shots = 1000

results = []

for name, pairs in combinations:
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"Pairs: {pairs}")
    print(f"{'='*70}")

    # Check for overlapping qubits
    all_qubits = []
    for q0, q1 in pairs:
        if q0 in all_qubits or q1 in all_qubits:
            print(f"WARNING: Overlapping qubits detected! Skipping this combination.")
            continue
        all_qubits.extend([q0, q1])

    if len(all_qubits) != 12:
        print(f"ERROR: Need exactly 12 qubits, got {len(all_qubits)}")
        continue

    # Create circuit
    qc = create_leggett_circuit_for_angle_six(phi_rad)

    # Transpile with this layout
    print(f"  Transpiling with layout: {all_qubits}")
    qc_transpiled = transpile(qc, backend=backend, optimization_level=3, initial_layout=all_qubits)

    # Run with noise model
    print(f"  Running noise simulation...")
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc_transpiled, shots=num_shots).result()
    counts = result.get_counts()

    # Extract correlations
    correlations = extract_correlations_from_counts_six(counts, num_shots)

    # Calculate L3
    leggett_result = calc_leggett_for_angle(correlations, phi_rad)

    L3 = leggett_result['L3']
    L3_theory = leggett_result['L3_theory']
    bound = leggett_result['bound']
    violated = leggett_result['violated']

    # Calculate individual errors
    corr_errors = [abs(c - (-0.9659)) for c in correlations]
    avg_error = np.mean(corr_errors)
    max_error = np.max(corr_errors)

    print(f"\n  Results:")
    print(f"    Correlations: {[f'{c:.4f}' for c in correlations]}")
    print(f"    L₃ (exp):     {L3:.4f}")
    print(f"    L₃ (theory):  {L3_theory:.4f}")
    print(f"    Bound:        {bound:.4f}")
    print(f"    Violated:     {violated}")
    print(f"    Margin:       {L3 - bound:+.4f} ({100*(L3-bound)/bound:+.2f}%)")
    print(f"    Avg error:    {avg_error:.4f}")
    print(f"    Max error:    {max_error:.4f}")

    results.append({
        'name': name,
        'pairs': pairs,
        'L3': L3,
        'L3_theory': L3_theory,
        'bound': bound,
        'margin': L3 - bound,
        'margin_pct': 100*(L3-bound)/bound,
        'violated': violated,
        'correlations': correlations,
        'avg_error': avg_error,
        'max_error': max_error
    })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n{'Combination':<20} {'L₃':>8} {'Margin':>8} {'Margin %':>10} {'Avg Err':>10} {'Max Err':>10}")
print("-"*70)

for r in sorted(results, key=lambda x: x['L3'], reverse=True):
    print(f"{r['name']:<20} {r['L3']:>8.4f} {r['margin']:>+8.4f} {r['margin_pct']:>+9.2f}% {r['avg_error']:>10.4f} {r['max_error']:>10.4f}")

# Save results
with open('top_pair_combinations_test.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to top_pair_combinations_test.json")

# Find best combination
best = max(results, key=lambda x: x['L3'])
print(f"\n{'='*70}")
print(f"BEST COMBINATION: {best['name']}")
print(f"{'='*70}")
print(f"Pairs: {best['pairs']}")
print(f"L₃ = {best['L3']:.4f} (margin: {best['margin']:+.4f}, {best['margin_pct']:+.2f}%)")
print(f"Layout: {[q for pair in best['pairs'] for q in pair]}")
