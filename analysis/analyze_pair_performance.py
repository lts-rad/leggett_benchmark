#!/usr/bin/env python3
"""
Analyze which qubit pairs are performing well vs poorly in the 24qb test.
"""

import json

with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT.json', 'r') as f:
    results = json.load(f)

labels = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]

# Find φ=+30 and φ=-30 results
for r in results:
    if abs(r['phi_deg'] - 30) < 0.1:
        print('φ = +30°:')
        print('Correlations (measured):')
        pairs_pos = [(67,68), (29,30), (54,55), (93,92), (109,110), (2,3)]
        for i, (label, pair, meas, theory) in enumerate(zip(labels, pairs_pos, r['correlations'], r['correlations_theory'])):
            error = meas - theory
            abs_error = abs(error)
            status = '✓ GOOD' if abs_error < 0.015 else '✗ BAD'
            print(f'  {i}: {label:12s} qubits {pair[0]:3d},{pair[1]:3d}  meas={meas:+.4f}  theory={theory:+.4f}  error={error:+.4f}  {status}')
        print()
    elif abs(r['phi_deg'] + 30) < 0.1:
        print('φ = -30°:')
        print('Correlations (measured):')
        pairs_neg = [(13,14), (112,113), (153,152), (75,74), (94,95), (107,97)]
        for i, (label, pair, meas, theory) in enumerate(zip(labels, pairs_neg, r['correlations'], r['correlations_theory'])):
            error = meas - theory
            abs_error = abs(error)
            status = '✓ GOOD' if abs_error < 0.015 else '✗ BAD'
            print(f'  {i}: {label:12s} qubits {pair[0]:3d},{pair[1]:3d}  meas={meas:+.4f}  theory={theory:+.4f}  error={error:+.4f}  {status}')
        print()

print("\nSUMMARY:")
print("Keep these good performers (error < 0.015):")
print("Need to replace bad performers (error >= 0.015):")
