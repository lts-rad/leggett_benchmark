#!/usr/bin/env python3
"""
Keep the good performers and iterate through candidates for other positions.
Simulate each combination with noise model.
"""

import numpy as np
import json
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import itertools
from leggett import create_leggett_circuit_for_angle_six, extract_correlations_from_counts_six, calc_leggett_for_angle

# Fixed good performers (from your analysis)
# φ+30: Keep a1b1'=(29,30) and a2b2'=(93,92)
# φ-30: Keep a1b1'=(112,113), a3b3=(94,95), a3b3'=(107,97)

FIXED_POS = {
    1: (29, 30),   # a1b1'
    3: (93, 92),   # a2b2'
}

FIXED_NEG = {
    1: (112, 113), # a1b1'
    4: (94, 95),   # a3b3
    5: (107, 97),  # a3b3'
}

def test_layout(pairs_pos, pairs_neg, backend, noise_model, phi_deg=30, num_shots=5000):
    """Test layouts by running +phi and -phi separately (each 12 qubits)."""
    phi_rad = np.radians(phi_deg)
    simulator = AerSimulator(noise_model=noise_model)

    # Build +phi layout (12 qubits)
    layout_pos = []
    for pair in pairs_pos:
        layout_pos.extend(pair)

    # Run +phi
    qc_pos = create_leggett_circuit_for_angle_six(phi_rad)
    qc_pos_transpiled = transpile(qc_pos, backend=backend, optimization_level=2, initial_layout=layout_pos)
    result_pos_run = simulator.run(qc_pos_transpiled, shots=num_shots).result()
    counts_pos = result_pos_run.get_counts()
    correlations_pos = extract_correlations_from_counts_six(counts_pos, num_shots)
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)

    # Build -phi layout (12 qubits)
    layout_neg = []
    for pair in pairs_neg:
        layout_neg.extend(pair)

    # Run -phi
    qc_neg = create_leggett_circuit_for_angle_six(-phi_rad)
    qc_neg_transpiled = transpile(qc_neg, backend=backend, optimization_level=2, initial_layout=layout_neg)
    result_neg_run = simulator.run(qc_neg_transpiled, shots=num_shots).result()
    counts_neg = result_neg_run.get_counts()
    correlations_neg = extract_correlations_from_counts_six(counts_neg, num_shots)
    result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

    errors_pos = [abs(m - t) for m, t in zip(result_pos['correlations'], result_pos['correlations_theory'])]
    errors_neg = [abs(m - t) for m, t in zip(result_neg['correlations'], result_neg['correlations_theory'])]

    return {
        'pairs_pos': pairs_pos,
        'pairs_neg': pairs_neg,
        'result_pos': result_pos,
        'result_neg': result_neg,
        'errors_pos': errors_pos,
        'errors_neg': errors_neg,
        'max_error': max(max(errors_pos), max(errors_neg)),
        'mean_error': (np.mean(errors_pos) + np.mean(errors_neg)) / 2,
        'L3_pos': result_pos['L3'],
        'L3_neg': result_neg['L3'],
    }

def build_layout(pairs_pos, pairs_neg):
    """Build flat 24-qubit layout from 6+6 pairs."""
    layout = []
    for pair in pairs_pos:
        layout.extend(pair)
    for pair in pairs_neg:
        layout.extend(pair)
    return layout

def main():
    # Load candidates
    with open('viable_pairs_analysis.json', 'r') as f:
        analysis = json.load(f)

    # Get qubits used in fixed pairs
    fixed_qubits = set()
    for pair in list(FIXED_POS.values()) + list(FIXED_NEG.values()):
        fixed_qubits.update(pair)

    # Get candidate pairs that don't use fixed qubits
    candidates = []
    for pair_data in analysis['viable_pairs']:
        q1, q2 = pair_data['qubits']
        if q1 not in fixed_qubits and q2 not in fixed_qubits:
            candidates.append((q1, q2))

    print("="*70)
    print("ITERATING THROUGH LAYOUT COMBINATIONS")
    print("="*70)
    print(f"\nFixed pairs for +30°:")
    for pos, pair in FIXED_POS.items():
        labels = ['a1b1', "a1b1'", 'a2b2', "a2b2'", 'a3b3', "a3b3'"]
        print(f"  Position {pos} ({labels[pos]}): {pair}")

    print(f"\nFixed pairs for -30°:")
    for pos, pair in FIXED_NEG.items():
        labels = ['a1b1', "a1b1'", 'a2b2', "a2b2'", 'a3b3', "a3b3'"]
        print(f"  Position {pos} ({labels[pos]}): {pair}")

    print(f"\nAvailable candidate pairs: {len(candidates)}")
    print(f"\nNeed to fill:")
    print(f"  +30°: positions 0,2,4,5 (a1b1, a2b2, a3b3, a3b3')")
    print(f"  -30°: positions 0,2,3 (a1b1, a2b2, a2b2')")

    # Setup backend
    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)

    # Test top N combinations
    N_combinations = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"\nTesting top {N_combinations} combinations...")
    print("="*70)

    results = []

    # Use top candidates for iteration
    top_candidates = candidates[:20]  # Use top 20 for combinations

    tested = 0
    for combo_pos in itertools.combinations(top_candidates, 4):  # 4 for +30
        for combo_neg in itertools.combinations(top_candidates, 3):  # 3 for -30
            # Check no overlap between pos and neg
            used_pos = set()
            for pair in combo_pos:
                used_pos.update(pair)
            used_neg = set()
            for pair in combo_neg:
                used_neg.update(pair)

            if used_pos & used_neg:  # Overlap
                continue

            # Build full pairs lists
            pairs_pos = [None] * 6
            pairs_neg = [None] * 6

            # Fill fixed
            for pos, pair in FIXED_POS.items():
                pairs_pos[pos] = pair
            for pos, pair in FIXED_NEG.items():
                pairs_neg[pos] = pair

            # Fill variable positions
            pairs_pos[0] = combo_pos[0]  # a1b1
            pairs_pos[2] = combo_pos[1]  # a2b2
            pairs_pos[4] = combo_pos[2]  # a3b3
            pairs_pos[5] = combo_pos[3]  # a3b3'

            pairs_neg[0] = combo_neg[0]  # a1b1
            pairs_neg[2] = combo_neg[1]  # a2b2
            pairs_neg[3] = combo_neg[2]  # a2b2'

            layout = build_layout(pairs_pos, pairs_neg)

            print(f"\nTest {tested+1}:")
            print(f"  +30 replacements: {combo_pos}")
            print(f"  -30 replacements: {combo_neg}")

            try:
                result = test_layout(pairs_pos, pairs_neg, backend, noise_model, phi_deg=30, num_shots=5000)
                results.append({
                    'combo_pos': combo_pos,
                    'combo_neg': combo_neg,
                    'layout': layout,
                    'L3_pos': result['L3_pos'],
                    'L3_neg': result['L3_neg'],
                    'max_error': result['max_error'],
                    'mean_error': result['mean_error'],
                })
                print(f"  L3(+)={result['L3_pos']:.4f}, L3(-)={result['L3_neg']:.4f}, max_err={result['max_error']:.4f}")

                tested += 1
                if tested >= N_combinations:
                    break
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

        if tested >= N_combinations:
            break

    # Sort by max error
    results.sort(key=lambda x: x['max_error'])

    print(f"\n{'='*70}")
    print("BEST COMBINATIONS")
    print(f"{'='*70}")

    for i, r in enumerate(results[:5]):
        print(f"\nRank {i+1}:")
        print(f"  L3(+)={r['L3_pos']:.4f}, L3(-)={r['L3_neg']:.4f}")
        print(f"  Max error: {r['max_error']:.4f}, Mean error: {r['mean_error']:.4f}")
        print(f"  Layout: {r['layout']}")

    # Save
    with open('layout_iteration_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved {len(results)} results to layout_iteration_results.json")

if __name__ == "__main__":
    main()
