import os
#!/usr/bin/env python3
"""
Test different qubit layouts by keeping good performers and trying replacements.
Run actual 24qb simulations with noise model to find best combination.
"""

import numpy as np
import json
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, '/Users/adc/qsim/relative_phase_variant/geometry/simulator_gnuradio/paper')
from leggett import create_leggett_circuit_twelve, extract_correlations_from_counts_twelve, calc_leggett_for_angle

# Good performers to keep (from analysis)
KEEP_PAIRS_POS = [
    (29, 30),   # a1b1' - error +0.0179
    (93, 92),   # a2b2' - error +0.0199
]

KEEP_PAIRS_NEG = [
    (112, 113), # a1b1' - error +0.0179
    (107, 97),  # a3b3' - error +0.0179
    (94, 95),   # a3b3  - error +0.0279
]

# Positions in layout:
# φ+30: [a1b1, a1b1', a2b2, a2b2', a3b3, a3b3']  <- indices 0-11
# φ-30: [a1b1, a1b1', a2b2, a2b2', a3b3, a3b3']  <- indices 12-23

def test_layout(layout_flat, backend, noise_model, phi_deg=30, num_shots=5000):
    """
    Test a specific 24-qubit layout.

    Returns dict with results for both +phi and -phi.
    """
    phi_rad = np.radians(phi_deg)

    # Create circuit
    qc = create_leggett_circuit_twelve(phi_rad)

    # Transpile
    qc_transpiled = transpile(
        qc,
        backend=backend,
        optimization_level=2,
        initial_layout=layout_flat
    )

    # Run simulation
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc_transpiled, shots=num_shots).result()
    counts = result.get_counts()

    # Extract correlations
    correlations = extract_correlations_from_counts_twelve(counts, num_shots)

    # Calculate L3 for both +phi and -phi
    result_pos = calc_leggett_for_angle(correlations[:6], phi_rad)
    result_neg = calc_leggett_for_angle(correlations[6:], phi_rad)

    # Calculate errors
    errors_pos = [abs(m - t) for m, t in zip(result_pos['correlations'], result_pos['correlations_theory'])]
    errors_neg = [abs(m - t) for m, t in zip(result_neg['correlations'], result_neg['correlations_theory'])]

    return {
        'layout': layout_flat,
        'phi_deg': phi_deg,
        'result_pos': result_pos,
        'result_neg': result_neg,
        'errors_pos': errors_pos,
        'errors_neg': errors_neg,
        'max_error_pos': max(errors_pos),
        'max_error_neg': max(errors_neg),
        'mean_error_pos': np.mean(errors_pos),
        'mean_error_neg': np.mean(errors_neg),
        'L3_pos': result_pos['L3'],
        'L3_neg': result_neg['L3'],
    }

def create_test_layout(replacement_pairs_pos, replacement_pairs_neg):
    """
    Create a 24-qubit layout with specific replacements.

    φ+30: [a1b1, a1b1', a2b2, a2b2', a3b3, a3b3']  <- use replacement_pairs_pos
    φ-30: [a1b1, a1b1', a2b2, a2b2', a3b3, a3b3']  <- use replacement_pairs_neg

    Keep good performers in their positions.
    """
    # Initialize with placeholders
    layout_pos = [None] * 6  # 6 pairs for +30
    layout_neg = [None] * 6  # 6 pairs for -30

    # Fill in the kept pairs at correct positions
    # φ+30:
    # Position 1 (a1b1'): (29,30)
    # Position 3 (a2b2'): (93,92)
    layout_pos[1] = KEEP_PAIRS_POS[0]  # (29,30) at a1b1'
    layout_pos[3] = KEEP_PAIRS_POS[1]  # (93,92) at a2b2'

    # φ-30:
    # Position 1 (a1b1'): (112,113)
    # Position 4 (a3b3):  (94,95)
    # Position 5 (a3b3'): (107,97)
    layout_neg[1] = KEEP_PAIRS_NEG[0]  # (112,113) at a1b1'
    layout_neg[4] = KEEP_PAIRS_NEG[1]  # (94,95) at a3b3
    layout_neg[5] = KEEP_PAIRS_NEG[2]  # (107,97) at a3b3'

    # Fill in replacement pairs
    # φ+30 needs: a1b1 (0), a2b2 (2), a3b3 (4), a3b3' (5)
    pos_needed = [0, 2, 4, 5]
    for i, pos in enumerate(pos_needed):
        layout_pos[pos] = replacement_pairs_pos[i]

    # φ-30 needs: a1b1 (0), a2b2 (2), a2b2' (3)
    neg_needed = [0, 2, 3]
    for i, pos in enumerate(neg_needed):
        layout_neg[pos] = replacement_pairs_neg[i]

    # Flatten to 24 qubits
    layout_flat = []
    for pair in layout_pos:
        layout_flat.extend(pair)
    for pair in layout_neg:
        layout_flat.extend(pair)

    return layout_flat

def main():
    # Load viable pairs
    with open('viable_pairs_analysis.json', 'r') as f:
        analysis = json.load(f)

    # Get list of candidate pairs excluding those we're keeping
    used_qubits = set()
    for pair in KEEP_PAIRS_POS + KEEP_PAIRS_NEG:
        used_qubits.update(pair)

    candidate_pairs = []
    for pair_data in analysis['viable_pairs']:
        q1, q2 = pair_data['qubits']
        if q1 not in used_qubits and q2 not in used_qubits:
            candidate_pairs.append(tuple(pair_data['qubits']))

    print("="*70)
    print("TESTING REPLACEMENT LAYOUTS")
    print("="*70)
    print(f"\nKept pairs (+30): {KEEP_PAIRS_POS}")
    print(f"Kept pairs (-30): {KEEP_PAIRS_NEG}")
    print(f"\nAvailable candidate pairs: {len(candidate_pairs)}")
    print(f"\nNeed 4 replacements for +30° and 3 for -30°")

    # Setup backend
    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)

    # Test with top candidates
    print(f"\n{'='*70}")
    print("TESTING LAYOUT WITH TOP CANDIDATES")
    print(f"{'='*70}")

    # Use top 7 candidates (4 for +30, 3 for -30)
    replacement_pos = candidate_pairs[:4]  # Top 4 for +30
    replacement_neg = candidate_pairs[4:7]  # Next 3 for -30

    print(f"\nReplacements for +30° (a1b1, a2b2, a3b3, a3b3'): {replacement_pos}")
    print(f"Replacements for -30° (a1b1, a2b2, a2b2'): {replacement_neg}")

    layout = create_test_layout(replacement_pos, replacement_neg)
    print(f"\nFull layout (24 qubits): {layout}")

    print(f"\nRunning simulation...")
    result = test_layout(layout, backend, noise_model, phi_deg=30, num_shots=5000)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nφ = +30°:")
    print(f"  L₃ = {result['L3_pos']:.4f}")
    print(f"  Max error = {result['max_error_pos']:.4f}")
    print(f"  Mean error = {result['mean_error_pos']:.4f}")
    print(f"  Errors: {[f'{e:.4f}' for e in result['errors_pos']]}")

    print(f"\nφ = -30°:")
    print(f"  L₃ = {result['L3_neg']:.4f}")
    print(f"  Max error = {result['max_error_neg']:.4f}")
    print(f"  Mean error = {result['mean_error_neg']:.4f}")
    print(f"  Errors: {[f'{e:.4f}' for e in result['errors_neg']]}")

    # Save result
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'test_replacement_layout_result.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nSaved results to {output_file}")

if __name__ == "__main__":
    main()
