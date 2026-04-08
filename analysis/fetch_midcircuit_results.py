#!/usr/bin/env python3
"""
Fetch and process results from existing IBM job IDs for midcircuit Leggett tests.
"""

import numpy as np
import json
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

# Job IDs from the runs
JOBS = {
    (30, 29): 'd4qahjjher1c73bb39sg',
    (86, 87): 'd4qahjkfitbs739gq5ng',
    (110, 111): 'd4qahjc5fjns73d07740',
    (13, 14): 'd4qahjbher1c73bb39rg',
    (112, 113): 'd4qahjc5fjns73d0774g',
    (153, 152): 'd4qahjcfitbs739gq5n0',
}

ANGLE_DEG = 30
NUM_SHOTS = 1000


def extract_correlations_from_midcircuit_counts(counts, num_shots, num_angles=1):
    """Extract correlation values from mid-circuit measurement counts."""
    total_correlations = num_angles * 6
    correlations = [0.0] * total_correlations

    for outcome_str, count in counts.items():
        creg_results = outcome_str.split()
        for i in range(total_correlations):
            bits = creg_results[total_correlations - 1 - i]
            alice_bit = int(bits[1])
            bob_bit = int(bits[0])
            if alice_bit == bob_bit:
                correlations[i] += count
            else:
                correlations[i] -= count

    correlations = [c / num_shots for c in correlations]

    # Always return list of lists
    angle_results = []
    for angle_idx in range(num_angles):
        start_idx = angle_idx * 6
        angle_corrs = correlations[start_idx:start_idx + 6]
        angle_results.append(angle_corrs)
    return angle_results


def process_job_result(job, qubit_pair, angle_deg, num_shots):
    """Process a single job result."""
    result = job.result()
    pub_result = result[0]

    # Extract counts from classical registers
    counts = {}
    total_cregs = 6  # 6 correlations for single angle

    creg_int_arrays = []
    for creg_idx in range(total_cregs):
        creg_name = f'c{creg_idx}'
        bit_array = getattr(pub_result.data, creg_name)
        int_array = bit_array.array.reshape(-1)
        creg_int_arrays.append(int_array)

    # Reconstruct measurement strings
    for shot_idx in range(num_shots):
        outcome_parts = []
        for creg_idx in range(total_cregs):
            bit_val = int(creg_int_arrays[creg_idx][shot_idx])
            outcome_parts.append(f"{bit_val:02b}")
        outcome_str = " ".join(reversed(outcome_parts))
        counts[outcome_str] = counts.get(outcome_str, 0) + 1

    # Extract correlations
    all_angle_correlations = extract_correlations_from_midcircuit_counts(counts, num_shots, 1)
    correlations = all_angle_correlations[0]

    # Calculate theoretical values
    phi_rad = np.radians(angle_deg)
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    correlations_theory = [
        -np.dot(a1, b1), -np.dot(a1, b1_prime),
        -np.dot(a2, b2), -np.dot(a2, b2_prime),
        -np.dot(a3, b3), -np.dot(a3, b3_prime)
    ]

    # Calculate L3
    L3_exp = (1/3) * (abs(correlations[0] + correlations[1]) +
                      abs(correlations[2] + correlations[3]) +
                      abs(correlations[4] + correlations[5]))

    L3_th = (1/3) * (abs(correlations_theory[0] + correlations_theory[1]) +
                     abs(correlations_theory[2] + correlations_theory[3]) +
                     abs(correlations_theory[4] + correlations_theory[5]))

    L_bound = 2 - (2/3) * abs(np.sin(phi_rad/2))
    violated = L3_exp > L_bound

    return {
        'qubit_pair': list(qubit_pair),
        'phi_deg': float(angle_deg),
        'correlations': [float(c) for c in correlations],
        'correlations_theory': [float(c) for c in correlations_theory],
        'L3_exp': float(L3_exp),
        'L3_theory': float(L3_th),
        'bound': float(L_bound),
        'violated': bool(violated)
    }


def main():
    service = QiskitRuntimeService()

    all_results = []

    print("="*70)
    print("FETCHING MIDCIRCUIT LEGGETT RESULTS FROM IBM")
    print("="*70)

    for qubit_pair, job_id in JOBS.items():
        print(f"\nFetching job {job_id} for qubits {qubit_pair}...")
        job = service.job(job_id)

        try:
            result = process_job_result(job, qubit_pair, ANGLE_DEG, NUM_SHOTS)
            all_results.append(result)

            print(f"  Qubits {qubit_pair}: L3={result['L3_exp']:.4f}, bound={result['bound']:.4f}, violated={result['violated']}")
            print(f"    Correlations: {[f'{c:.3f}' for c in result['correlations']]}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'leggett_midcircuit_phi{ANGLE_DEG}_best_pairs_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Summary
    print("\nSUMMARY:")
    print("-"*70)
    for r in all_results:
        status = "VIOLATED" if r['violated'] else "not violated"
        print(f"  Qubits {r['qubit_pair']}: L3={r['L3_exp']:.4f} vs bound={r['bound']:.4f} - {status}")


if __name__ == "__main__":
    main()
