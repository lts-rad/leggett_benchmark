#!/usr/bin/env python3
"""
Analyze 64-pair results by individual pair groups.

With 64 pairs cycling through 12 correlation types:
- Group 0: pairs 0-11 (complete L3 measurement)
- Group 1: pairs 12-23 (complete L3 measurement)
- Group 2: pairs 24-35 (complete L3 measurement)
- Group 3: pairs 36-47 (complete L3 measurement)
- Group 4: pairs 48-59 (complete L3 measurement)
- Partial: pairs 60-63 (incomplete)

This shows which pair groups perform best.
"""

import json
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

# The 64 adjacent pairs used
ADJACENT_PAIRS_64 = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (16, 23), (17, 27), (18, 31), (19, 35), (20, 21), (24, 25), (28, 29), (32, 33),
    (36, 41), (37, 45), (38, 49), (39, 53), (42, 43), (46, 47), (50, 51), (54, 55),
    (56, 63), (57, 67), (58, 71), (59, 75), (60, 61), (64, 65), (68, 69), (72, 73),
    (76, 81), (77, 85), (78, 89), (79, 93), (82, 83), (86, 87), (90, 91), (94, 95),
    (96, 103), (97, 107), (98, 111), (99, 115), (100, 101), (104, 105), (108, 109), (112, 113),
    (116, 121), (117, 125), (118, 129), (119, 133), (122, 123), (126, 127), (130, 131), (134, 135),
    (136, 143), (137, 147), (138, 151), (139, 155), (140, 141), (144, 145), (148, 149), (152, 153)
]


def extract_pair_correlations(counts, num_pairs=64):
    """Extract correlation for each individual pair."""
    num_qubits = num_pairs * 2
    pair_correlations = [0.0] * num_pairs
    total_counts = sum(counts.values())

    for bitstring, count in counts.items():
        for pair_idx in range(num_pairs):
            alice_bit = int(bitstring[num_qubits - 1 - 2*pair_idx])
            bob_bit = int(bitstring[num_qubits - 2 - 2*pair_idx])

            if alice_bit == bob_bit:
                pair_correlations[pair_idx] += count
            else:
                pair_correlations[pair_idx] -= count

    # Normalize
    pair_correlations = [c / total_counts for c in pair_correlations]
    return pair_correlations


def calc_L3_from_correlations(corrs_6):
    """Calculate L3 from 6 correlations for one angle sign."""
    C = corrs_6
    L3 = (1/3) * (abs(C[0] + C[1]) + abs(C[2] + C[3]) + abs(C[4] + C[5]))
    return L3


def analyze_job(job_id, phi_deg):
    """Analyze a single job by pair groups."""
    print(f"\n{'='*70}")
    print(f"Analyzing job {job_id} (φ = ±{phi_deg}°)")
    print(f"{'='*70}")

    # Fetch job results
    service = QiskitRuntimeService()
    job = service.job(job_id)
    result = job.result()

    pub_result = result[0]
    counts = dict(pub_result.data.meas.get_counts())
    num_shots = sum(counts.values())

    print(f"Total shots: {num_shots}")
    print(f"Unique bitstrings: {len(counts)}")

    # Extract per-pair correlations
    pair_corrs = extract_pair_correlations(counts)

    # Calculate bound
    phi_rad = np.radians(phi_deg)
    bound_pos = 2 - (2/3) * abs(np.sin(phi_rad/2))
    bound_neg = 2 - (2/3) * abs(np.sin(-phi_rad/2))

    print(f"\nLeggett bound: {bound_pos:.4f}")

    # Print individual pair correlations
    print(f"\n{'Pair':<6} {'Qubits':<12} {'CorrType':<10} {'Correlation':>12}")
    print("-" * 45)

    for i, corr in enumerate(pair_corrs):
        corr_type = i % 12
        angle_sign = '+φ' if corr_type < 6 else '-φ'
        qubits = ADJACENT_PAIRS_64[i]
        print(f"{i:<6} {str(qubits):<12} {corr_type:<10} {corr:>+12.4f}")

    # Calculate L3 for each complete group of 12 pairs
    print(f"\n{'='*70}")
    print("L3 BY PAIR GROUP (5 complete groups)")
    print(f"{'='*70}")

    group_results = []
    for group_idx in range(5):
        start_pair = group_idx * 12
        end_pair = start_pair + 12

        # Get correlations for this group
        group_corrs = pair_corrs[start_pair:end_pair]

        # +phi uses correlation types 0-5, -phi uses 6-11
        corrs_pos = group_corrs[0:6]
        corrs_neg = group_corrs[6:12]

        L3_pos = calc_L3_from_correlations(corrs_pos)
        L3_neg = calc_L3_from_correlations(corrs_neg)

        violated_pos = L3_pos > bound_pos
        violated_neg = L3_neg > bound_neg

        group_results.append({
            'group': group_idx,
            'pairs': f"{start_pair}-{end_pair-1}",
            'qubits': [ADJACENT_PAIRS_64[i] for i in range(start_pair, end_pair)],
            'L3_pos': L3_pos,
            'L3_neg': L3_neg,
            'violated_pos': violated_pos,
            'violated_neg': violated_neg
        })

        status_pos = "VIOLATION!" if violated_pos else ""
        status_neg = "VIOLATION!" if violated_neg else ""

        print(f"\nGroup {group_idx} (pairs {start_pair}-{end_pair-1}):")
        print(f"  Qubits: {[ADJACENT_PAIRS_64[i] for i in range(start_pair, end_pair)]}")
        print(f"  +{phi_deg}°: L₃ = {L3_pos:.4f} (bound {bound_pos:.4f}) {status_pos}")
        print(f"  -{phi_deg}°: L₃ = {L3_neg:.4f} (bound {bound_neg:.4f}) {status_neg}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY BY GROUP")
    print(f"{'='*70}")
    print(f"{'Group':<8} {'Pairs':<12} {'L3(+φ)':<12} {'L3(-φ)':<12} {'Best?'}")
    print("-" * 55)

    best_pos = max(group_results, key=lambda x: x['L3_pos'])
    best_neg = max(group_results, key=lambda x: x['L3_neg'])

    for g in group_results:
        is_best = ""
        if g == best_pos:
            is_best += "+φ "
        if g == best_neg:
            is_best += "-φ"
        print(f"{g['group']:<8} {g['pairs']:<12} {g['L3_pos']:<12.4f} {g['L3_neg']:<12.4f} {is_best}")

    print(f"\nBest group for +φ: Group {best_pos['group']} with L₃ = {best_pos['L3_pos']:.4f}")
    print(f"Best group for -φ: Group {best_neg['group']} with L₃ = {best_neg['L3_neg']:.4f}")

    return group_results


def load_jobs_from_json(json_file):
    """Load unique job IDs and angles from JSON results file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract unique (job_id, abs(phi_deg)) pairs
    # Since both +φ and -φ share the same job, use absolute value
    seen = set()
    jobs = []
    for entry in data:
        job_id = entry['job_id']
        phi_deg = abs(entry['phi_deg'])
        if (job_id, phi_deg) not in seen:
            seen.add((job_id, phi_deg))
            jobs.append((job_id, phi_deg))

    # Sort by angle
    jobs.sort(key=lambda x: x[1])
    return jobs


def main():
    import sys
    import os

    # Default JSON file
    default_json = "leggett_results_ibm_ibm_pittsburgh_64pairs_20251208_141241_10k.json"

    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        # Load jobs from specified JSON file
        json_file = sys.argv[1]
        print(f"Loading jobs from: {json_file}")
        jobs = load_jobs_from_json(json_file)
        print(f"Found {len(jobs)} unique jobs:")
        for job_id, phi_deg in jobs:
            print(f"  {job_id}: ±{phi_deg}°")
    elif len(sys.argv) > 1:
        # Analyze specific job
        job_id = sys.argv[1]
        phi_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 30
        analyze_job(job_id, phi_deg)
        return
    elif os.path.exists(default_json):
        # Use default JSON file
        print(f"Loading jobs from default: {default_json}")
        jobs = load_jobs_from_json(default_json)
        print(f"Found {len(jobs)} unique jobs:")
        for job_id, phi_deg in jobs:
            print(f"  {job_id}: ±{phi_deg}°")
    else:
        # Fallback to hardcoded jobs
        jobs = [
            ('d4qhg0s5fjns73d0eg30', 15),
            ('d4qhg0s5fjns73d0eg3g', 25),
            ('d4qhg0rher1c73bbaha0', 30),
            ('d4qhg145fjns73d0eg40', 45),
            ('d4qhg17t3pms73978mag', 60),
        ]
        print("Using hardcoded jobs")

    # Analyze all jobs
    all_results = {}
    for job_id, phi_deg in jobs:
        results = analyze_job(job_id, phi_deg)
        all_results[phi_deg] = results

    # Overall best groups
    print(f"\n{'='*70}")
    print("OVERALL BEST PERFORMING GROUPS")
    print(f"{'='*70}")

    group_scores = [0] * 5
    for phi_deg, results in all_results.items():
        for g in results:
            # Score based on how close to bound
            phi_rad = np.radians(phi_deg)
            bound = 2 - (2/3) * abs(np.sin(phi_rad/2))
            score = (g['L3_pos'] + g['L3_neg']) / 2 / bound
            group_scores[g['group']] += score

    print("\nGroup scores (higher = better):")
    for i, score in enumerate(group_scores):
        print(f"  Group {i}: {score:.4f}")

    best_group = group_scores.index(max(group_scores))
    print(f"\nBest overall group: {best_group}")
    print(f"Qubit pairs: {ADJACENT_PAIRS_64[best_group*12:(best_group+1)*12]}")


if __name__ == "__main__":
    main()
