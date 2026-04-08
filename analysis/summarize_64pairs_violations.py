#!/usr/bin/env python3
"""
Summary of individual pair performance across all angles.
Shows each pair's correlation at each angle, ranked by quality.
"""

import json
import sys
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

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


def load_jobs_from_json(json_file):
    """Load unique job IDs and angles from JSON results file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    seen = set()
    jobs = []
    for entry in data:
        job_id = entry['job_id']
        phi_deg = abs(entry['phi_deg'])
        if (job_id, phi_deg) not in seen:
            seen.add((job_id, phi_deg))
            jobs.append((job_id, phi_deg))

    jobs.sort(key=lambda x: x[1])
    return jobs


def extract_pair_correlations(counts, num_pairs=64):
    num_qubits = num_pairs * 2
    pair_corrs = [0.0] * num_pairs
    total = sum(counts.values())
    for bitstring, count in counts.items():
        for p in range(num_pairs):
            a = int(bitstring[num_qubits - 1 - 2*p])
            b = int(bitstring[num_qubits - 2 - 2*p])
            if a == b:
                pair_corrs[p] += count
            else:
                pair_corrs[p] -= count
    return [c / total for c in pair_corrs]


def get_theoretical_correlation(phi_deg, corr_type):
    """Get theoretical correlation for given angle and correlation type."""
    # Correlation types 0-5 are for +phi, 6-11 are for -phi
    phi_rad = np.radians(phi_deg if corr_type < 6 else -phi_deg)
    return -np.cos(phi_rad / 2)


def main():
    # Load JSON file
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "leggett_results_ibm_ibm_pittsburgh_64pairs_20251208_141241_10k.json"

    print(f"Loading jobs from: {json_file}")
    jobs = load_jobs_from_json(json_file)

    # Fetch all job results
    service = QiskitRuntimeService()
    all_pair_corrs = {}  # {phi_deg: [64 correlations]}

    for job_id, phi_deg in jobs:
        print(f"Fetching {phi_deg}°...", end=" ", flush=True)
        job = service.job(job_id)
        result = job.result()
        counts = dict(result[0].data.meas.get_counts())
        all_pair_corrs[phi_deg] = extract_pair_correlations(counts)
    print("Done!\n")

    angles = sorted(all_pair_corrs.keys())

    # Calculate quality for each pair based on distance from QM theory
    pair_quality = []
    for p in range(64):
        corr_type = p % 12
        sign = '+' if corr_type < 6 else '-'

        # Calculate average distance from QM theory
        distances = []
        bad_angles = []
        correlations_with_theory = {}

        for phi in angles:
            corr = all_pair_corrs[phi][p]
            expected = get_theoretical_correlation(phi, corr_type)
            distance = abs(corr - expected)
            distances.append(distance)
            correlations_with_theory[phi] = {'measured': corr, 'theory': expected, 'distance': distance}

            if corr > 0:  # Wrong sign!
                bad_angles.append(f"{phi}° WRONG SIGN!")
            elif abs(corr) < 0.5:  # Too weak
                bad_angles.append(f"{phi}° weak ({corr:.2f})")

        avg_distance = np.mean(distances)

        pair_quality.append({
            'pair_idx': p,
            'qubits': ADJACENT_PAIRS_64[p],
            'corr_type': corr_type,
            'sign': sign,
            'avg_distance': avg_distance,
            'bad_angles': bad_angles,
            'correlations': correlations_with_theory
        })

    # Sort by quality (smallest distance from theory = best)
    pair_quality.sort(key=lambda x: (x['avg_distance'], len(x['bad_angles'])))

    # Print header
    print("=" * 120)
    print("INDIVIDUAL PAIR PERFORMANCE (sorted by avg distance from QM theory - lower = better)")
    print("=" * 120)
    print(f"{'Rank':<5} {'Pair':<6} {'Qubits':<12} {'Type':<6} {'AvgΔ':<8}", end="")
    for phi in angles:
        print(f"Δ@{phi}°".center(8), end="")
    print("  Issues")
    print("-" * 120)

    # Print each pair
    for rank, pq in enumerate(pair_quality, 1):
        print(f"{rank:<5} {pq['pair_idx']:<6} {str(pq['qubits']):<12} {pq['sign']}φ[{pq['corr_type']%6}] ", end="")
        print(f"{pq['avg_distance']:.4f}  ", end="")

        for phi in angles:
            d = pq['correlations'][phi]['distance']
            m = pq['correlations'][phi]['measured']
            if m > 0:  # Wrong sign
                print(f"{'BAD':^8}", end="")
            elif d > 0.15:  # Large deviation
                print(f"{d:.3f}!".center(8), end="")
            elif d > 0.08:  # Medium deviation
                print(f"{d:.3f}*".center(8), end="")
            else:  # Good
                print(f"{d:.4f}".center(8), end="")

        if pq['bad_angles']:
            print(f"  <- {len(pq['bad_angles'])} issues")
        else:
            print()

    # Summary of problematic pairs
    print("\n" + "=" * 120)
    print("PROBLEMATIC PAIRS (wrong sign or |corr| < 0.5 at any angle)")
    print("=" * 120)

    problem_pairs = [pq for pq in pair_quality if pq['bad_angles']]
    if problem_pairs:
        for pq in problem_pairs:
            print(f"Pair {pq['pair_idx']} {pq['qubits']}: {', '.join(pq['bad_angles'])}")
    else:
        print("None!")

    # Top 12 pairs (closest to QM theory)
    print("\n" + "=" * 120)
    print("TOP 12 PAIRS (closest to QM theory)")
    print("=" * 120)
    for i, pq in enumerate(pair_quality[:12], 1):
        print(f"{i:2}. Pair {pq['pair_idx']:2} {str(pq['qubits']):<12} avg_distance={pq['avg_distance']:.4f}")

    # Bottom 12 pairs (furthest from QM theory)
    print("\n" + "=" * 120)
    print("BOTTOM 12 PAIRS (furthest from QM theory)")
    print("=" * 120)
    for i, pq in enumerate(pair_quality[-12:], 53):
        print(f"{i:2}. Pair {pq['pair_idx']:2} {str(pq['qubits']):<12} avg_distance={pq['avg_distance']:.4f}")


if __name__ == "__main__":
    main()
