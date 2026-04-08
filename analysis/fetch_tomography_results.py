#!/usr/bin/env python3
"""
Fetch and parse results from submitted tomography jobs.
"""

import numpy as np
import json
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from tomography import (
    reconstruct_density_matrix,
    calculate_tangle,
    calculate_purity,
    calculate_fidelity_singlet,
)

# Job IDs from the submitted tomography run
TOMOGRAPHY_JOBS = {
    'XX': 'd4viugng0u6s73darakg',
    'XY': 'd4viugkgk3fc73av1in0',
    'XZ': 'd4viuguaec6c738sed3g',
    'YX': 'd4viugsgk3fc73av1io0',
    'YY': 'd4viuh4gk3fc73av1iog',
    'YZ': 'd4viuh4gk3fc73av1ip0',
    'ZX': 'd4viuheaec6c738sed4g',
    'ZY': 'd4viuhfg0u6s73daralg',
    'ZZ': 'd4viuhdeastc73ci9e80',
}

# The qubit pairs used
QUBIT_PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

NUM_PAIRS = 12
NUM_SHOTS = 8192  # Default, will be read from job


def extract_correlations(counts, num_shots, num_pairs=12):
    """Extract correlation for each pair from counts."""
    corrs = [0.0] * num_pairs
    n_qubits = 2 * num_pairs  # 24 qubits

    for bitstring, count in counts.items():
        for pair_idx in range(num_pairs):
            # Alice is qubit 2*pair_idx, Bob is 2*pair_idx + 1
            a_pos = (n_qubits - 1) - 2 * pair_idx
            b_pos = (n_qubits - 1) - (2 * pair_idx + 1)

            a_val = (-1) ** int(bitstring[a_pos])
            b_val = (-1) ** int(bitstring[b_pos])
            corrs[pair_idx] += a_val * b_val * count

    return [c / num_shots for c in corrs]


def get_counts_from_result(result):
    """Extract counts from job result, handling different Qiskit versions."""
    pub_result = result[0]

    # Try different attribute names
    if hasattr(pub_result.data, 'meas'):
        return pub_result.data.meas.get_counts()
    elif hasattr(pub_result.data, 'c'):
        return pub_result.data.c.get_counts()
    else:
        # List all available attributes
        data_attrs = [a for a in dir(pub_result.data) if not a.startswith('_')]
        print(f"  Available data attributes: {data_attrs}")

        # Try the first one that looks like measurement data
        for attr in data_attrs:
            obj = getattr(pub_result.data, attr)
            if hasattr(obj, 'get_counts'):
                print(f"  Using attribute: {attr}")
                return obj.get_counts()

        raise AttributeError(f"Cannot find counts in result. Attributes: {data_attrs}")


def main():
    print("=" * 70)
    print("FETCHING TOMOGRAPHY RESULTS")
    print("=" * 70)

    service = QiskitRuntimeService()

    # Check job statuses
    print("\n--- Job Status ---")
    all_done = True
    for basis_name, job_id in TOMOGRAPHY_JOBS.items():
        job = service.job(job_id)
        status = str(job.status())
        print(f"  {basis_name}: {job_id} -> {status}")
        if status not in ['DONE', 'COMPLETED']:
            all_done = False

    if not all_done:
        print("\nNot all jobs are complete yet. Run again later.")
        return

    # Fetch results
    print("\n--- Fetching Results ---")
    all_results = {}

    for basis_name, job_id in TOMOGRAPHY_JOBS.items():
        print(f"  {basis_name}: ", end="", flush=True)
        job = service.job(job_id)
        result = job.result()

        try:
            counts = get_counts_from_result(result)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Get actual shot count
        total_counts = sum(counts.values())

        corrs = extract_correlations(counts, total_counts, NUM_PAIRS)
        all_results[basis_name] = corrs
        print(f"OK ({total_counts} shots, pair0={corrs[0]:.3f})")

    if len(all_results) < 9:
        print(f"\nWARNING: Only got {len(all_results)}/9 bases!")

    # Analyze per pair
    print(f"\n{'='*70}")
    print("PER-PAIR TOMOGRAPHY ANALYSIS")
    print(f"{'='*70}")

    pair_results = []

    for pair_idx, (q1, q2) in enumerate(QUBIT_PAIRS):
        exp = {b: all_results[b][pair_idx] for b in all_results}

        rho = reconstruct_density_matrix(exp, single_exp=None)
        tangle, conc = calculate_tangle(rho)
        purity = calculate_purity(rho)
        fidelity = calculate_fidelity_singlet(rho)
        vis = (2 * conc + 1) / 3

        pair_results.append({
            'pair': (q1, q2),
            'visibility': float(vis),
            'concurrence': float(conc),
            'tangle': float(tangle),
            'purity': float(purity),
            'fidelity': float(fidelity),
            'expectations': {k: float(v) for k, v in exp.items()}
        })

    # Print results table
    print(f"\n{'Pair':>12} {'Visibility':>10} {'Concurrence':>12} {'Purity':>10} {'Fidelity':>10}")
    print("-" * 60)

    for p in pair_results:
        q1, q2 = p['pair']
        print(f"({q1:>3},{q2:>3})   {p['visibility']:>10.4f}   {p['concurrence']:>10.4f}   {p['purity']:>8.4f}   {p['fidelity']:>8.4f}")

    vis_list = [p['visibility'] for p in pair_results]
    conc_list = [p['concurrence'] for p in pair_results]
    purity_list = [p['purity'] for p in pair_results]
    fidelity_list = [p['fidelity'] for p in pair_results]

    print("-" * 60)
    print(f"{'Average':>12} {np.mean(vis_list):>10.4f}   {np.mean(conc_list):>10.4f}   {np.mean(purity_list):>8.4f}   {np.mean(fidelity_list):>8.4f}")
    print(f"{'Std':>12} {np.std(vis_list):>10.4f}   {np.std(conc_list):>10.4f}   {np.std(purity_list):>8.4f}   {np.std(fidelity_list):>8.4f}")

    # Print detailed expectations
    print(f"\n{'='*70}")
    print("PAULI EXPECTATIONS (XX, YY, ZZ should be ~ -1 for singlet)")
    print(f"{'='*70}")
    print(f"\n{'Pair':>12}  {'XX':>8}  {'YY':>8}  {'ZZ':>8}  {'XY':>8}  {'XZ':>8}  {'YZ':>8}")
    print("-" * 70)
    for p in pair_results:
        q1, q2 = p['pair']
        e = p['expectations']
        print(f"({q1:>3},{q2:>3})   {e['XX']:>8.4f}  {e['YY']:>8.4f}  {e['ZZ']:>8.4f}  {e['XY']:>8.4f}  {e['XZ']:>8.4f}  {e['YZ']:>8.4f}")

    # Save results
    output = {
        'qubit_pairs': QUBIT_PAIRS,
        'job_ids': TOMOGRAPHY_JOBS,
        'timestamp': datetime.now().isoformat(),
        'pair_results': pair_results,
        'raw_basis_results': {k: v for k, v in all_results.items()},
        'summary': {
            'avg_visibility': float(np.mean(vis_list)),
            'std_visibility': float(np.std(vis_list)),
            'avg_concurrence': float(np.mean(conc_list)),
            'avg_purity': float(np.mean(purity_list)),
            'avg_fidelity': float(np.mean(fidelity_list))
        }
    }

    output_file = "tomography_12pairs_ibm_pittsburgh.json"
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
