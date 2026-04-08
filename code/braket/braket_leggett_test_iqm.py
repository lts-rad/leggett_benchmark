#!/usr/bin/env python3
"""
Leggett Inequality Test using AWS Braket (IonQ Forte) - Sequential Version
Based on arXiv:0801.2241v2

This version runs SEPARATE jobs for each angle to avoid large circuit decoherence.
Each job measures 6 correlations (3 pairs × 2 measurements) using 1 singlet = 12 qubits.

Actually, we can do even better: 6 independent singlets in parallel (6 correlations),
which is 12 qubits total, and we submit one job per angle.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_braket_provider import AWSBraketBackend
from braket.aws import AwsDevice

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from leggett import *


def run_single_angle(phi_deg, num_shots, use_braket, device_arn, dry_run=False):
    """
    Run Leggett test for a single angle φ.

    Args:
        phi_deg: Angle in degrees
        num_shots: Number of measurements
        use_braket: If True, run on AWS Braket
        device_arn: ARN of AWS Braket device
        dry_run: If True, only transpile without submitting

    Returns:
        Dictionary with results: correlations, L3, L3_theory, bound, violated, job_id, etc.
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running angle φ = {phi_deg:+.1f}° ({num_shots} shots)")
    print(f"{'='*70}")

    # Create circuit
    qc = create_leggett_circuit_for_angle(phi_rad)

    # Run on Braket or simulator
    if use_braket:
        device = AwsDevice(device_arn)
        backend = AWSBraketBackend(device=device)

        print(f"  Transpiling for {device_arn.split('/')[-1]}...")
        print(f"  Original circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

        # Transpile for Braket backend with manual qubit mapping to prevent expansion
        # Map logical qubits 0-23 to physical qubits 0-23
        initial_layout = list(range(24))
        qc_transpiled = transpile(qc, backend=backend, optimization_level=1,
                                   initial_layout=initial_layout)
        print(f"  Transpiled: {qc_transpiled.num_qubits} qubits, depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

        if dry_run:
            print(f"  DRY RUN: Would submit job here. Exiting.")
            return None

        print(f"  Submitting job...")
        job = backend.run(qc_transpiled, shots=num_shots)
        job_id = job.job_id()
        print(f"  Job ID: {job_id}")
        print(f"  Waiting for results...")

        result = job.result()
        counts = result.get_counts()
    else:
        print(f"  Running on local simulator...")
        simulator = AerSimulator(method='matrix_product_state')
        result = simulator.run(qc, shots=num_shots).result()
        counts = result.get_counts()
        job_id = "local_simulator"

    # Extract correlations for both +phi and -phi
    correlations_pos, correlations_neg = extract_correlations_from_counts(counts, num_shots)

    # Calculate results for both angles
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
    result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

    print(f"\n  Results for φ = +{abs(phi_deg):.1f}°:")
    print(f"    Correlations (exp): {result_pos['correlations']}")
    print(f"    Correlations (th):  {result_pos['correlations_theory']}")
    print(f"    L₃ (exp):      {result_pos['L3']:.4f}")
    print(f"    L₃ (theory):   {result_pos['L3_theory']:.4f}")
    print(f"    L₃ bound:      {result_pos['bound']:.4f}")
    print(f"    Violated:      {result_pos['violated']}")

    print(f"\n  Results for φ = -{abs(phi_deg):.1f}°:")
    print(f"    Correlations (exp): {result_neg['correlations']}")
    print(f"    Correlations (th):  {result_neg['correlations_theory']}")
    print(f"    L₃ (exp):      {result_neg['L3']:.4f}")
    print(f"    L₃ (theory):   {result_neg['L3_theory']:.4f}")
    print(f"    L₃ bound:      {result_neg['bound']:.4f}")
    print(f"    Violated:      {result_neg['violated']}")

    print(f"\n  Unique bitstrings: {len(counts)}")

    # Return results for both angles
    return {
        'positive': {
            'phi_deg': abs(phi_deg),
            'phi_rad': phi_rad,
            **result_pos,
            'job_id': job_id,
            'num_shots': num_shots,
            'unique_bitstrings': len(counts),
            'timestamp': datetime.now().isoformat()
        },
        'negative': {
            'phi_deg': -abs(phi_deg),
            'phi_rad': -phi_rad,
            **result_neg,
            'job_id': job_id,
            'num_shots': num_shots,
            'unique_bitstrings': len(counts),
            'timestamp': datetime.now().isoformat()
        }
    }


def main():
    import sys

    # Parse command line arguments
    use_braket = '--braket' in sys.argv
    dry_run = '--dry-run' in sys.argv
    device_arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald"
    num_shots = 1000  # Default shots
    output_file = "leggett_results_iqm_emerald_sequential.json"

    for i, arg in enumerate(sys.argv):
        if arg == '--device-arn' and i + 1 < len(sys.argv):
            device_arn = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    print("="*70)
    print("LEGGETT INEQUALITY TEST: Sequential Jobs (AWS Braket)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Backend: {'AWS Braket - ' + device_arn.split('/')[-1] if use_braket else 'Local Simulator'}")
    print(f"  Shots per angle: {num_shots}")
    print(f"  Circuit size: 12 qubits (6 singlet pairs)")
    print(f"  Dry run: {dry_run}")

    # Test angles - complementary pairs (only positive, circuit handles both)
    # This tests: ±15, ±25, ±30, ±45, ±60
    test_angles = [15, 25, 30, 45, 60]

    print(f"\nTest angles: {test_angles}")
    print(f"Total jobs to submit: {len(test_angles)}")

    if dry_run and use_braket:
        print("\n*** DRY RUN MODE: Will transpile one circuit and exit ***")
        run_single_angle(test_angles[0], num_shots, use_braket, device_arn, dry_run=True)
        return

    # Run all angles
    results = []

    for angle in test_angles:
        result = run_single_angle(angle, num_shots, use_braket, device_arn)
        if result:
            results.append(result['positive'])
            results.append(result['negative'])

        # Brief pause between job submissions when using Braket
        if use_braket and angle != test_angles[-1]:
            print(f"\nWaiting 2 seconds before next submission...")
            time.sleep(2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    violations = sum(1 for r in results if r['violated'])
    print(f"\nTotal angles tested: {len(results)}")
    print(f"Violations: {violations}/{len(results)}")

    print(f"\nAll results:")
    print(f"{'Angle':>8} {'L₃(exp)':>10} {'L₃(th)':>10} {'Bound':>10} {'Δ(exp-th)':>12} {'Status':>12}")
    print("-"*70)

    for r in results:
        status = "VIOLATION" if r['violated'] else "No violation"
        delta = r['L3'] - r['L3_theory']
        print(f"{r['phi_deg']:>+8.1f}° {r['L3']:>10.4f} {r['L3_theory']:>10.4f} {r['bound']:>10.4f} {delta:>+12.4f} {status:>12}")

    # Save results to JSON
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Done!")


if __name__ == "__main__":
    main()
