#!/usr/bin/env python3
"""
Leggett Inequality Test using IonQ Forte-1 Noisy Simulator
Based on arXiv:0801.2241v2

This version uses the IonQ native API with the Forte-1 noisy simulator.
Uses 24 qubits (12 singlet pairs) to test complementary angles ±φ simultaneously.
Each job tests both +φ and -φ in a single circuit, measuring 12 correlations total.

Usage:
    python3 ionq_leggett_test.py [--shots NUM] [--output FILE] [--dry-run]

Requirements:
    - qiskit_ionq package installed
    - API key in 'cred_ionqsim' file or IONQ_API_KEY environment variable
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit_ionq import IonQProvider

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle, extract_correlations_from_counts, calc_leggett_for_angle



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
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

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
        },
        'counts': counts
    }


def main():
    import sys

    import os

    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    num_shots = 1000  # Default shots
    use_noise_model = False  # Set to True to use Forte-1 noise model
    noise_model = 'forte-1' if use_noise_model else None
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    output_file = os.path.join(_data_dir, 'leggett_results_ionq_forte_NOISE_MODEL_24qb.json') if use_noise_model else os.path.join(_data_dir, 'leggett_results_ionq_forte_IDEAL_24qb.json')

    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    # Load API key from file or environment
    cred_file = 'cred_ionqsim'
    if os.path.exists(cred_file):
        with open(cred_file, 'r') as f:
            api_key = f.read().strip()
    else:
        api_key = os.getenv("IONQ_API_KEY")
        if not api_key:
            print("ERROR: No IonQ API key found!")
            print("  Either create 'cred_ionqsim' file or set IONQ_API_KEY environment variable")
            sys.exit(1)

    # Initialize IonQ provider
    print(f"\nConnecting to IonQ...")
    provider = IonQProvider(api_key)
    backend = provider.get_backend("simulator")
    print(f"Backend: {backend.name}")

    print("\n" + "="*70)
    print("LEGGETT INEQUALITY TEST: IonQ Simulator")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Backend: IonQ {'Forte-1 Noisy' if use_noise_model else 'Ideal'} Simulator (via IonQ API)")
    print(f"  Noise model: {noise_model if use_noise_model else 'None (ideal)'}")
    print(f"  Shots per angle pair: {num_shots}")
    print(f"  Circuit size: 24 qubits (12 singlet pairs, testing ±φ simultaneously)")
    print(f"  Dry run: {dry_run}")
    print(f"  Output file: {output_file}")

    # Test angles - complementary pairs (only positive, circuit handles both)
    # This tests: ±15, ±25, ±30, ±45, ±60
    test_angles = [15, 25, 30, 45, 60]

    print(f"\nTest angles: {test_angles}")
    print(f"Total jobs to submit: {len(test_angles)}")

    if dry_run:
        print("\n*** DRY RUN MODE: Will create and transpile one circuit then exit ***")
        phi_rad = np.radians(test_angles[0])
        qc = create_leggett_circuit_for_angle(phi_rad)
        print(f"  Circuit created: {qc.num_qubits} qubits, depth {qc.depth()}")
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")
        return

    # Run all angles
    results = []

    if True:  # Always use IonQ API
        # Phase 1: Submit all jobs in parallel (don't wait for results)
        print("\n" + "="*70)
        print("PHASE 1: SUBMITTING ALL JOBS")
        print("="*70)

        jobs = []
        for angle in test_angles:
            phi_rad = np.radians(angle)
            print(f"\n{'='*70}")
            print(f"Submitting job for angle φ = ±{abs(angle):.1f}° ({num_shots} shots)")
            print(f"{'='*70}")

            # Create and submit circuit
            qc = create_leggett_circuit_for_angle(phi_rad)

            print(f"  Transpiling for IonQ...")
            print(f"  Original circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
            qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
            print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

            if use_noise_model:
                print(f"  Submitting job with noise model '{noise_model}'...")
                job = backend.run(qc_transpiled, shots=num_shots, noise_model=noise_model)
            else:
                print(f"  Submitting job (ideal simulator)...")
                job = backend.run(qc_transpiled, shots=num_shots)
            job_id = job.job_id()
            print(f"  Job ID: {job_id}")

            jobs.append((angle, job))

        # Phase 2: Collect all results
        print("\n" + "="*70)
        print("PHASE 2: COLLECTING RESULTS")
        print("="*70)
        print(f"\nWaiting for {len(jobs)} jobs to complete...")
        print("This may take several minutes to hours depending on queue.\n")

        for angle, job in jobs:
            print(f"\nFetching results for φ = ±{abs(angle):.1f}°...")
            print(f"  Job ID: {job.job_id()}")
            result = job.result()
            counts = result.get_counts()

            # Process results (same as run_single_angle)
            phi_rad = np.radians(angle)
            correlations_pos, correlations_neg = extract_correlations_from_counts(counts, num_shots)

            # Calculate results for both angles
            from datetime import datetime

            result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
            result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

            results.append({
                'phi_deg': abs(angle), 'phi_rad': phi_rad, **result_pos,
                'job_id': job.job_id(), 'num_shots': num_shots,
                'unique_bitstrings': len(counts), 'timestamp': datetime.now().isoformat()
            })
            results.append({
                'phi_deg': -abs(angle), 'phi_rad': -phi_rad, **result_neg,
                'job_id': job.job_id(), 'num_shots': num_shots,
                'unique_bitstrings': len(counts), 'timestamp': datetime.now().isoformat()
            })

            print(f"  ✓ Results collected for ±{abs(angle):.1f}°")
    else:
        # Simulator: run sequentially
        for angle in test_angles:
            result = run_single_angle(angle, num_shots, use_braket, device_arn)
            if result:
                results.append(result['positive'])
                results.append(result['negative'])

    # Sort results by angle
    results.sort(key=lambda x: x['phi_deg'])

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
