#!/usr/bin/env python3
"""
Leggett Inequality Test using Azure Quantum (Quantinuum H2) - Sequential Version
Based on arXiv:0801.2241v2

This version runs SEPARATE jobs for each angle to avoid large circuit decoherence.
Each job measures 12 correlations (both +φ and -φ) using 12 independent singlet pairs = 24 qubits total.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit_aer import AerSimulator
from azure.quantum.qiskit import AzureQuantumProvider

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from leggett import create_leggett_circuit_for_angle, extract_correlations_from_counts, calc_leggett_for_angle

def run_single_angle(phi_deg, num_shots, use_azure, resource_id, location='eastus', dry_run=False, use_noise_model=False):
    """
    Run Leggett test for a single angle φ.

    Args:
        phi_deg: Angle in degrees
        num_shots: Number of measurements
        use_azure: If True, run on Azure Quantum H2 hardware
        resource_id: Azure Quantum resource ID
        location: Azure location (default: 'eastus')
        dry_run: If True, only transpile without submitting
        use_noise_model: If True, simulate with noise model (not supported for Azure)

    Returns:
        Dictionary with results: correlations, L3, L3_theory, bound, violated, job_id, etc.
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running complementary angles φ = ±{abs(phi_deg):.1f}° ({num_shots} shots, 24 qubits)")
    print(f"{'='*70}")

    # Create circuit
    qc = create_leggett_circuit_for_angle(phi_rad)

    # Run on Azure Quantum or simulator
    if use_azure:
        # Connect to Azure Quantum
        print(f"  Connecting to Azure Quantum...")
        provider = AzureQuantumProvider(
            resource_id=resource_id,
            location=location
        )
        backend = provider.get_backend("pasqal.sim.emu-tn")

        print(f"  Transpiling for Quantinuum H2...")
        print(f"  Original circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
        qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
        print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

        if dry_run:
            print(f"  DRY RUN: Would submit job here. Exiting.")
            return None

        # Run on hardware
        print(f"  Submitting job to Quantinuum H2...")
        job = backend.run(qc_transpiled, shots=num_shots)
        job_id = job.job_id()
        print(f"  Job ID: {job_id}")
        print(f"  Waiting for results (this may take hours depending on queue)...")

        # Wait with status updates
        start_time = time.time()
        while True:
            status = job.status()
            status_str = str(status)
            elapsed = time.time() - start_time
            print(f"  Status: {status_str} (elapsed: {elapsed/60:.1f} min)", end='\r')

            # Check if job is done (convert status to string for comparison)
            if 'DONE' in status_str or 'COMPLETED' in status_str or 'Succeeded' in status_str or \
               'ERROR' in status_str or 'CANCELLED' in status_str or 'Failed' in status_str:
                print()  # New line after status updates
                break

            time.sleep(30)  # Check every 30 seconds

        if 'ERROR' in status_str or 'CANCELLED' in status_str or 'Failed' in status_str:
            print(f"  Job {status_str}! Skipping this angle.")
            return None

        result = job.result()
        counts = result.get_counts()
        job_id = job.job_id()

    else:
        if use_noise_model:
            print(f"  ERROR: Noise model not supported for Azure Quantum H2")
            return None
        else:
            # Perfect noiseless simulation
            print(f"  Running on local noiseless simulator...")
            simulator = AerSimulator(method='matrix_product_state')
            result = simulator.run(qc, shots=num_shots).result()
            counts = result.get_counts()
            job_id = "local_simulator_noiseless"

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

    # Parse command line arguments
    use_azure = '--azure' in sys.argv
    dry_run = '--dry-run' in sys.argv
    use_noise_model = '--noise-model' in sys.argv
    resource_id = os.environ.get("AZURE_QUANTUM_RESOURCE_ID", "")
    location = "westus"  # Default
    num_shots = 1000  # Default shots
    output_file = None  # Will be set based on use_azure

    for i, arg in enumerate(sys.argv):
        if arg == '--resource-id' and i + 1 < len(sys.argv):
            resource_id = sys.argv[i + 1]
        elif arg == '--location' and i + 1 < len(sys.argv):
            location = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    # Set default output file if not specified
    if output_file is None:
        if use_azure:
            output_file = f"leggett_results_azure_sim_pasqal-1_sequential_24qb.json"
        elif use_noise_model:
            output_file = f"leggett_results_azure_pasqal_NOISE_MODEL_24qb.json"
        else:
            output_file = "leggett_results_azure_pasqal_SIM_noiseless_24qb.json"

    print("="*70)
    print("LEGGETT INEQUALITY TEST: Sequential Jobs with 24 Qubits (Azure Quantum H2)")
    print("="*70)
    print(f"\nConfiguration:")
    if use_azure:
        print(f"  Backend: Azure Quantum - Quantinuum H2-1 Emulator (56 qubits)")
        print(f"  Resource ID: {resource_id}")
        print(f"  Location: {location}")
        print(f"  Note: Using H2-1 emulator. For hardware, add Quantinuum provider to workspace")
    elif use_noise_model:
        print(f"  Backend: Not supported")
    else:
        print(f"  Backend: Local noiseless simulator")
    print(f"  Shots per angle pair: {num_shots}")
    print(f"  Circuit size: 24 qubits (12 singlet pairs, testing ±φ simultaneously)")
    print(f"  Dry run: {dry_run}")
    print(f"  Output file: {output_file}")

    # Test angles - now testing complementary pairs (only positive, circuit handles both)
    # This tests: ±15, ±25, ±30, ±45, ±60
    test_angles = [15, 25, 30, 45, 60]

    print(f"\nTest angles: {test_angles}")
    print(f"Total jobs to submit: {len(test_angles)}")

    if dry_run and use_azure:
        print("\n*** DRY RUN MODE: Will transpile one circuit and exit ***")
        run_single_angle(test_angles[0], num_shots, use_azure, resource_id, location, dry_run=True)
        return

    # Run all angles
    results = []

    for angle in test_angles:
        result = run_single_angle(angle, num_shots, use_azure, resource_id, location,
                                 use_noise_model=use_noise_model)
        if result:
            # Add both positive and negative angle results to results list
            results.append(result['positive'])
            results.append(result['negative'])

        # Brief pause between job submissions when using Azure
        if use_azure and angle != test_angles[-1]:
            print(f"\nWaiting 2 seconds before next submission...")
            time.sleep(2)

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
