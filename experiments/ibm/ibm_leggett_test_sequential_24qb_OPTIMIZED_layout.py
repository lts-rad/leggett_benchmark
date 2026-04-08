import os
#!/usr/bin/env python3
"""
Leggett Inequality Test using IBM Quantum - Sequential Version with 24 Qubits
Using OPTIMIZED layout from iteration process.

This version runs a single job measuring 12 correlations (6 for +φ, 6 for -φ)
using 12 independent singlet pairs = 24 qubits total.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle, extract_correlations_from_counts, calc_leggett_for_angle

# OPTIMIZED layout from iteration process
# This layout showed max_error=0.0339, mean_error=0.0193 in noise simulations
OPTIMIZED_LAYOUT = [67, 68, 29, 30, 54, 55, 93, 92, 109, 110, 2, 3,
                    13, 14, 112, 113, 87, 86, 20, 21, 94, 95, 107, 97]

def run_hardware_test(phi_deg=30, num_shots=1000, backend_name='ibm_pittsburgh', dry_run=False):
    """
    Run Leggett test on IBM hardware with optimized layout.

    Args:
        phi_deg: Angle in degrees
        num_shots: Number of measurements
        backend_name: Name of IBM backend
        dry_run: If True, only transpile without submitting

    Returns:
        Dictionary with results and job_id
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"LEGGETT TEST - IBM QUANTUM HARDWARE")
    print(f"Optimized Layout from Iteration Process")
    print(f"{'='*70}")
    print(f"Angle: φ = ±{abs(phi_deg):.1f}°")
    print(f"Shots: {num_shots}")
    print(f"Backend: {backend_name}")
    print(f"Layout: {OPTIMIZED_LAYOUT}")
    print(f"{'='*70}")

    # Create circuit (measures both +phi and -phi in single circuit)
    qc = create_leggett_circuit_for_angle(phi_rad)
    print(f"\nOriginal circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    # Connect to IBM Quantum
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    print(f"\nTranspiling for {backend_name}...")
    qc_transpiled = transpile(qc, backend=backend, optimization_level=3, initial_layout=OPTIMIZED_LAYOUT)
    print(f"Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

    # Log physical qubit mapping
    layout = qc_transpiled.layout
    print(f"\nPhysical qubit mapping:")
    for logical_idx in range(qc.num_qubits):
        physical_qubit = layout.initial_layout._v2p[qc.qubits[logical_idx]]
        pair_idx = logical_idx // 2
        role = 'Alice' if logical_idx % 2 == 0 else 'Bob'
        phi_sign = '+30°' if pair_idx < 6 else '-30°'
        print(f"  Logical {logical_idx:2d} (pair {pair_idx:2d}, {role:5s}, φ={phi_sign}) -> Physical qubit {physical_qubit:3d}")

    if dry_run:
        print(f"\nDRY RUN: Would submit job here. Exiting.")
        return None

    # Configure error mitigation
    options = SamplerOptions()
    options.execution.init_qubits = True
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XY4"
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True
    print(f"\nError mitigation: twirling + dynamical decoupling (XY4) enabled")

    # Run with Sampler
    print(f"\nSubmitting job to {backend_name}...")
    sampler = Sampler(backend, options=options)
    job = sampler.run([qc_transpiled], shots=num_shots)
    job_id = job.job_id()

    print(f"\n{'='*70}")
    print(f"JOB SUBMITTED!")
    print(f"{'='*70}")
    print(f"Job ID: {job_id}")
    print(f"Status URL: https://quantum.ibm.com/jobs/{job_id}")
    print(f"{'='*70}")
    print(f"\nWaiting for results...")
    print(f"(this may take hours depending on queue)")

    # Wait with status updates
    start_time = time.time()
    last_status = None
    while True:
        status = job.status()
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {status} (elapsed: {elapsed/60:.1f} min)")
            last_status = status

        if status in ['DONE', 'COMPLETED', 'ERROR', 'CANCELLED']:
            print()  # New line after status updates
            break

        time.sleep(30)  # Check every 30 seconds

    if status == 'ERROR' or status == 'CANCELLED':
        print(f"Job {status}!")
        return None

    result = job.result()

    # Extract counts from PUB result
    pub_result = result[0]
    counts_array = pub_result.data.meas.get_counts()

    # Convert to standard counts dictionary
    counts = {}
    for bitstring, count in counts_array.items():
        counts[bitstring] = count

    # Extract correlations for both +phi and -phi
    correlations_pos, correlations_neg = extract_correlations_from_counts(counts, num_shots)

    # Calculate results for both angles
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
    result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nφ = +{abs(phi_deg):.1f}°:")
    print(f"  Correlations (exp): {result_pos['correlations']}")
    print(f"  Correlations (th):  {result_pos['correlations_theory']}")
    print(f"  L₃ (exp):      {result_pos['L3']:.4f}")
    print(f"  L₃ (theory):   {result_pos['L3_theory']:.4f}")
    print(f"  L₃ bound:      {result_pos['bound']:.4f}")
    print(f"  Violated:      {result_pos['violated']}")

    print(f"\nφ = -{abs(phi_deg):.1f}°:")
    print(f"  Correlations (exp): {result_neg['correlations']}")
    print(f"  Correlations (th):  {result_neg['correlations_theory']}")
    print(f"  L₃ (exp):      {result_neg['L3']:.4f}")
    print(f"  L₃ (theory):   {result_neg['L3_theory']:.4f}")
    print(f"  L₃ bound:      {result_neg['bound']:.4f}")
    print(f"  Violated:      {result_neg['violated']}")

    print(f"\nUnique bitstrings: {len(counts)}")

    # Return results
    return {
        'timestamp': datetime.now().isoformat(),
        'job_id': job_id,
        'backend': backend_name,
        'layout': OPTIMIZED_LAYOUT,
        'phi_deg': phi_deg,
        'num_shots': num_shots,
        'positive': {
            'phi_deg': abs(phi_deg),
            'phi_rad': phi_rad,
            **result_pos,
        },
        'negative': {
            'phi_deg': -abs(phi_deg),
            'phi_rad': -phi_rad,
            **result_neg,
        },
        'counts': counts,
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run Leggett test on IBM hardware with optimized layout')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots (default: 1000)')
    parser.add_argument('--backend', type=str, default='ibm_pittsburgh', help='Backend name (default: ibm_pittsburgh)')
    parser.add_argument('--dry-run', action='store_true', help='Only transpile, do not submit job')

    args = parser.parse_args()

    result = run_hardware_test(
        phi_deg=30,
        num_shots=args.shots,
        backend_name=args.backend,
        dry_run=args.dry_run
    )

    if result:
        # Save with unique filename including timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_{args.backend}_OPTIMIZED_layout_{timestamp}.json')

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n{'='*70}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()
