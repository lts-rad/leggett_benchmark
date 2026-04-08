import os
#!/usr/bin/env python3
"""
Leggett Inequality Test using IBM Quantum - 70 Pairs (140 Qubits)
Based on arXiv:0801.2241v2

Uses 70 entangled pairs simultaneously for maximum parallelism.
Submits ALL angle jobs concurrently (no waiting between submissions).
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_70_pairs, extract_correlations_from_counts_70_pairs, calc_leggett_for_angle


def submit_job_for_angle(phi_deg, num_shots, backend, sampler, enable_error_mitigation=True):
    """
    Transpile and submit a job for a single angle (non-blocking).

    Returns:
        Tuple of (phi_deg, job, qc_transpiled) or None if failed
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n  Preparing φ = ±{abs(phi_deg):.1f}°...")

    # Create 70-pair circuit
    qc = create_leggett_circuit_70_pairs(phi_rad)

    print(f"    Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    # Transpile (let Qiskit find optimal layout for 140 qubits)
    qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
    print(f"    Transpiled: depth {qc_transpiled.depth()}")

    # Submit job (non-blocking)
    job = sampler.run([qc_transpiled], shots=num_shots)
    job_id = job.job_id()
    print(f"    Submitted job: {job_id}")

    return (phi_deg, job, job_id)


def wait_for_job(phi_deg, job, job_id, num_shots):
    """
    Wait for a job to complete and extract results.

    Returns:
        Dictionary with results or None if failed
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n  Waiting for φ = ±{abs(phi_deg):.1f}° (job {job_id})...")

    start_time = time.time()
    while True:
        status = job.status()
        elapsed = time.time() - start_time

        if status in ['DONE', 'COMPLETED', 'ERROR', 'CANCELLED']:
            print(f"    Status: {status} (elapsed: {elapsed/60:.1f} min)")
            break

        time.sleep(30)

    if status in ['ERROR', 'CANCELLED']:
        print(f"    Job {status}! Skipping.")
        return None

    result = job.result()

    # Extract counts
    pub_result = result[0]
    counts_array = pub_result.data.meas.get_counts()
    counts = {bitstring: count for bitstring, count in counts_array.items()}

    # Extract correlations using 70-pair extraction
    correlations_pos, correlations_neg = extract_correlations_from_counts_70_pairs(counts, num_shots)

    # Calculate results for both angles
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
    result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

    print(f"\n  Results for φ = +{abs(phi_deg):.1f}°:")
    print(f"    L₃ (exp): {result_pos['L3']:.4f}, bound: {result_pos['bound']:.4f}, violated: {result_pos['violated']}")

    print(f"  Results for φ = -{abs(phi_deg):.1f}°:")
    print(f"    L₃ (exp): {result_neg['L3']:.4f}, bound: {result_neg['bound']:.4f}, violated: {result_neg['violated']}")

    return {
        'positive': {
            'phi_deg': abs(phi_deg),
            'phi_rad': phi_rad,
            **result_pos,
            'job_id': job_id,
            'num_shots': num_shots,
            'num_pairs': 70,
            'unique_bitstrings': len(counts),
            'timestamp': datetime.now().isoformat()
        },
        'negative': {
            'phi_deg': -abs(phi_deg),
            'phi_rad': -phi_rad,
            **result_neg,
            'job_id': job_id,
            'num_shots': num_shots,
            'num_pairs': 70,
            'unique_bitstrings': len(counts),
            'timestamp': datetime.now().isoformat()
        }
    }


def main():
    import sys

    # Parse arguments
    use_ibm = '--ibm' in sys.argv
    dry_run = '--dry-run' in sys.argv
    backend_name = "ibm_pittsburgh"
    num_shots = 1000
    output_file = None
    enable_error_mitigation = '--no-error-mitigation' not in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if use_ibm:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_{backend_name}_70pairs_{timestamp}.json')
        else:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_sim_70pairs_{timestamp}.json')

    print("="*70)
    print("LEGGETT INEQUALITY TEST: 70 Pairs (140 Qubits)")
    print("="*70)
    print(f"\nConfiguration:")
    if use_ibm:
        print(f"  Backend: IBM Quantum - {backend_name} (REAL HARDWARE)")
    else:
        print(f"  Backend: Local noiseless simulator")
    print(f"  Shots per angle: {num_shots}")
    print(f"  Circuit size: 140 qubits (70 singlet pairs)")
    print(f"  Redundancy: ~5.8 copies of each correlation")
    print(f"  Error mitigation: {'enabled' if enable_error_mitigation else 'disabled'}")
    print(f"  Output file: {output_file}")

    # Test angles
    test_angles = [15, 25, 30, 45, 60]
    print(f"\nTest angles: {test_angles} (each job measures both +φ and -φ)")
    print(f"Total jobs to submit: {len(test_angles)}")

    if use_ibm:
        # Connect to IBM
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

        # Configure sampler options
        options = SamplerOptions()
        if enable_error_mitigation:
            options.execution.init_qubits = True
            options.dynamical_decoupling.enable = True
            options.dynamical_decoupling.sequence_type = "XY4"
            options.twirling.enable_gates = True
            options.twirling.enable_measure = True
            print(f"\n  Error mitigation: twirling + dynamical decoupling (XY4)")

        sampler = Sampler(backend, options=options)

        if dry_run:
            print("\n*** DRY RUN: Testing transpilation only ***")
            qc = create_leggett_circuit_70_pairs(np.radians(30))
            print(f"  Original: {qc.num_qubits} qubits, depth {qc.depth()}")
            qc_t = transpile(qc, backend=backend, optimization_level=3)
            print(f"  Transpiled: depth {qc_t.depth()}, ops {qc_t.count_ops()}")
            return

        # Submit ALL jobs concurrently
        print(f"\n{'='*70}")
        print("SUBMITTING ALL JOBS CONCURRENTLY")
        print(f"{'='*70}")

        submitted_jobs = []
        for angle in test_angles:
            job_info = submit_job_for_angle(angle, num_shots, backend, sampler, enable_error_mitigation)
            if job_info:
                submitted_jobs.append(job_info)

        print(f"\n  All {len(submitted_jobs)} jobs submitted!")
        print(f"  Job IDs: {[j[2] for j in submitted_jobs]}")

        # Now wait for all results
        print(f"\n{'='*70}")
        print("WAITING FOR RESULTS")
        print(f"{'='*70}")

        results = []
        for phi_deg, job, job_id in submitted_jobs:
            result = wait_for_job(phi_deg, job, job_id, num_shots)
            if result:
                results.append(result['positive'])
                results.append(result['negative'])

    else:
        # Simulator mode
        simulator = AerSimulator(method='matrix_product_state')
        results = []

        for angle in test_angles:
            phi_rad = np.radians(angle)
            print(f"\n  Running φ = ±{angle}° on simulator...")

            qc = create_leggett_circuit_70_pairs(phi_rad)
            result = simulator.run(qc, shots=num_shots).result()
            counts = result.get_counts()

            correlations_pos, correlations_neg = extract_correlations_from_counts_70_pairs(counts, num_shots)

            result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
            result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

            print(f"    +{angle}°: L₃={result_pos['L3']:.4f}, bound={result_pos['bound']:.4f}")
            print(f"    -{angle}°: L₃={result_neg['L3']:.4f}, bound={result_neg['bound']:.4f}")

            results.append({
                'phi_deg': angle,
                'phi_rad': phi_rad,
                **result_pos,
                'job_id': 'simulator',
                'num_shots': num_shots,
                'num_pairs': 70,
                'timestamp': datetime.now().isoformat()
            })
            results.append({
                'phi_deg': -angle,
                'phi_rad': -phi_rad,
                **result_neg,
                'job_id': 'simulator',
                'num_shots': num_shots,
                'num_pairs': 70,
                'timestamp': datetime.now().isoformat()
            })

    # Sort results by angle
    results.sort(key=lambda x: x['phi_deg'])

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    violations = sum(1 for r in results if r['violated'])
    print(f"\nTotal angles tested: {len(results)}")
    print(f"Violations: {violations}/{len(results)}")

    print(f"\n{'Angle':>8} {'L₃(exp)':>10} {'L₃(th)':>10} {'Bound':>10} {'Status':>12}")
    print("-"*60)

    for r in results:
        status = "VIOLATION" if r['violated'] else "No violation"
        print(f"{r['phi_deg']:>+8.1f}° {r['L3']:>10.4f} {r['L3_theory']:>10.4f} {r['bound']:>10.4f} {status:>12}")

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Done!")


if __name__ == "__main__":
    main()
