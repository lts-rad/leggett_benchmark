import os
#!/usr/bin/env python3
"""
Leggett Inequality Test using IBM Quantum - Sequential Version
Based on arXiv:0801.2241v2

This version runs SEPARATE jobs for each angle to avoid large circuit decoherence.
Each job measures 6 correlations using 6 independent singlet pairs = 12 qubits total.
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

from leggett import create_leggett_circuit_for_angle_0011_six, extract_correlations_from_counts_six, extract_correlations_from_counts, calc_leggett_for_angle

def run_single_angle(phi_deg, num_shots, use_ibm, backend_name, dry_run=False, enable_error_mitigation=True, use_noise_model=False):
    """
    Run Leggett test for a single angle φ.

    Args:
        phi_deg: Angle in degrees
        num_shots: Number of measurements
        use_ibm: If True, run on IBM Quantum
        backend_name: Name of IBM backend (e.g., 'ibm_pittsburgh')
        dry_run: If True, only transpile without submitting
        enable_error_mitigation: If True, enable error mitigation
        use_noise_model: If True, simulate with IBM backend's noise model

    Returns:
        Dictionary with results: correlations, L3, L3_theory, bound, violated, job_id, etc.
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running complementary angles φ = ±{abs(phi_deg):.1f}° ({num_shots} shots, 12 qubits)")
    print(f"{'='*70}")

    # Create circuit
    qc = create_leggett_circuit_for_angle_0011_six(phi_rad)

    # Run on IBM or simulator
    if use_ibm:
        # Connect to IBM Quantum
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

        print(f"  Transpiling for {backend_name}...")
        print(f"  Original circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

        if dry_run:
            print(f"  DRY RUN: Would submit job here. Exiting.")
            return None

        # Configure error mitigation
        options = SamplerOptions()
        if enable_error_mitigation:
            options.execution.init_qubits = True
            options.dynamical_decoupling.enable = True
            options.dynamical_decoupling.sequence_type = "XX"
            options.twirling.enable_gates = True
            options.twirling.enable_measure = True
            print(f"  Error mitigation: twirling + dynamical decoupling (XY4) enabled")
        else:
            print(f"  Error mitigation: disabled")

        # Run with Sampler
        print(f"  Submitting job...")
        sampler = Sampler(backend, options=options)
        job = sampler.run([qc_transpiled], shots=num_shots)
        job_id = job.job_id()
        print(f"  Job ID: {job_id}")
        print(f"  Waiting for results (this may take hours depending on queue)...")
        print(f"  You can check status at: https://quantum.ibm.com/jobs/{job_id}")

        # Wait with status updates
        import time
        start_time = time.time()
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            print(f"  Status: {status} (elapsed: {elapsed/60:.1f} min)", end='\r')

            if status in ['DONE', 'COMPLETED', 'ERROR', 'CANCELLED']:
                print()  # New line after status updates
                break

            time.sleep(30)  # Check every 30 seconds

        if status == 'ERROR' or status == 'CANCELLED':
            print(f"  Job {status}! Skipping this angle.")
            return None

        result = job.result()

        # Extract counts from PUB result
        pub_result = result[0]
        counts_array = pub_result.data.meas.get_counts()

        # Convert to standard counts dictionary
        counts = {}
        for bitstring, count in counts_array.items():
            counts[bitstring] = count

    else:
        if use_noise_model:
            # Use noise model from IBM backend
            print(f"  Running on local simulator with {backend_name} noise model...")
            service = QiskitRuntimeService()
            backend = service.backend(backend_name)

            # Get noise model from backend
            from qiskit_aer.noise import NoiseModel
            noise_model = NoiseModel.from_backend(backend)

            # Transpile for the backend
            print(f"  Transpiling for noise model...")
            qc_transpiled = transpile(qc, backend=backend, optimization_level=3)

            print(f"Running in aer now")
            # Run with noise
            simulator = AerSimulator(noise_model=noise_model)
            result = simulator.run(qc_transpiled, shots=num_shots).result()
            counts = result.get_counts()
            job_id = f"noise_model_{backend_name}"
        else:
            # Perfect noiseless simulation
            print(f"  Running on local noiseless simulator...")
            simulator = AerSimulator(method='matrix_product_state')
            result = simulator.run(qc, shots=num_shots).result()
            counts = result.get_counts()
            job_id = "local_simulator_noiseless"

    # Extract correlations for both +phi and -phi
    correlations_pos = extract_correlations_from_counts_six(counts, num_shots)


    # Calculate results for both angles
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)

    print(f"\n  Results for φ = +{(phi_deg):.1f}°:")
    print(f"    Correlations (exp): {result_pos['correlations']}")
    print(f"    Correlations (th):  {result_pos['correlations_theory']}")
    print(f"    L₃ (exp):      {result_pos['L3']:.4f}")
    print(f"    L₃ (theory):   {result_pos['L3_theory']:.4f}")
    print(f"    L₃ bound:      {result_pos['bound']:.4f}")
    print(f"    Violated:      {result_pos['violated']}")


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
        'counts': counts
    }


def main():
    import sys

    # Parse command line arguments
    use_ibm = '--ibm' in sys.argv
    dry_run = '--dry-run' in sys.argv
    use_noise_model = '--noise-model' in sys.argv
    backend_name = "ibm_pittsburgh"  # Default backend
    num_shots = 1000  # Default shots
    # Default output file depends on whether using real hardware or simulator
    output_file = None  # Will be set based on use_ibm
    enable_error_mitigation = '--no-error-mitigation' not in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    # Set default output file if not specified
    if output_file is None:
        if use_ibm:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_{backend_name}_sequential_b0011_12qb.json')
        elif use_noise_model:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_{backend_name}_NOISE_MODEL_b0011_12qb.json')
        else:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'leggett_results_ibm_sequential_SIM_noiseless_b0011_12qb.json')

    print("="*70)
    print("LEGGETT INEQUALITY TEST: Sequential Jobs with 12 Qubits (IBM Quantum)")
    print("="*70)
    print(f"\nConfiguration:")
    if use_ibm:
        print(f"  Backend: IBM Quantum - {backend_name} (REAL HARDWARE)")
    elif use_noise_model:
        print(f"  Backend: Local simulator with {backend_name} noise model")
    else:
        print(f"  Backend: Local noiseless simulator")
    print(f"  Shots per angle pair: {num_shots}")
    print(f"  Circuit size: 12 qubits (12 singlet pairs, testing ±φ simultaneously)")
    print(f"  Error mitigation: {'enabled' if enable_error_mitigation else 'disabled'}")
    print(f"  Dry run: {dry_run}")
    print(f"  Output file: {output_file}")

    test_angles = [15, 25, 30, 45, 60, -15, -25, -30, -45, -60]

    print(f"\nTest angles: {test_angles}")
    print(f"Total jobs to submit: {len(test_angles)}")

    if dry_run and use_ibm:
        print("\n*** DRY RUN MODE: Will transpile one circuit and exit ***")
        run_single_angle(test_angles[0], num_shots, use_ibm, backend_name, dry_run=True, enable_error_mitigation=enable_error_mitigation)
        return

    # Run all angles
    results = []

    for angle in test_angles:
        result = run_single_angle(angle, num_shots, use_ibm, backend_name,
                                 enable_error_mitigation=enable_error_mitigation,
                                 use_noise_model=use_noise_model)
        if result:
            # Add both positive and negative angle results to results list
            results.append(result['positive'])

        # Brief pause between job submissions when using IBM
        if use_ibm and angle != test_angles[-1]:
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
