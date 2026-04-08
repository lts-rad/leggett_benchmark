import os
#!/usr/bin/env python3
"""
Leggett Inequality Test - ALL ANGLES IN ONE CIRCUIT

This version creates ONE circuit that tests all 10 angles at once.
Uses 2 qubits with mid-circuit measurements, measuring 120 correlations total
(12 correlations per angle × 10 angles).

This is the most efficient approach for IBM hardware with mid-circuit measurement support.
"""

import sys, os

from ibm_leggett_2qb_single import (
    create_leggett_circuit_midcircuit,
    extract_correlations_from_midcircuit_counts
)

import numpy as np
import json
from datetime import datetime
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Leggett test with ALL angles in one circuit (IBM)')
    parser.add_argument('--hardware', action='store_true', help='Run on IBM hardware')
    parser.add_argument('--backend', type=str, default='ibm_brisbane', help='IBM backend name')
    parser.add_argument('--noise-model', type=str, help='Use noise model from specified backend')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots per circuit')
    args = parser.parse_args()

    # Test angles
    angles_deg = [-60, -45, -30, -25, -15, 15, 25, 30, 45, 60]

    print("="*70)
    print("LEGGETT INEQUALITY TEST - ALL ANGLES IN ONE CIRCUIT (IBM)")
    print("="*70)

    USE_IBM_HARDWARE = args.hardware
    USE_NOISE_MODEL = args.noise_model is not None
    BACKEND_NAME = args.backend
    NUM_SHOTS = args.shots

    # Setup backend
    if USE_IBM_HARDWARE:
        service = QiskitRuntimeService()
        backend = service.backend(BACKEND_NAME)
        print(f"Using IBM backend: {BACKEND_NAME}")
    elif USE_NOISE_MODEL:
        from qiskit_aer.noise import NoiseModel
        service = QiskitRuntimeService()
        noise_backend = service.backend(args.noise_model)
        noise_model = NoiseModel.from_backend(noise_backend)
        backend = AerSimulator(noise_model=noise_model)
        print(f"Using AerSimulator with noise model from: {args.noise_model}")
    else:
        backend = AerSimulator()
        print("Using ideal simulator (AerSimulator)")

    # Create single circuit for all angles
    angles_rad = [np.radians(a) for a in angles_deg]
    print(f"\nCreating circuit with ALL {len(angles_deg)} angles...")
    print(f"  Total correlations: {len(angles_deg) * 6}")
    qc = create_leggett_circuit_midcircuit(angles_rad)
    print(f"  Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    # Run circuit
    if USE_IBM_HARDWARE:
        print(f"\n  Transpiling for {BACKEND_NAME}...")
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

        options = SamplerOptions()
        options.execution.init_qubits = True

        print(f"  Submitting job...")
        sampler = Sampler(backend, options=options)
        job = sampler.run([qc_transpiled], shots=NUM_SHOTS)
        print(f"  Job ID: {job.job_id()}")
        print(f"  Waiting for results...")

        result = job.result()
        pub_result = result[0]

        # Extract counts from all classical registers
        counts = {}
        total_cregs = len(angles_deg) * 6

        creg_int_arrays = []
        for creg_idx in range(total_cregs):
            creg_name = f'c{creg_idx}'
            bit_array = getattr(pub_result.data, creg_name)
            int_array = bit_array.array.reshape(-1)
            creg_int_arrays.append(int_array)

        for shot_idx in range(NUM_SHOTS):
            outcome_parts = []
            for creg_idx in range(total_cregs):
                bit_val = int(creg_int_arrays[creg_idx][shot_idx])
                outcome_parts.append(f"{bit_val:02b}")

            outcome_str = " ".join(reversed(outcome_parts))
            counts[outcome_str] = counts.get(outcome_str, 0) + 1
    else:
        print(f"  Running on simulator...")
        result = backend.run(qc, shots=NUM_SHOTS).result()
        counts = result.get_counts()

    # Extract correlations for all angles
    all_angle_correlations = extract_correlations_from_midcircuit_counts(counts, NUM_SHOTS, len(angles_deg))

    # Process results for each angle
    results = []
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    for angle_deg, correlations in zip(angles_deg, all_angle_correlations):
        phi_rad = np.radians(angle_deg)

        b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
        b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
        b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
        b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
        b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
        b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

        C_a1b1_th = -np.dot(a1, b1)
        C_a1b1p_th = -np.dot(a1, b1_prime)
        C_a2b2_th = -np.dot(a2, b2)
        C_a2b2p_th = -np.dot(a2, b2_prime)
        C_a3b3_th = -np.dot(a3, b3)
        C_a3b3p_th = -np.dot(a3, b3_prime)

        correlations_theory = [C_a1b1_th, C_a1b1p_th, C_a2b2_th,
                               C_a2b2p_th, C_a3b3_th, C_a3b3p_th]

        L3_exp = (1/3) * (abs(correlations[0] + correlations[1]) +
                          abs(correlations[2] + correlations[3]) +
                          abs(correlations[4] + correlations[5]))

        L3_th = (1/3) * (abs(C_a1b1_th + C_a1b1p_th) +
                         abs(C_a2b2_th + C_a2b2p_th) +
                         abs(C_a3b3_th + C_a3b3p_th))

        L_bound = 2 - (2/3) * abs(np.sin(phi_rad/2))
        violated = L3_exp > L_bound

        print(f"  φ = {angle_deg:+4.0f}°: L₃(exp) = {L3_exp:.4f}, L₃(th) = {L3_th:.4f}, bound = {L_bound:.4f}, Violated: {violated}")

        results.append({
            'phi_deg': float(angle_deg),
            'correlations': [float(c) for c in correlations],
            'correlations_theory': [float(c) for c in correlations_theory],
            'L3_exp': float(L3_exp),
            'L3_theory': float(L3_th),
            'bound': float(L_bound),
            'violated': int(violated) != 0
        })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if USE_IBM_HARDWARE:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_all_angles_{BACKEND_NAME}_{timestamp}.json')
    elif USE_NOISE_MODEL:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_all_angles_noise_{args.noise_model}_{timestamp}.json')
    else:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_all_angles_ideal_sim_{timestamp}.json')

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Summary
    violations = sum(1 for r in results if r['violated'])

    print(f"\nSummary:")
    print(f"  Angles tested: {len(angles_deg)}")
    print(f"  Violations: {violations}/{len(angles_deg)}")
    print(f"\nThis single circuit tested ALL {len(angles_deg)} angles with {len(angles_deg) * 6} total correlations!")
