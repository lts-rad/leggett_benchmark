#!/usr/bin/env python3
"""
Leggett Inequality Test using IonQ via AWS Braket - 2 Qubit Sequential Version

This version runs jobs with 2 qubits (1 singlet pair) at a time.
For each angle φ, we run 12 separate jobs to measure all 12 correlations
(6 for +φ and 6 for -φ).

This minimizes circuit depth and decoherence at the cost of more jobs.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ionq import IonQProvider
import os


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles (theta, phi)."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def measure_polarization(qc, qubit, theta, phi_angle):
    """Apply measurement rotation to qubit."""
    # Correct order: RZ then RY with negative signs
    qc.rz(-phi_angle, qubit)
    qc.ry(-theta, qubit)


def create_single_correlation_circuit(a_vec, b_vec):
    """
    Create a 2-qubit circuit to measure one correlation C(a, b).

    Args:
        a_vec: Alice's measurement direction
        b_vec: Bob's measurement direction

    Returns:
        QuantumCircuit with 2 qubits measuring one singlet pair
    """
    qc = QuantumCircuit(2)

    # Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    qc.x(1)  # Start with |01⟩
    qc.h(0)
    qc.cx(0, 1)
    qc.z(1)

    # Apply measurement rotations
    theta_a, phi_a = poincare_to_angles(a_vec)
    theta_b, phi_b = poincare_to_angles(b_vec)

    measure_polarization(qc, 0, theta_a, phi_a)
    measure_polarization(qc, 1, theta_b, phi_b)

    # Measure
    qc.measure_all()

    # Remove barriers and clear global phase for Braket compatibility
    qc.global_phase = 0
    qc.data = [gate for gate in qc.data if gate.operation.name != 'barrier']

    return qc


def extract_correlation_from_counts(counts, num_shots):
    """
    Extract correlation from 2-qubit measurement counts.

    For singlet: C(a,b) = -a·b
    Convention: same bits → +1, different bits → -1
    """
    correlation = 0.0

    for bitstring, count in counts.items():
        alice_bit = int(bitstring[1])  # qubit 0
        bob_bit = int(bitstring[0])    # qubit 1

        # Same bits → +1, different bits → -1
        if alice_bit == bob_bit:
            correlation += count
        else:
            correlation -= count

    return correlation / num_shots


def run_leggett_test_2qb(angles_deg, num_shots=10000, use_ionq=False, noise_model=None):
    """
    Run Leggett inequality test with 2 qubits at a time.

    Args:
        angles_deg: List of angles in degrees to test
        num_shots: Number of shots per circuit
        use_ionq: If True, run on IonQ via native API; if False, use ideal simulator
        noise_model: Noise model name for IonQ (e.g., 'forte-1', 'aria-1')

    Returns:
        List of result dictionaries, one per angle
    """

    # Setup
    if use_ionq:
        # Load API key
        cred_file = 'cred_ionqsim'
        if os.path.exists(cred_file):
            with open(cred_file, 'r') as f:
                api_key = f.read().strip()
        else:
            api_key = os.getenv("IONQ_API_KEY")
            if not api_key:
                raise RuntimeError("No IonQ API key found! Create 'cred_ionqsim' file or set IONQ_API_KEY environment variable")

        provider = IonQProvider(api_key)
        backend = provider.get_backend("simulator")
        print(f"Using IonQ simulator via native API")
        if noise_model:
            print(f"  Noise model: {noise_model}")
    else:
        backend = AerSimulator()
        print("Using ideal simulator (AerSimulator)")

    all_results = []

    # Alice's measurement directions (fixed for all angles)
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    for angle_deg in angles_deg:
        print(f"\n{'='*70}")
        print(f"Testing φ = {angle_deg:+.0f}°")
        print(f"{'='*70}")

        phi_rad = np.radians(angle_deg)

        # Bob's measurement directions for this angle
        b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
        b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
        b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
        b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
        b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
        b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

        # All 12 measurement pairs (6 for +φ, 6 for -φ)
        measurement_pairs = [
            # +φ correlations
            (a1, b1, "C(a1,b1)"),
            (a1, b1_prime, "C(a1,b1')"),
            (a2, b2, "C(a2,b2)"),
            (a2, b2_prime, "C(a2,b2')"),
            (a3, b3, "C(a3,b3)"),
            (a3, b3_prime, "C(a3,b3')"),
            # -φ correlations
            (a1, b1, "C(a1,b1)[-φ]"),
            (a1, b1_prime, "C(a1,b1')[-φ]"),
            (a2, b2, "C(a2,b2)[-φ]"),
            (a2, b2_prime, "C(a2,b2')[-φ]"),
            (a3, b3, "C(a3,b3)[-φ]"),
            (a3, b3_prime, "C(a3,b3')[-φ]"),
        ]

        correlations_pos = []
        correlations_neg = []

        if use_ionq:
            # Phase 1: Submit all 12 jobs in parallel
            print(f"  Submitting all 12 jobs...")
            jobs = []
            for idx, (a_vec, b_vec, label) in enumerate(measurement_pairs):
                # Create and transpile circuit
                qc = create_single_correlation_circuit(a_vec, b_vec)
                qc_transpiled = transpile(qc, backend=backend, optimization_level=3)

                # Submit job with optional noise model
                if noise_model:
                    job = backend.run(qc_transpiled, shots=num_shots, noise_model=noise_model)
                else:
                    job = backend.run(qc_transpiled, shots=num_shots)

                jobs.append((idx, label, job))
                print(f"    Job {idx+1}/12: {label} submitted (ID: {job.job_id()})")

            # Phase 2: Collect all results
            print(f"\n  Waiting for all 12 jobs to complete...")
            for idx, label, job in jobs:
                print(f"    Job {idx+1}/12: {label} ...", end=' ', flush=True)
                result = job.result()
                counts = result.get_counts()

                # Extract correlation
                corr = extract_correlation_from_counts(counts, num_shots)

                if idx < 6:
                    correlations_pos.append(corr)
                else:
                    correlations_neg.append(corr)

                print(f"C = {corr:+.4f}")
        else:
            # Simulator: run sequentially (fast enough)
            for idx, (a_vec, b_vec, label) in enumerate(measurement_pairs):
                print(f"  Job {idx+1}/12: {label} ...", end=' ', flush=True)

                # Create circuit
                qc = create_single_correlation_circuit(a_vec, b_vec)

                # Run on ideal simulator
                result = backend.run(qc, shots=num_shots).result()
                counts = result.get_counts()

                # Extract correlation
                corr = extract_correlation_from_counts(counts, num_shots)

                if idx < 6:
                    correlations_pos.append(corr)
                else:
                    correlations_neg.append(corr)

                print(f"C = {corr:+.4f}")

        # Calculate theoretical correlations
        C_a1b1_th = -np.dot(a1, b1)
        C_a1b1p_th = -np.dot(a1, b1_prime)
        C_a2b2_th = -np.dot(a2, b2)
        C_a2b2p_th = -np.dot(a2, b2_prime)
        C_a3b3_th = -np.dot(a3, b3)
        C_a3b3p_th = -np.dot(a3, b3_prime)

        correlations_theory = [C_a1b1_th, C_a1b1p_th, C_a2b2_th,
                               C_a2b2p_th, C_a3b3_th, C_a3b3p_th]

        # Calculate L3 for +φ
        L3_pos = (1/3) * (abs(correlations_pos[0] + correlations_pos[1]) +
                          abs(correlations_pos[2] + correlations_pos[3]) +
                          abs(correlations_pos[4] + correlations_pos[5]))

        L3_th = (1/3) * (abs(C_a1b1_th + C_a1b1p_th) +
                         abs(C_a2b2_th + C_a2b2p_th) +
                         abs(C_a3b3_th + C_a3b3p_th))

        L_bound = 2 - (2/3) * abs(np.sin(phi_rad/2))

        violated_pos = L3_pos > L_bound

        # Calculate L3 for -φ
        L3_neg = (1/3) * (abs(correlations_neg[0] + correlations_neg[1]) +
                          abs(correlations_neg[2] + correlations_neg[3]) +
                          abs(correlations_neg[4] + correlations_neg[5]))

        violated_neg = L3_neg > L_bound

        print(f"\n  Results for +φ:")
        print(f"    L₃ = {L3_pos:.4f}, L_bound = {L_bound:.4f}, Violation: {violated_pos}")
        print(f"  Results for -φ:")
        print(f"    L₃ = {L3_neg:.4f}, L_bound = {L_bound:.4f}, Violation: {violated_neg}")

        # Store results (convert numpy types to Python types for JSON serialization)
        all_results.append({
            'phi_deg': float(angle_deg),
            'correlations_pos': [float(c) for c in correlations_pos],
            'correlations_neg': [float(c) for c in correlations_neg],
            'correlations_theory': [float(c) for c in correlations_theory],
            'L3_pos': float(L3_pos),
            'L3_neg': float(L3_neg),
            'L3_theory': float(L3_th),
            'bound': float(L_bound),
            'violated_pos': int(violated_pos) != 0,
            'violated_neg': int(violated_neg) != 0
        })

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Leggett inequality test with 2 qubits sequentially on IonQ')
    parser.add_argument('--hardware', action='store_true', help='Run on IonQ via native API')
    parser.add_argument('--noise-model', type=str, help='IonQ noise model name (forte-1, aria-1, etc.)')
    parser.add_argument('--shots', type=int, default=10000, help='Number of shots per circuit')
    args = parser.parse_args()

    # Test angles
    angles_deg = [-60, -45, -30, -25, -15, 15, 25, 30, 45, 60]

    # Run test
    print("="*70)
    print("LEGGETT INEQUALITY TEST - 2 QUBIT SEQUENTIAL MODE (IonQ)")
    print("="*70)

    # Determine mode
    USE_IONQ = args.hardware
    NOISE_MODEL = args.noise_model
    NUM_SHOTS = args.shots

    results = run_leggett_test_2qb(
        angles_deg=angles_deg,
        num_shots=NUM_SHOTS,
        use_ionq=USE_IONQ,
        noise_model=NOISE_MODEL
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if USE_IONQ:
        if NOISE_MODEL:
            output_file = f'leggett_results_ionq_2qb_{NOISE_MODEL}_{timestamp}.json'
        else:
            output_file = f'leggett_results_ionq_2qb_api_{timestamp}.json'
    else:
        output_file = f'leggett_results_ionq_2qb_ideal_sim_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Summary
    violations_pos = sum(1 for r in results if r['violated_pos'])
    violations_neg = sum(1 for r in results if r['violated_neg'])

    print(f"\nSummary:")
    print(f"  Angles tested: {len(angles_deg)}")
    print(f"  Violations (+φ): {violations_pos}/{len(angles_deg)}")
    print(f"  Violations (-φ): {violations_neg}/{len(angles_deg)}")
