#!/usr/bin/env python3
"""
Angle sweep for Leggett inequality test on IonQ simulator
Runs multiple jobs to cover a range of angles
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ionq_leggett_test import leggett_test_quantum
from qiskit_ionq import IonQProvider


def main():
    print("="*70)
    print("LEGGETT INEQUALITY ANGLE SWEEP (IonQ)")
    print("="*70)

    # Parse command line arguments
    num_shots = 100  # Default shots
    noise_model = 'forte-1'  # Use Forte-1 noise model by default
    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])

    # Load API key
    cred_file = 'cred_ionqsim'
    if os.path.exists(cred_file):
        with open(cred_file, 'r') as f:
            api_key = f.read().strip()
    else:
        api_key = os.getenv("IONQ_API_KEY")
        if not api_key:
            print("ERROR: No IonQ API key found!")
            sys.exit(1)

    # Initialize IonQ provider
    print(f"\nConnecting to IonQ...")
    provider = IonQProvider(api_key)
    backend = provider.get_backend("simulator")
    print(f"Backend: {backend.name}")

    # Angle sweep: test specific angles from 15° to 45° in 5° increments
    # We'll test positive and negative versions
    test_angles = [15, 20, 25, 30, 35, 40, 45]

    print(f"\nAngle sweep: {test_angles} (positive and negative)")
    print(f"Shots per job: {num_shots}")
    print(f"Noise model: {noise_model}")
    print(f"Total jobs: {len(test_angles)} (each tests both +φ and -φ)")
    print(f"\nNote: Each angle requires 6 correlations × 2 qubits = 12 qubits")
    print(f"      Running 2 angles per job (±φ) = 24 qubits total")

    # Store all results
    all_phi_degs = []
    all_L3_values = []
    all_L3_theoretical = []
    all_L_bounds = []
    all_violations = []

    # Run each angle pair
    for i, phi_deg in enumerate(test_angles):
        print(f"\n{'='*70}")
        print(f"Job {i+1}/{len(test_angles)}: Testing φ = ±{phi_deg}°")
        print(f"{'='*70}")

        # Test both +φ and -φ in one job (N=2)
        # We manually set the angles instead of using random generation
        print(f"  Angles: +{phi_deg}°, -{phi_deg}°")
        print(f"  Circuit: 2 angles × 6 correlations = 12 singlet pairs (24 qubits)")

        # Build measurement pairs manually for this specific angle
        from ionq_leggett_test import poincare_to_angles, quantum_correlation_multi

        phi = np.radians(phi_deg)

        # Alice's measurement directions
        a1 = np.array([1, 0, 0])
        a2 = np.array([0, 1, 0])
        a3 = np.array([0, 0, 1])

        # Bob's measurement directions for +φ
        b1_pos = np.array([np.cos(phi/2), np.sin(phi/2), 0])
        b1_prime_pos = np.array([np.cos(phi/2), -np.sin(phi/2), 0])
        b2_pos = np.array([0, np.cos(phi/2), np.sin(phi/2)])
        b2_prime_pos = np.array([0, np.cos(phi/2), -np.sin(phi/2)])
        b3_pos = np.array([np.sin(phi/2), 0, np.cos(phi/2)])
        b3_prime_pos = np.array([-np.sin(phi/2), 0, np.cos(phi/2)])

        # Bob's measurement directions for -φ
        phi_neg = -phi
        b1_neg = np.array([np.cos(phi_neg/2), np.sin(phi_neg/2), 0])
        b1_prime_neg = np.array([np.cos(phi_neg/2), -np.sin(phi_neg/2), 0])
        b2_neg = np.array([0, np.cos(phi_neg/2), np.sin(phi_neg/2)])
        b2_prime_neg = np.array([0, np.cos(phi_neg/2), -np.sin(phi_neg/2)])
        b3_neg = np.array([np.sin(phi_neg/2), 0, np.cos(phi_neg/2)])
        b3_prime_neg = np.array([-np.sin(phi_neg/2), 0, np.cos(phi_neg/2)])

        # Build all measurement pairs (12 pairs total)
        all_measurement_pairs = [
            # +φ angle (6 pairs)
            (a1, b1_pos), (a1, b1_prime_pos),
            (a2, b2_pos), (a2, b2_prime_pos),
            (a3, b3_pos), (a3, b3_prime_pos),
            # -φ angle (6 pairs)
            (a1, b1_neg), (a1, b1_prime_neg),
            (a2, b2_neg), (a2, b2_prime_neg),
            (a3, b3_neg), (a3, b3_prime_neg),
        ]

        # Run the job
        all_correlations = quantum_correlation_multi(all_measurement_pairs, num_shots, backend, dry_run=False, noise_model=noise_model)

        # Parse results for +φ
        C_a1b1_pos, C_a1b1p_pos, C_a2b2_pos, C_a2b2p_pos, C_a3b3_pos, C_a3b3p_pos = all_correlations[0:6]
        L3_pos = (1/3) * (abs(C_a1b1_pos + C_a1b1p_pos) +
                          abs(C_a2b2_pos + C_a2b2p_pos) +
                          abs(C_a3b3_pos + C_a3b3p_pos))

        # Parse results for -φ
        C_a1b1_neg, C_a1b1p_neg, C_a2b2_neg, C_a2b2p_neg, C_a3b3_neg, C_a3b3p_neg = all_correlations[6:12]
        L3_neg = (1/3) * (abs(C_a1b1_neg + C_a1b1p_neg) +
                          abs(C_a2b2_neg + C_a2b2p_neg) +
                          abs(C_a3b3_neg + C_a3b3p_neg))

        # Compute theoretical values
        C_a1b1_th = -np.dot(a1, b1_pos)
        C_a1b1p_th = -np.dot(a1, b1_prime_pos)
        C_a2b2_th = -np.dot(a2, b2_pos)
        C_a2b2p_th = -np.dot(a2, b2_prime_pos)
        C_a3b3_th = -np.dot(a3, b3_pos)
        C_a3b3p_th = -np.dot(a3, b3_prime_pos)

        L3_th = (1/3) * (abs(C_a1b1_th + C_a1b1p_th) +
                         abs(C_a2b2_th + C_a2b2p_th) +
                         abs(C_a3b3_th + C_a3b3p_th))

        # Leggett bound (same for ±φ)
        L_bound = 2 - (2/3) * abs(np.sin(phi/2))

        # Check violations
        violated_pos = L3_pos > L_bound
        violated_neg = L3_neg > L_bound

        # Store results
        all_phi_degs.extend([phi_deg, -phi_deg])
        all_L3_values.extend([L3_pos, L3_neg])
        all_L3_theoretical.extend([L3_th, L3_th])
        all_L_bounds.extend([L_bound, L_bound])
        all_violations.extend([violated_pos, violated_neg])

        # Print results for this job
        print(f"\n  Results:")
        status_pos = "VIOLATION" if violated_pos else "No violation"
        status_neg = "VIOLATION" if violated_neg else "No violation"
        print(f"    +{phi_deg:2.0f}°: L₃ = {L3_pos:.4f} (theory: {L3_th:.4f}, Δ={L3_pos-L3_th:+.4f}), bound = {L_bound:.4f} [{status_pos}]")
        print(f"    -{phi_deg:2.0f}°: L₃ = {L3_neg:.4f} (theory: {L3_th:.4f}, Δ={L3_neg-L3_th:+.4f}), bound = {L_bound:.4f} [{status_neg}]")

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Angle Sweep Results")
    print(f"{'='*70}")
    print(f"Total angles tested: {len(all_phi_degs)}")
    print(f"Violations: {sum(all_violations)}/{len(all_violations)}")
    print(f"Average L₃ (experiment): {np.mean(all_L3_values):.4f}")
    print(f"Average L₃ (theoretical): {np.mean(all_L3_theoretical):.4f}")
    print(f"L₃ range: [{np.min(all_L3_values):.4f}, {np.max(all_L3_values):.4f}]")

    print(f"\nAll results:")
    print(f"{'Angle':>8s} {'L₃ (exp)':>10s} {'L₃ (theory)':>12s} {'Δ':>8s} {'Bound':>8s} {'Status':>15s}")
    print("-"*70)
    for i in range(len(all_phi_degs)):
        status = "VIOLATION" if all_violations[i] else "No violation"
        diff = all_L3_values[i] - all_L3_theoretical[i]
        print(f"{all_phi_degs[i]:+7.0f}° {all_L3_values[i]:10.4f} {all_L3_theoretical[i]:12.4f} {diff:+8.4f} {all_L_bounds[i]:8.4f} {status:>15s}")


if __name__ == "__main__":
    main()
