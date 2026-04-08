import os
#!/usr/bin/env python3
"""
Leggett Inequality Test using IBM Quantum - 2 Qubit Sequential Version

This version runs jobs with 2 qubits (1 singlet pair) at a time.
For each angle φ, we run 12 separate jobs to measure all 12 correlations
(6 for +φ and 6 for -φ).

This minimizes circuit depth and decoherence at the cost of more jobs.
"""

import numpy as np
import json
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from datetime import datetime


from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from leggett import poincare_to_angles, measure_polarization, extract_correlations_from_counts


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

    return qc



def run_leggett_test_2qb_with_noise(angles_deg, num_shots=10000, noise_model=None, noise_backend_name=''):
    """
    Run Leggett inequality test with 2 qubits at a time using a noise model.

    Args:
        angles_deg: List of angles in degrees to test
        num_shots: Number of shots per circuit
        noise_model: Qiskit NoiseModel to use
        noise_backend_name: Name of backend noise model came from (for labeling)

    Returns:
        List of result dictionaries, one per angle
    """

    # Setup simulator with noise model
    backend = AerSimulator(noise_model=noise_model)
    print(f"Using AerSimulator with noise model from: {noise_backend_name}")

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
        ]
        print("YO",phi_rad)

        correlations_pos = []

        # Run 12 separate 2-qubit jobs
        for idx, (a_vec, b_vec, label) in enumerate(measurement_pairs):
            print(f"  Job {idx+1}/12: {label} ...", end=' ', flush=True)

            # Create circuit
            qc = create_single_correlation_circuit(a_vec, b_vec)

            # Run on noisy simulator
            result = backend.run(qc, shots=num_shots).result()
            counts = result.get_counts()

            # Extract correlation
            corr = extract_correlations_from_counts(counts, num_shots)

            if idx < 6:
                correlations_pos.append(corr)

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

        print(f"\n  Results for +φ:")
        print(f"    L₃ = {L3_pos:.4f}, L_bound = {L_bound:.4f}, Violation: {violated_pos}")
        print(f"  Results for -φ:")
        print(f"    L₃ = {L3_neg:.4f}, L_bound = {L_bound:.4f}, Violation: {violated_neg}")

        # Store results (convert numpy types to Python types for JSON serialization)
        all_results.append({
            'phi_deg': float(angle_deg),
            'correlations_pos': [float(c) for c in correlations_pos],
            'correlations_theory': [float(c) for c in correlations_theory],
            'L3_pos': float(L3_pos),
            'L3_neg': float(L3_neg),
            'L3_theory': float(L3_th),
            'bound': float(L_bound),
            'violated_pos': int(violated_pos) != 0,
            'violated_neg': int(violated_neg) != 0
        })

    return all_results


def run_leggett_test_2qb(angles_deg, num_shots=10000, use_ibm=False, backend_name='ibm_brisbane'):
    """
    Run Leggett inequality test with 2 qubits at a time.

    Args:
        angles_deg: List of angles in degrees to test
        num_shots: Number of shots per circuit
        use_ibm: If True, run on IBM hardware; if False, use ideal simulator
        backend_name: Name of IBM backend

    Returns:
        List of result dictionaries, one per angle
    """

    # Setup
    if use_ibm:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        print(f"Using IBM backend: {backend_name}")
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

        # All 12 measurement pairs (6 for +φ)
        measurement_pairs = [
            # +φ correlations
            (a1, b1, "C(a1,b1)"),
            (a1, b1_prime, "C(a1,b1')"),
            (a2, b2, "C(a2,b2)"),
            (a2, b2_prime, "C(a2,b2')"),
            (a3, b3, "C(a3,b3)"),
            (a3, b3_prime, "C(a3,b3')"),
        ]

        correlations_pos = []

        # Run 12 separate 2-qubit jobs
        for idx, (a_vec, b_vec, label) in enumerate(measurement_pairs):
            print(f"  Job {idx+1}/12: {label} ...", end=' ', flush=True)

            # Create circuit
            qc = create_single_correlation_circuit(a_vec, b_vec)

            if use_ibm:
                # Transpile and run on IBM hardware
                qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
                sampler = Sampler(backend)
                job = sampler.run([qc_transpiled], shots=num_shots)
                result = job.result()
                counts = result[0].data.meas.get_counts()
            else:
                # Run on ideal simulator
                result = backend.run(qc, shots=num_shots).result()
                counts = result.get_counts()

            # Extract correlation
            corr = extract_correlation_from_counts(counts, num_shots)

            if idx < 6:
                correlations_pos.append(corr)

            print(f"C = {corr:+.4f}")

            # Rate limiting for IBM
            if use_ibm and idx < 11:
                time.sleep(1)

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

        print(f"\n  Results for +φ:")
        print(f"    L₃ = {L3_pos:.4f}, L_bound = {L_bound:.4f}, Violation: {violated_pos}")
        print(f"  Results for -φ:")
        print(f"    L₃ = {L3_neg:.4f}, L_bound = {L_bound:.4f}, Violation: {violated_neg}")

        # Store results (convert numpy types to Python types for JSON serialization)
        all_results.append({
            'phi_deg': float(angle_deg),
            'correlations_pos': [float(c) for c in correlations_pos],
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

    parser = argparse.ArgumentParser(description='Run Leggett inequality test with 2 qubits sequentially')
    parser.add_argument('--hardware', action='store_true', help='Run on IBM hardware')
    parser.add_argument('--backend', type=str, default='ibm_brisbane', help='IBM backend name')
    parser.add_argument('--noise-model', type=str, help='Use noise model from specified backend (e.g., ibm_pittsburgh)')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots per circuit')
    args = parser.parse_args()

    # Test angles
    angles_deg = [-60, -45, -30, -25, -15, 15, 25, 30, 45, 60]
    angles_deg=[30]

    # Run test
    print("="*70)
    print("LEGGETT INEQUALITY TEST - 2 QUBIT SEQUENTIAL MODE")
    print("="*70)

    # Determine mode
    USE_IBM_HARDWARE = args.hardware
    USE_NOISE_MODEL = args.noise_model is not None
    BACKEND_NAME = args.backend
    NUM_SHOTS = args.shots

    if USE_NOISE_MODEL:
        # Need to modify the function to accept noise model
        # For now, call a modified version
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        noise_backend = service.backend(args.noise_model)
        noise_model = NoiseModel.from_backend(noise_backend)

        print(f"Using noise model from: {args.noise_model}")

        # We need to pass noise_model to the simulator
        # Let's create a custom version
        results = run_leggett_test_2qb_with_noise(
            angles_deg=angles_deg,
            num_shots=NUM_SHOTS,
            noise_model=noise_model,
            noise_backend_name=args.noise_model
        )
    else:
        results = run_leggett_test_2qb(
            angles_deg=angles_deg,
            num_shots=NUM_SHOTS,
            use_ibm=USE_IBM_HARDWARE,
            backend_name=BACKEND_NAME
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if USE_IBM_HARDWARE:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_2qb_{BACKEND_NAME}_{timestamp}.json')
    elif USE_NOISE_MODEL:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_2qb_noise_{args.noise_model}_{timestamp}.json')
    else:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_2qb_ideal_sim_{timestamp}.json')

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
