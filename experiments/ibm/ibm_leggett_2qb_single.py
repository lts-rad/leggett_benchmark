import os
#!/usr/bin/env python3
"""
Leggett Inequality Test using IBM Quantum - Single Circuit with Mid-Circuit Measurements

This version creates a single circuit with 2 qubits that are reused 12 times
with mid-circuit measurements and barriers between each correlation measurement.
For each angle φ, all 12 correlations (6 for +φ, 6 for -φ) are measured in one circuit.

IBM hardware supports mid-circuit measurements, allowing qubit reuse.
"""

import numpy as np
import json
from datetime import datetime

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions


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


def create_leggett_circuit_midcircuit(phi_rad):
    """
    Create a circuit with mid-circuit measurements for all 12 correlations.

    Uses 2 qubits (1 singlet pair) that are reset and reused.
    Each measurement is separated by a barrier.

    Args:
        phi_rad: Angle φ in radians (will test both +phi and -phi), or list/array of angles

    Returns:
        QuantumCircuit with 2 qubits and 12*len(angles) classical registers
    """
    # Handle both single angle and array of angles
    if isinstance(phi_rad, (list, np.ndarray)):
        angles_rad = phi_rad
    else:
        angles_rad = [phi_rad]

    # 2 qubits
    qc = QuantumCircuit(2)

    # Alice's measurement directions (same for all)
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    creg_idx = 0

    for phi in angles_rad:
        # Bob's measurement directions for this phi angle
        b1 = np.array([np.cos(phi/2), np.sin(phi/2), 0])
        b1_prime = np.array([np.cos(phi/2), -np.sin(phi/2), 0])
        b2 = np.array([0, np.cos(phi/2), np.sin(phi/2)])
        b2_prime = np.array([0, np.cos(phi/2), -np.sin(phi/2)])
        b3 = np.array([np.sin(phi/2), 0, np.cos(phi/2)])
        b3_prime = np.array([-np.sin(phi/2), 0, np.cos(phi/2)])

        # 6 correlations for this angle
        measurement_pairs = [
            (a1, b1),
            (a1, b1_prime),
            (a2, b2),
            (a2, b2_prime),
            (a3, b3),
            (a3, b3_prime),
        ]

        # Measure each correlation sequentially with mid-circuit measurements
        for i, (a_vec, b_vec) in enumerate(measurement_pairs):
            # Create classical register for this correlation
            creg = ClassicalRegister(2, name=f'c{creg_idx}')
            qc.add_register(creg)

            # Reset qubits
            if creg_idx > 0:
                qc.reset(0)
                qc.reset(1)

            qc.barrier()  # Barrier before each correlation measurement

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

            # Barrier before measurement to ensure rotations are complete
            qc.barrier()

            # Measure into dedicated classical register
            qc.measure([0, 1], creg)

            creg_idx += 1

    return qc


def extract_correlations_from_midcircuit_counts(counts, num_shots, num_angles=1):
    """
    Extract correlation values from mid-circuit measurement counts.

    Args:
        counts: Dictionary of measurement outcomes
        num_shots: Total number of shots
        num_angles: Number of angles tested (default 1)

    Returns:
        If num_angles == 1: List of 6 correlations
        If num_angles > 1: List of lists, each with 6 correlations
    """
    total_correlations = num_angles * 6
    correlations = [0.0] * total_correlations

    for outcome_str, count in counts.items():
        # outcome_str format: "cN ... c1 c0"
        # Each c_i is 2 bits (qubit1 qubit0)
        # Split by spaces
        creg_results = outcome_str.split()

        # Process all classical registers
        for i in range(total_correlations):
            # Get the 2-bit result for this correlation
            bits = creg_results[total_correlations - 1 - i]  # Reverse order
            alice_bit = int(bits[1])  # qubit 0
            bob_bit = int(bits[0])    # qubit 1

            # Correlation for singlet: same bits → +1, different bits → -1
            if alice_bit == bob_bit:
                correlations[i] += count
            else:
                correlations[i] -= count

    # Normalize
    correlations = [c / num_shots for c in correlations]

    # Split into angle results
    if num_angles == 1:
        # Return single list of 6 correlations
        return correlations[0:6]
    else:
        # Return list of lists
        angle_results = []
        for angle_idx in range(num_angles):
            start_idx = angle_idx * 6
            angle_corrs = correlations[start_idx:start_idx + 6]
            angle_results.append(angle_corrs)
        return angle_results


def run_leggett_test_midcircuit(angles_deg, num_shots=1000, use_ibm=False,
                                backend_name='ibm_brisbane', noise_model=None,
                                noise_backend_name=''):
    """
    Run Leggett inequality test with mid-circuit measurements.

    Creates ONE circuit with all angles, submits ONE job, then extracts all results.

    Args:
        angles_deg: List of angles in degrees to test
        num_shots: Number of shots per circuit
        use_ibm: If True, run on IBM hardware; if False, use simulator
        backend_name: Name of IBM backend
        noise_model: Optional Qiskit NoiseModel
        noise_backend_name: Name of backend noise model came from

    Returns:
        List of result dictionaries, one per angle
    """

    # Setup
    if use_ibm:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        print(f"Using IBM backend: {backend_name}")
    elif noise_model:
        backend = AerSimulator(noise_model=noise_model)
        print(f"Using AerSimulator with noise model from: {noise_backend_name}")
    else:
        backend = AerSimulator()
        print("Using ideal simulator (AerSimulator)")

    # Create ONE circuit with ALL angles
    angles_rad = [np.radians(a) for a in angles_deg]
    print(f"\n{'='*70}")
    print(f"Creating circuit with ALL {len(angles_deg)} angles")
    print(f"  Total correlations: {len(angles_deg) * 12}")
    print(f"{'='*70}")
    qc = create_leggett_circuit_midcircuit(angles_rad)
    print(f"  Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    # Run the single circuit
    if use_ibm:
        # Transpile and run on IBM hardware
        print(f"\n  Transpiling for {backend_name}...")
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

        # Configure sampler
        options = SamplerOptions()
        options.execution.init_qubits = True

        print(f"  Submitting job...")
        sampler = Sampler(backend, options=options)
        job = sampler.run([qc_transpiled], shots=num_shots)
        print(f"  Job ID: {job.job_id()}")
        print(f"  Waiting for results...")

        result = job.result()
        pub_result = result[0]

        # Extract counts from ALL classical registers (6 * num_angles)
        counts = {}
        total_cregs = len(angles_deg) * 6

        creg_int_arrays = []
        for creg_idx in range(total_cregs):
            creg_name = f'c{creg_idx}'
            bit_array = getattr(pub_result.data, creg_name)
            int_array = bit_array.array.reshape(-1)
            creg_int_arrays.append(int_array)

        # Reconstruct full measurement strings for each shot
        for shot_idx in range(num_shots):
            outcome_parts = []
            for creg_idx in range(total_cregs):
                bit_val = int(creg_int_arrays[creg_idx][shot_idx])
                outcome_parts.append(f"{bit_val:02b}")

            outcome_str = " ".join(reversed(outcome_parts))
            counts[outcome_str] = counts.get(outcome_str, 0) + 1
    else:
        # Run on simulator
        print(f"  Running on simulator...")
        result = backend.run(qc, shots=num_shots).result()
        counts = result.get_counts()

    # Extract correlations for ALL angles
    all_angle_correlations = extract_correlations_from_midcircuit_counts(counts, num_shots, len(angles_deg))

    # Process results for each angle
    all_results = []
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    for angle_deg, correlations in zip(angles_deg, all_angle_correlations):
        phi_rad = np.radians(angle_deg)

        # Bob's measurement directions for this angle (for theoretical calculations)
        b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
        b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
        b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
        b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
        b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
        b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

        # Calculate theoretical correlations
        C_a1b1_th = -np.dot(a1, b1)
        C_a1b1p_th = -np.dot(a1, b1_prime)
        C_a2b2_th = -np.dot(a2, b2)
        C_a2b2p_th = -np.dot(a2, b2_prime)
        C_a3b3_th = -np.dot(a3, b3)
        C_a3b3p_th = -np.dot(a3, b3_prime)

        correlations_theory = [C_a1b1_th, C_a1b1p_th, C_a2b2_th,
                               C_a2b2p_th, C_a3b3_th, C_a3b3p_th]

        # Calculate L3
        L3_exp = (1/3) * (abs(correlations[0] + correlations[1]) +
                          abs(correlations[2] + correlations[3]) +
                          abs(correlations[4] + correlations[5]))

        L3_th = (1/3) * (abs(C_a1b1_th + C_a1b1p_th) +
                         abs(C_a2b2_th + C_a2b2p_th) +
                         abs(C_a3b3_th + C_a3b3p_th))

        L_bound = 2 - (2/3) * abs(np.sin(phi_rad/2))

        violated = L3_exp > L_bound

        print(f"  φ = {angle_deg:+4.0f}°: L₃(exp)={L3_exp:.4f}, L₃(th)={L3_th:.4f}, bound={L_bound:.4f}, Violated: {violated}")

        # Store results (convert numpy types to Python types for JSON serialization)
        all_results.append({
            'phi_deg': float(angle_deg),
            'correlations': [float(c) for c in correlations],
            'correlations_theory': [float(c) for c in correlations_theory],
            'L3_exp': float(L3_exp),
            'L3_theory': float(L3_th),
            'bound': float(L_bound),
            'violated': int(violated) != 0
        })

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Leggett test with mid-circuit measurements (IBM)')
    parser.add_argument('--hardware', action='store_true', help='Run on IBM hardware')
    parser.add_argument('--backend', type=str, default='ibm_brisbane', help='IBM backend name')
    parser.add_argument('--noise-model', type=str, help='Use noise model from specified backend')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots per circuit')
    args = parser.parse_args()

    angles_deg = [-60, -45, -30, -25, -15, 15, 25, 30, 45, 60]

    # Run test
    print("="*70)
    print("LEGGETT INEQUALITY TEST - MID-CIRCUIT MEASUREMENT MODE (IBM)")
    print("="*70)

    # Determine mode
    USE_IBM_HARDWARE = args.hardware
    USE_NOISE_MODEL = args.noise_model is not None
    BACKEND_NAME = args.backend
    NUM_SHOTS = args.shots

    if USE_NOISE_MODEL:
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        noise_backend = service.backend(args.noise_model)
        noise_model = NoiseModel.from_backend(noise_backend)

        results = run_leggett_test_midcircuit(
            angles_deg=angles_deg,
            num_shots=NUM_SHOTS,
            use_ibm=False,
            noise_model=noise_model,
            noise_backend_name=args.noise_model
        )
    else:
        results = run_leggett_test_midcircuit(
            angles_deg=angles_deg,
            num_shots=NUM_SHOTS,
            use_ibm=USE_IBM_HARDWARE,
            backend_name=BACKEND_NAME
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if USE_IBM_HARDWARE:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_midcircuit_{BACKEND_NAME}_{timestamp}.json')
    elif USE_NOISE_MODEL:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_midcircuit_noise_{args.noise_model}_{timestamp}.json')
    else:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_midcircuit_ideal_sim_{timestamp}.json')

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
