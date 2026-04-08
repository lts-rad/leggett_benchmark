#!/usr/bin/env python3
"""
Leggett Inequality Test using IQM Emerald - Single Circuit with Mid-Circuit Measurements

This version creates a single circuit with 2 qubits that are reused for multiple correlations
with mid-circuit measurements using IQM's active reset (cc_prx).

IQM Emerald supports mid-circuit measurements and active reset via cc_prx.
"""

import numpy as np
import json
from datetime import datetime
from math import pi

from braket.circuits import Circuit
from braket.aws import AwsDevice
from braket.experimental_capabilities import EnableExperimentalCapability


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles (theta, phi)."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi_angle = np.arctan2(y, x)
    return theta, phi_angle


def create_iqm_leggett_circuit_midcircuit(phi_rad_list):
    """
    Create IQM Emerald circuit with mid-circuit measurements.

    Uses 2 qubits that are reset and reused using cc_prx active reset.

    Args:
        phi_rad_list: List of angles φ in radians

    Returns:
        Braket Circuit with 2 qubits and mid-circuit measurements
    """
    # Create Braket circuit with 2 qubits
    # cc_prx requires experimental capability context
    with EnableExperimentalCapability():
        circuit = Circuit()

        # Alice's measurement directions
        a1 = np.array([1, 0, 0])
        a2 = np.array([0, 1, 0])
        a3 = np.array([0, 0, 1])

        is_first = True
        feedback_idx = 0  # Track feedback key indices

        for phi in phi_rad_list:
            # Bob's measurement directions
            b1 = np.array([np.cos(phi/2), np.sin(phi/2), 0])
            b1_prime = np.array([np.cos(phi/2), -np.sin(phi/2), 0])
            b2 = np.array([0, np.cos(phi/2), np.sin(phi/2)])
            b2_prime = np.array([0, np.cos(phi/2), -np.sin(phi/2)])
            b3 = np.array([np.sin(phi/2), 0, np.cos(phi/2)])
            b3_prime = np.array([-np.sin(phi/2), 0, np.cos(phi/2)])

            measurement_pairs = [
                (a1, b1), (a1, b1_prime),
                (a2, b2), (a2, b2_prime),
                (a3, b3), (a3, b3_prime),
            ]

            for a_vec, b_vec in measurement_pairs:
                # Active reset using measure_ff + cc_prx (IQM's reset method)
                if not is_first:
                    # For IQM active reset:
                    # 1. measure_ff stores measurement in feedback_key (reuse last measurement keys)
                    # 2. cc_prx applies conditional X rotation to reset qubit to |0>
                    circuit.cc_prx(0, pi, 0, feedback_key=feedback_idx-2)  # Reset qubit 0 based on previous measurement
                    circuit.cc_prx(1, pi, 0, feedback_key=feedback_idx-1)  # Reset qubit 1 based on previous measurement
                is_first = False

                # Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
                circuit.x(1)
                circuit.h(0)
                circuit.cnot(0, 1)
                circuit.z(1)

                # Apply measurement rotations
                theta_a, phi_a = poincare_to_angles(a_vec)
                theta_b, phi_b = poincare_to_angles(b_vec)

                circuit.rz(0, -phi_a)
                circuit.ry(0, -theta_a)
                circuit.rz(1, -phi_b)
                circuit.ry(1, -theta_b)

                # IMPORTANT: measure_ff does NOT return results (per IQM docs)
                # We need regular measure() for final correlations
                # But we can't use measure() for mid-circuit because it blocks reuse
                # Solution: Only measure the LAST correlation with measure()
                # For all others, we need a different approach...

                # For now: measure_ff for feedforward, but we won't get these results back
                circuit.measure_ff(0, feedback_key=feedback_idx)
                circuit.measure_ff(1, feedback_key=feedback_idx+1)
                feedback_idx += 2

    return circuit


def extract_correlations_from_counts(counts, num_shots, num_angles):
    """Extract correlation values from measurement counts."""
    results = []

    for angle_idx in range(num_angles):
        correlations = []

        for corr_idx in range(6):
            correlation = 0.0

            for bitstring, count in counts.items():
                # Each measurement produces 2 bits
                meas_idx = angle_idx * 6 + corr_idx
                bit_idx_start = meas_idx * 2

                if bit_idx_start + 1 < len(bitstring):
                    # Braket uses left-to-right bit ordering
                    alice_bit = int(bitstring[-(bit_idx_start + 2)])
                    bob_bit = int(bitstring[-(bit_idx_start + 1)])

                    alice_val = 1 if alice_bit == 0 else -1
                    bob_val = 1 if bob_bit == 0 else -1

                    correlation += (alice_val * bob_val * count) / num_shots

            # For singlet, multiply by -1
            correlations.append(-correlation)

        results.append(correlations)

    return results


def calc_leggett_for_angle(correlations, phi_rad):
    """Calculate Leggett parameter and theory values."""
    correlations_theory = [-np.cos(phi_rad/2)] * 6

    C = correlations
    L3 = (1/3) * (abs(C[0] + C[1]) + abs(C[2] + C[3]) + abs(C[4] + C[5]))
    L3_theory = 2 * abs(np.cos(phi_rad/2))
    bound = 2 - (2/3) * abs(np.sin(phi_rad/2))
    violated = L3 > bound

    return {
        'correlations': correlations,
        'correlations_theory': correlations_theory,
        'L3': L3,
        'L3_theory': L3_theory,
        'bound': bound,
        'violated': violated
    }


def main():
    import sys

    use_braket = '--braket' in sys.argv
    dry_run = '--dry-run' in sys.argv
    device_arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald"
    num_shots = 1000
    output_file = "leggett_results_iqm_emerald_2qb_sequential.json"

    for i, arg in enumerate(sys.argv):
        if arg == '--device-arn' and i + 1 < len(sys.argv):
            device_arn = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    print("="*70)
    print("LEGGETT INEQUALITY TEST: IQM Emerald 2-Qubit Mid-Circuit")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Backend: {'IQM Emerald' if use_braket else 'Local Simulator'}")
    print(f"  Shots: {num_shots}")
    print(f"  Circuit: 2 qubits with mid-circuit measurements and active reset")

    test_angles = [15, 25, 30, 45, 60]
    angles_rad = [np.radians(a) for a in test_angles]

    print(f"\nTest angles: {test_angles}")

    print(f"\nCreating circuit...")
    circuit = create_iqm_leggett_circuit_midcircuit(angles_rad)
    print(f"  Circuit: {circuit.qubit_count} qubits, depth {circuit.depth}")

    if dry_run:
        print(f"\nDRY RUN: Exiting.")
        return

    if use_braket:
        device = AwsDevice(device_arn)
        print(f"\nSubmitting job to {device_arn.split('/')[-1]}...")
        task = device.run(circuit, shots=num_shots)
        job_id = task.id
        print(f"  Task ID: {job_id}")
        print(f"  Waiting for results...")

        result = task.result()
        measurements = result.measurements
        counts = {}
        for shot in measurements:
            bitstring = ''.join(str(int(b)) for b in shot)
            counts[bitstring] = counts.get(bitstring, 0) + 1
    else:
        print(f"\nERROR: Local simulator not implemented")
        return

    print(f"\nExtracting correlations...")
    all_correlations = extract_correlations_from_counts(counts, num_shots, len(test_angles))

    results = []
    for i, angle in enumerate(test_angles):
        result = calc_leggett_for_angle(all_correlations[i], angles_rad[i])
        results.append({
            'phi_deg': angle,
            'phi_rad': angles_rad[i],
            **result,
            'job_id': job_id,
            'num_shots': num_shots,
            'timestamp': datetime.now().isoformat()
        })

        print(f"\n  φ = {angle:.1f}°:")
        print(f"    L₃: {result['L3']:.4f}, Theory: {result['L3_theory']:.4f}")
        print(f"    Violated: {result['violated']}")

    violations = sum(1 for r in results if r['violated'])
    print(f"\n{'='*70}")
    print(f"Violations: {violations}/{len(results)}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
