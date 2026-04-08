import os
#!/usr/bin/env python3
"""
Leggett Inequality Test using AWS Braket LocalEmulator for IQM Garnet

This version uses the Braket LocalEmulator which:
1. Uses real-time calibration data from IQM Garnet
2. Applies depolarizing noise based on actual device calibration
3. Uses manual transpilation to IQM native gates (PRX, CZ)

Based on arXiv:0801.2241v2
"""

import numpy as np
import json
from datetime import datetime

from braket.aws import AwsDevice
from braket.circuits import Circuit

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import calc_leggett_for_angle


def poincare_to_angles(vec):
    """Convert Poincare sphere vector to spherical angles (theta, phi)."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def create_iqm_native_circuit(a_vec, b_vec, q_alice, q_bob):
    """
    Create a circuit using IQM Garnet native gates: PRX(theta, phi) and CZ ONLY.

    IQM Garnet's native gate set in verbatim mode:
    - PRX(theta, phi) = Rz(phi) @ Rx(theta) @ Rz(-phi)
    - CZ

    Key identities:
    - Ry(theta) = PRX(theta, pi/2)
    - Rx(theta) = PRX(theta, 0)
    - H = Ry(-pi/2) @ X
    """
    circ = Circuit()

    def prx(q, theta, phi):
        """Apply PRX gate."""
        if abs(theta) > 1e-10:
            circ.prx(q, theta, phi)

    def x(q):
        """X = PRX(pi, 0) = Rx(pi)"""
        prx(q, np.pi, 0)

    def y(q):
        """Y = PRX(pi, pi/2) = Ry(pi)"""
        prx(q, np.pi, np.pi/2)

    def ry(q, theta):
        """Ry(theta) = PRX(theta, pi/2)"""
        prx(q, theta, np.pi/2)

    def h(q):
        """H = Ry(-pi/2) @ X"""
        prx(q, np.pi, 0)        # X
        prx(q, -np.pi/2, np.pi/2)  # Ry(-pi/2)

    def cnot(control, target):
        """CNOT = H(t) @ CZ @ H(t)"""
        h(target)
        circ.cz(control, target)
        h(target)

    def z(q):
        """Z = Y @ X (up to global phase)"""
        prx(q, np.pi, np.pi/2)  # Y
        prx(q, np.pi, 0)        # X

    # Create singlet state |Psi-> = (|01> - |10>)/sqrt(2)
    x(q_bob)              # X on Bob: |00> -> |01>
    h(q_alice)            # H on Alice
    cnot(q_alice, q_bob)  # CNOT
    z(q_bob)              # Z on Bob

    # Apply measurement rotations
    theta_a, phi_a = poincare_to_angles(a_vec)
    theta_b, phi_b = poincare_to_angles(b_vec)

    # Rotation to measurement basis: Rz(-phi) @ Ry(-theta) = PRX(-theta, pi/2 - phi)
    prx(q_alice, -theta_a, np.pi/2 - phi_a)
    prx(q_bob, -theta_b, np.pi/2 - phi_b)

    return circ


def run_single_correlation(a_vec, b_vec, pair, emulator, num_shots):
    """
    Run a single correlation measurement using the LocalEmulator.
    """
    q_alice, q_bob = pair

    # Create circuit with IQM native gates
    inner_circ = create_iqm_native_circuit(a_vec, b_vec, q_alice, q_bob)

    # Wrap in verbatim box for emulator
    final_circ = Circuit().add_verbatim_box(inner_circ)

    # Run on emulator
    task = emulator.run(final_circ, shots=num_shots)
    result = task.result()
    counts = result.measurement_counts

    # Calculate correlation
    n_same = 0
    n_diff = 0

    # Bitstring ordering: sorted qubit indices
    all_qubits = sorted([q_alice, q_bob])
    qubit_to_pos = {q: i for i, q in enumerate(all_qubits)}

    for bitstring, count in counts.items():
        alice_bit = int(bitstring[qubit_to_pos[q_alice]])
        bob_bit = int(bitstring[qubit_to_pos[q_bob]])

        if alice_bit == bob_bit:
            n_same += count
        else:
            n_diff += count

    correlation = (n_same - n_diff) / num_shots
    return correlation


def run_single_angle(phi_deg, num_shots, emulator, physical_pairs):
    """
    Run Leggett test for a single angle phi using the LocalEmulator.
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running angle phi = +{phi_deg:.1f} deg ({num_shots} shots)")
    print(f"{'='*70}")

    # Alice's measurement directions (fixed on Poincare sphere)
    a1 = np.array([1, 0, 0])  # +x
    a2 = np.array([0, 1, 0])  # +y
    a3 = np.array([0, 0, 1])  # +z

    # Bob's measurement directions for +phi
    b1_pos = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime_pos = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2_pos = np.array([-np.sin(phi_rad/2), np.cos(phi_rad/2), 0])
    b2_prime_pos = np.array([np.sin(phi_rad/2), np.cos(phi_rad/2), 0])
    b3_pos = np.array([0, 0, 1])
    b3_prime_pos = np.array([0, 0, 1])

    # 6 measurement configurations
    measurement_configs = [
        (a1, b1_pos, "C(a1, b1)"),
        (a1, b1_prime_pos, "C(a1, b1')"),
        (a2, b2_pos, "C(a2, b2)"),
        (a2, b2_prime_pos, "C(a2, b2')"),
        (a3, b3_pos, "C(a3, b3)"),
        (a3, b3_prime_pos, "C(a3, b3')"),
    ]

    print(f"  Using physical pairs: {physical_pairs}")

    correlations = []

    for i, ((a_vec, b_vec, label), pair) in enumerate(zip(measurement_configs, physical_pairs)):
        print(f"    {label} on pair {pair}... ", end='', flush=True)

        try:
            corr = run_single_correlation(a_vec, b_vec, pair, emulator, num_shots)
            correlations.append(corr)
            print(f"C = {corr:+.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            correlations.append(0.0)

    # Calculate Leggett result
    result = calc_leggett_for_angle(correlations, phi_rad)

    print(f"\n  Results for phi = +{abs(phi_deg):.1f} deg:")
    print(f"    Correlations (exp): {[f'{c:.4f}' for c in result['correlations']]}")
    print(f"    Correlations (th):  {[f'{c:.4f}' for c in result['correlations_theory']]}")
    print(f"    L3 (exp):      {result['L3']:.4f}")
    print(f"    L3 (theory):   {result['L3_theory']:.4f}")
    print(f"    L3 bound:      {result['bound']:.4f}")
    print(f"    Violated:      {result['violated']}")

    return {
        'phi_deg': abs(phi_deg),
        'phi_rad': phi_rad,
        **result,
        'pairs_used': physical_pairs,
        'job_id': 'local_emulator',
        'num_shots': num_shots,
        'timestamp': datetime.now().isoformat()
    }


def main():
    import sys

    num_shots = 10000
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'leggett_results_iqm_garnet_emulator_BEST6.json')

    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    print("="*70)
    print("LEGGETT INEQUALITY TEST: IQM Garnet LocalEmulator")
    print("="*70)

    # Initialize device and emulator with real calibration data
    device_arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"

    print(f"\nFetching real-time calibration data from IQM Garnet...")
    iqm_device = AwsDevice(device_arn)
    emulator = iqm_device.emulator()
    print(f"  Device: {iqm_device.name}")
    print(f"  Status: {iqm_device.status}")

    # Physical qubit pairs - BEST 6 from LocalEmulator scan with real calibration
    # Selected based on lowest correlation error from qubit_pair_scan_iqm_garnet_EMULATOR_phi30.json
    physical_pairs = [
        (6, 5), (1, 4), (15, 14), (3, 8), (17, 12), (19, 20)
    ]

    print(f"\nConfiguration:")
    print(f"  Backend: IQM Garnet LocalEmulator (real calibration noise)")
    print(f"  Native gates: PRX, CZ")
    print(f"  Shots per angle: {num_shots}")
    print(f"  Physical pairs: {physical_pairs}")

    test_angles = [-60, -45, -30, -25, -15, 15, 25, 30, 45, 60]
    print(f"\nTest angles: {test_angles}")

    results = []

    for angle in test_angles:
        result = run_single_angle(angle, num_shots, emulator, physical_pairs)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    violations = sum(1 for r in results if r['violated'])
    print(f"\nTotal angles tested: {len(results)}")
    print(f"Violations: {violations}/{len(results)}")

    print(f"\nAll results:")
    print(f"{'Angle':>8} {'L3(exp)':>10} {'L3(th)':>10} {'Bound':>10} {'Delta(exp-th)':>12} {'Status':>12}")
    print("-"*70)

    for r in results:
        status = "VIOLATION" if r['violated'] else "No violation"
        delta = r['L3'] - r['L3_theory']
        print(f"{r['phi_deg']:>+8.1f} deg {r['L3']:>10.4f} {r['L3_theory']:>10.4f} {r['bound']:>10.4f} {delta:>+12.4f} {status:>12}")

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Done!")


if __name__ == "__main__":
    main()
