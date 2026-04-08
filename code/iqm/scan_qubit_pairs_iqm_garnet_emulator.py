#!/usr/bin/env python3
"""
Scan qubit pairs for IQM Garnet using AWS Braket LocalEmulator with REAL calibration data.

This version uses the actual device calibration from IQM Garnet:
- Real per-qubit gate errors from current calibration
- Real connectivity from device topology
- Manual transpilation to IQM native gates (PRX, CZ)

IQM Garnet native gates:
- PRX(theta, phi): phased Rx gate = Rz(phi) @ Rx(theta) @ Rz(-phi)
- CZ: controlled-Z gate
"""

import numpy as np
import json
from datetime import datetime

from braket.aws import AwsDevice
from braket.circuits import Circuit


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles (theta, phi)."""
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
    - Ry(θ) = PRX(θ, π/2)   since PRX(θ, π/2) = Rz(π/2) Rx(θ) Rz(-π/2) = Ry(θ)
    - Rx(θ) = PRX(θ, 0)
    - H = Ry(π/2) @ Z = PRX(π/2, π/2) followed by Z phase
    - But simpler: use the standard decomposition without virtual Z tracking
    """
    circ = Circuit()

    def prx(q, theta, phi):
        """Apply PRX gate."""
        if abs(theta) > 1e-10:
            circ.prx(q, theta, phi)

    def x(q):
        """X = PRX(π, 0) = Rx(π)"""
        prx(q, np.pi, 0)

    def y(q):
        """Y = PRX(π, π/2) = Ry(π)"""
        prx(q, np.pi, np.pi/2)

    def ry(q, theta):
        """Ry(θ) = PRX(θ, π/2)"""
        prx(q, theta, np.pi/2)

    def rx(q, theta):
        """Rx(θ) = PRX(θ, 0)"""
        prx(q, theta, 0)

    def h(q):
        """
        Hadamard gate decomposition into PRX.
        H = Ry(π/2) @ Rz(π) = Rz(π) @ Ry(-π/2) (up to global phase)

        More precisely: H = (X + Z)/√2
        Using: H = Rz(π/2) @ Ry(π/2) @ Rz(π/2) = PRX(π/2, π) @ Rz(π/2)

        But cleanest: H = PRX(π, 0) @ PRX(π/2, π/2)
        This is: X @ Ry(π/2) which equals H up to global phase

        Let's verify: Ry(π/2) @ X = Ry(π/2) @ Rx(π)
        = [[cos(π/4), -sin(π/4)], [sin(π/4), cos(π/4)]] @ [[0, 1], [1, 0]]
        = [[−sin(π/4), cos(π/4)], [cos(π/4), sin(π/4)]]
        That's not quite H.

        Correct: H = Ry(π/2) @ Z
        But Z requires decomposition. Instead use:
        H = Rx(π) @ Ry(π/2) = X @ Ry(π/2) (this gives -iH, same measurement)
        Or: H = Ry(-π/2) @ X

        Actually cleanest for IQM: H = PRX(-π/2, π/2) @ PRX(π, 0)
        = Ry(-π/2) @ X
        """
        # H = Ry(-π/2) @ X = PRX(-π/2, π/2) @ PRX(π, 0)
        prx(q, np.pi, 0)        # X
        prx(q, -np.pi/2, np.pi/2)  # Ry(-π/2)

    def cnot(control, target):
        """CNOT = H(t) @ CZ @ H(t)"""
        h(target)
        circ.cz(control, target)
        h(target)

    def z(q):
        """
        Z gate: need to decompose without Rz.
        Z = Rx(π) @ Ry(π) @ Rx(π) (up to global phase)
        Or: Z = Ry(π) @ Rx(π) (check: this is -iZ)

        Actually: Z = PRX(π, π/4) @ PRX(π, -π/4)
        Let me use: Z = Ry(π/2) @ X @ Ry(π/2)
        """
        # Z = e^{-iπZ/2} differs from Rz by global phase
        # Simpler: Z = Rx(π) @ Ry(π) = X @ Y
        prx(q, np.pi, np.pi/2)  # Y
        prx(q, np.pi, 0)        # X

    # ============================================
    # Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    # ============================================
    x(q_bob)              # X on Bob: |00⟩ → |01⟩
    h(q_alice)            # H on Alice: |01⟩ → (|0⟩+|1⟩)/√2 ⊗ |1⟩ = (|01⟩+|11⟩)/√2
    cnot(q_alice, q_bob)  # CNOT: → (|01⟩+|10⟩)/√2 = |Φ⁺⟩
    z(q_bob)              # Z on Bob: → (|01⟩-|10⟩)/√2 = |Ψ⁻⟩

    # ============================================
    # Apply measurement rotations
    # ============================================
    theta_a, phi_a = poincare_to_angles(a_vec)
    theta_b, phi_b = poincare_to_angles(b_vec)

    # Rotation to measurement basis: Rz(-phi) @ Ry(-theta)
    # Rz(-phi) @ Ry(-theta) = PRX(-theta, π/2 - phi)
    # because PRX(θ, φ) = Rz(φ) @ Rx(θ) @ Rz(-φ)
    # and we want Rz(-φ_a) @ Ry(-θ_a) = Rz(-φ_a) @ PRX(-θ_a, π/2)
    # = PRX(-θ_a, π/2 - φ_a)

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


def get_connected_pairs(iqm_device):
    """Get all connected qubit pairs from device topology."""
    topology = iqm_device.topology_graph
    edges = list(topology.edges())
    return edges


def test_qubit_pair(pair, emulator, phi_deg=30, num_shots=500):
    """Test a qubit pair by measuring C(a2, b2) and C(a2, b2') with φ = phi_deg."""
    phi_rad = np.radians(phi_deg)

    # Alice always on y-axis
    a2 = np.array([0, 1, 0])

    # Bob at ±φ/2 around y-axis in XY plane
    b2 = np.array([-np.sin(phi_rad/2), np.cos(phi_rad/2), 0])
    b2_prime = np.array([np.sin(phi_rad/2), np.cos(phi_rad/2), 0])

    # Theoretical correlations for singlet: C(a, b) = -a·b
    corr_b2_theory = -np.dot(a2, b2)
    corr_b2p_theory = -np.dot(a2, b2_prime)

    correlations = []
    errors = []

    for b_vec, label, theory in [(b2, 'b2', corr_b2_theory), (b2_prime, "b2'", corr_b2p_theory)]:
        corr = run_single_correlation(a2, b_vec, pair, emulator, num_shots)
        correlations.append(corr)
        errors.append(abs(corr - theory))

    avg_error = np.mean(errors)
    return avg_error, correlations, errors, {'theory_b2': corr_b2_theory, 'theory_b2p': corr_b2p_theory}


def main():
    import sys

    num_shots = 1000
    phi_deg = 30
    max_pairs = None

    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--phi' and i + 1 < len(sys.argv):
            phi_deg = float(sys.argv[i + 1])
        elif arg == '--max-pairs' and i + 1 < len(sys.argv):
            max_pairs = int(sys.argv[i + 1])

    print("="*70)
    print("SCANNING QUBIT PAIRS - IQM GARNET LocalEmulator")
    print("Using REAL calibration data from device")
    print("="*70)

    # Initialize device and emulator
    device_arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"

    print(f"\nFetching real-time calibration data from IQM Garnet...")
    iqm_device = AwsDevice(device_arn)
    emulator = iqm_device.emulator()
    print(f"  Device: {iqm_device.name}")
    print(f"  Status: {iqm_device.status}")

    # Get connected pairs from device topology
    connected_pairs = get_connected_pairs(iqm_device)
    print(f"\nFound {len(connected_pairs)} connected qubit pairs")

    if max_pairs:
        connected_pairs = connected_pairs[:max_pairs]
        print(f"  Testing first {max_pairs} pairs")

    print(f"\nConfiguration:")
    print(f"  Backend: IQM Garnet LocalEmulator (real calibration noise)")
    print(f"  Native gates: PRX, CZ, Rz")
    print(f"  Shots per measurement: {num_shots}")
    print(f"  φ = {phi_deg}°, φ/2 = {phi_deg/2}°")

    # Calculate theoretical correlations
    phi_rad = np.radians(phi_deg)
    a2 = np.array([0, 1, 0])
    b2 = np.array([-np.sin(phi_rad/2), np.cos(phi_rad/2), 0])
    b2_prime = np.array([np.sin(phi_rad/2), np.cos(phi_rad/2), 0])

    corr_b2_theory = -np.dot(a2, b2)
    corr_b2p_theory = -np.dot(a2, b2_prime)

    print(f"\nTheoretical C(a2, b2) = {corr_b2_theory:.4f}")
    print(f"Theoretical C(a2, b2') = {corr_b2p_theory:.4f}")
    print()

    results = []

    for i, pair in enumerate(connected_pairs):
        q1, q2 = pair
        print(f"Testing {i+1}/{len(connected_pairs)}: ({q1},{q2})... ", end='', flush=True)

        try:
            avg_error, correlations, errors, theory = test_qubit_pair(
                pair, emulator, phi_deg, num_shots
            )
            results.append({
                'qubits': [q1, q2],
                'avg_error': avg_error,
                'correlations': correlations,
                'errors': errors,
                'phi_deg': phi_deg,
                'theory': theory
            })
            print(f"avg_err={avg_error:.4f}, C(b2)={correlations[0]:+.4f}, C(b2')={correlations[1]:+.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\nNo results! All pairs failed.")
        return

    # Sort by error
    results.sort(key=lambda x: x['avg_error'])

    print("\n" + "="*70)
    print(f"TOP 20 BEST QUBIT PAIRS (by correlation accuracy)")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Avg Err':>10} {'C(a2,b2)':>11} {'C(a2,b2p)':>11}")
    print("-"*60)

    for i, r in enumerate(results[:20]):
        q1, q2 = r['qubits']
        print(f"{i+1:>4} ({q1:3d},{q2:3d})     {r['avg_error']:>10.4f} {r['correlations'][0]:>+11.4f} {r['correlations'][1]:>+11.4f}")

    if len(results) >= 10:
        print("\n" + "="*70)
        print("WORST 10 QUBIT PAIRS")
        print("="*70)
        print(f"{'Rank':>4} {'Qubits':>12} {'Avg Err':>10} {'C(a2,b2)':>11} {'C(a2,b2p)':>11}")
        print("-"*60)

        for i, r in enumerate(results[-10:]):
            q1, q2 = r['qubits']
            rank = len(results) - 10 + i + 1
            print(f"{rank:>4} ({q1:3d},{q2:3d})     {r['avg_error']:>10.4f} {r['correlations'][0]:>+11.4f} {r['correlations'][1]:>+11.4f}")

    output_file = f'qubit_pair_scan_iqm_garnet_EMULATOR_phi{int(phi_deg)}.json'

    with open(output_file, 'w') as f:
        json.dump({
            'device': 'IQM Garnet',
            'emulator': 'LocalEmulator with real calibration',
            'device_status': str(iqm_device.status),
            'phi_deg': phi_deg,
            'shots_per_measurement': num_shots,
            'theory_correlations': {
                'C_a2_b2': corr_b2_theory,
                'C_a2_b2p': corr_b2p_theory
            },
            'timestamp': datetime.now().isoformat(),
            'num_pairs_tested': len(results),
            'results': results
        }, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Saved to {output_file}")
    print(f"{'='*70}")

    print("\nRECOMMENDED LAYOUT FOR 12-QUBIT CIRCUIT (top 6 non-overlapping pairs):")
    layout = []
    used_qubits = set()
    selected_pairs = []

    for r in results:
        q1, q2 = r['qubits']
        if q1 not in used_qubits and q2 not in used_qubits:
            layout.extend([q1, q2])
            used_qubits.add(q1)
            used_qubits.add(q2)
            selected_pairs.append(r)
            if len(selected_pairs) >= 6:
                break

    print(f"  physical_pairs = {[(r['qubits'][0], r['qubits'][1]) for r in selected_pairs]}")

    print("\nBest non-overlapping pairs:")
    for i, r in enumerate(selected_pairs):
        q1, q2 = r['qubits']
        print(f"  Pair {i+1}: ({q1},{q2}) - avg error: {r['avg_error']:.4f}")
        print(f"           C(a2,b2)={r['correlations'][0]:+.4f}, C(a2,b2')={r['correlations'][1]:+.4f}")


if __name__ == "__main__":
    main()
