import os
#!/usr/bin/env python3
"""
Scan qubit pairs for Rigetti Ankaa-3 using AWS Braket LocalEmulator with REAL calibration data.

This version uses the actual device calibration from Ankaa-3:
- Real per-qubit gate errors from current calibration
- Real connectivity from device topology
- Proper transpilation to native gates (rx, rz, iswap)

Based on the working braket_leggett_test_rgti_emulator.py
"""

import numpy as np
import json
from datetime import datetime

from braket.aws import AwsDevice
from braket.circuits import Circuit

from qiskit import QuantumCircuit, transpile
from qiskit_braket_provider import AWSBraketBackend
from qiskit_braket_provider.providers.adapter import to_braket


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles (theta, phi)."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def create_qiskit_circuit_for_correlation(a_vec, b_vec):
    """
    Create a Qiskit circuit for ONE correlation measurement.
    Uses qubits 0 (Alice) and 1 (Bob).
    """
    qc = QuantumCircuit(2, 2)

    # Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    qc.x(1)      # X on Bob
    qc.h(0)      # H on Alice
    qc.cx(0, 1)  # CNOT
    qc.z(1)      # Z on Bob

    # Apply measurement rotations
    theta_a, phi_a = poincare_to_angles(a_vec)
    theta_b, phi_b = poincare_to_angles(b_vec)

    qc.rz(-phi_a, 0)
    qc.ry(-theta_a, 0)
    qc.rz(-phi_b, 1)
    qc.ry(-theta_b, 1)

    qc.measure([0, 1], [0, 1])

    return qc


def transpile_and_convert_to_braket(qc, ankaa3_device):
    """
    Transpile Qiskit circuit to Ankaa-3 native gates and convert to Braket.
    Uses qiskit-braket-provider for automatic transpilation.
    """
    backend = AWSBraketBackend(device=ankaa3_device)
    transpiled = transpile(qc, backend=backend, optimization_level=1)
    braket_circ = to_braket(transpiled)
    return braket_circ


def run_single_correlation(a_vec, b_vec, pair, emulator, ankaa3_device, num_shots):
    """
    Run a single correlation measurement using the LocalEmulator.
    """
    q_alice, q_bob = pair

    # Create Qiskit circuit on qubits 0, 1
    qc = create_qiskit_circuit_for_correlation(a_vec, b_vec)

    # Transpile to native gates and convert to Braket
    braket_circ = transpile_and_convert_to_braket(qc, ankaa3_device)

    # Remap qubits 0,1 to the physical pair
    remapped_circ = Circuit()
    qubit_map = {0: q_alice, 1: q_bob}

    for instr in braket_circ.instructions:
        gate = instr.operator
        gate_name = gate.name

        # Skip Measure and GPhase
        if gate_name in ('Measure', 'GPhase'):
            continue

        target_qubits = [qubit_map.get(q, q) for q in instr.target]

        # Ankaa-3 native gates: rx, rz, iswap
        if gate_name == 'Rz':
            remapped_circ.rz(target_qubits[0], gate.angle)
        elif gate_name == 'Rx':
            remapped_circ.rx(target_qubits[0], gate.angle)
        elif gate_name == 'Ry':
            # Ry(θ) = Rz(-π/2) Rx(θ) Rz(π/2)
            remapped_circ.rz(target_qubits[0], -np.pi/2)
            remapped_circ.rx(target_qubits[0], gate.angle)
            remapped_circ.rz(target_qubits[0], np.pi/2)
        elif gate_name == 'X':
            remapped_circ.rx(target_qubits[0], np.pi)
        elif gate_name == 'V':
            remapped_circ.rx(target_qubits[0], np.pi/2)
        elif gate_name == 'ISwap':
            remapped_circ.iswap(target_qubits[0], target_qubits[1])
        elif gate_name == 'CZ':
            remapped_circ.cz(target_qubits[0], target_qubits[1])
        else:
            raise ValueError(f"Unknown gate: {gate_name}")

    # Wrap in verbatim box for emulator
    final_circ = Circuit().add_verbatim_box(remapped_circ)

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


def get_connected_pairs(ankaa3_device):
    """Get all connected qubit pairs from device topology."""
    topology = ankaa3_device.topology_graph
    edges = list(topology.edges())
    return edges


def test_qubit_pair(pair, emulator, ankaa3_device, phi_deg=30, num_shots=500):
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
        corr = run_single_correlation(a2, b_vec, pair, emulator, ankaa3_device, num_shots)
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
    print("SCANNING QUBIT PAIRS - RIGETTI ANKAA-3 LocalEmulator")
    print("Using REAL calibration data from device")
    print("="*70)

    # Initialize device and emulator
    device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"

    print(f"\nFetching real-time calibration data from Ankaa-3...")
    ankaa3_device = AwsDevice(device_arn)
    emulator = ankaa3_device.emulator()
    print(f"  Device: {ankaa3_device.name}")
    print(f"  Status: {ankaa3_device.status}")

    # Get connected pairs from device topology
    connected_pairs = get_connected_pairs(ankaa3_device)
    print(f"\nFound {len(connected_pairs)} connected qubit pairs")

    if max_pairs:
        connected_pairs = connected_pairs[:max_pairs]
        print(f"  Testing first {max_pairs} pairs")

    print(f"\nConfiguration:")
    print(f"  Backend: Ankaa-3 LocalEmulator (real calibration noise)")
    print(f"  Transpilation: qiskit-braket-provider")
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
                pair, emulator, ankaa3_device, phi_deg, num_shots
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

    print("\n" + "="*70)
    print("WORST 10 QUBIT PAIRS")
    print("="*70)
    print(f"{'Rank':>4} {'Qubits':>12} {'Avg Err':>10} {'C(a2,b2)':>11} {'C(a2,b2p)':>11}")
    print("-"*60)

    for i, r in enumerate(results[-10:]):
        q1, q2 = r['qubits']
        rank = len(results) - 10 + i + 1
        print(f"{rank:>4} ({q1:3d},{q2:3d})     {r['avg_error']:>10.4f} {r['correlations'][0]:>+11.4f} {r['correlations'][1]:>+11.4f}")

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'qubit_pair_scan_rigetti_ankaa3_EMULATOR_phi{int(phi_deg)}.json')

    with open(output_file, 'w') as f:
        json.dump({
            'device': 'Rigetti Ankaa-3',
            'emulator': 'LocalEmulator with real calibration',
            'device_status': str(ankaa3_device.status),
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
