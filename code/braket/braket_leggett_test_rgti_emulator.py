#!/usr/bin/env python3
"""
Leggett Inequality Test using AWS Braket LocalEmulator for Rigetti Ankaa-3

This version uses the Braket LocalEmulator which:
1. Uses real-time calibration data from Ankaa-3
2. Applies depolarizing noise based on actual device calibration
3. Uses qiskit-braket-provider for transpilation to native gates

Based on arXiv:0801.2241v2
"""

import numpy as np
import json
from datetime import datetime

from braket.aws import AwsDevice
from braket.circuits import Circuit

from qiskit import QuantumCircuit, transpile
from qiskit_braket_provider import AWSBraketBackend
from qiskit_braket_provider.providers.adapter import to_braket

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from leggett import calc_leggett_for_angle


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
    # Create backend for transpilation
    backend = AWSBraketBackend(device=ankaa3_device)

    # Transpile to native gates (iswap, rx, rz)
    transpiled = transpile(qc, backend=backend, optimization_level=1)

    # Convert to Braket circuit
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

    # Now remap qubits 0,1 to the physical pair
    # Build new circuit with correct qubit indices
    remapped_circ = Circuit()
    qubit_map = {0: q_alice, 1: q_bob}

    for instr in braket_circ.instructions:
        gate = instr.operator
        gate_name = gate.name

        # Skip Measure and GPhase - emulator handles measurement, global phase is irrelevant
        if gate_name in ('Measure', 'GPhase'):
            continue

        target_qubits = [qubit_map.get(q, q) for q in instr.target]

        # Apply the gate to remapped qubits using the gate name
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
            remapped_circ.rx(target_qubits[0], np.pi)  # X = Rx(π)
        elif gate_name == 'V':
            remapped_circ.rx(target_qubits[0], np.pi/2)  # V = sqrt(X) = Rx(π/2)
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


def run_single_angle(phi_deg, num_shots, emulator, ankaa3_device, physical_pairs):
    """
    Run Leggett test for a single angle φ using the LocalEmulator.
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running angle φ = +{phi_deg:.1f}° ({num_shots} shots)")
    print(f"{'='*70}")

    # Alice's measurement directions (fixed on Poincaré sphere)
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
            corr = run_single_correlation(a_vec, b_vec, pair, emulator, ankaa3_device, num_shots)
            correlations.append(corr)
            print(f"C = {corr:+.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            correlations.append(0.0)

    # Calculate Leggett result
    result = calc_leggett_for_angle(correlations, phi_rad)

    print(f"\n  Results for φ = +{abs(phi_deg):.1f}°:")
    print(f"    Correlations (exp): {[f'{c:.4f}' for c in result['correlations']]}")
    print(f"    Correlations (th):  {[f'{c:.4f}' for c in result['correlations_theory']]}")
    print(f"    L₃ (exp):      {result['L3']:.4f}")
    print(f"    L₃ (theory):   {result['L3_theory']:.4f}")
    print(f"    L₃ bound:      {result['bound']:.4f}")
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

    num_shots = 1000
    output_file = "leggett_results_rigetti_ankaa3_emulator.json"

    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    print("="*70)
    print("LEGGETT INEQUALITY TEST: Rigetti Ankaa-3 LocalEmulator")
    print("="*70)

    # Initialize device and emulator with real calibration data
    device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"

    print(f"\nFetching real-time calibration data from Ankaa-3...")
    ankaa3_device = AwsDevice(device_arn)
    emulator = ankaa3_device.emulator()
    print(f"  Device: {ankaa3_device.name}")
    print(f"  Status: {ankaa3_device.status}")

    # Physical qubit pairs - BEST 6 from LocalEmulator scan with real calibration
    # Selected based on lowest correlation error from qubit_pair_scan_rigetti_ankaa3_EMULATOR_phi30.json
    physical_pairs = [
        (51, 44), (9, 16), (78, 77), (7, 14), (36, 43), (21, 28)
    ]

    print(f"\nConfiguration:")
    print(f"  Backend: Ankaa-3 LocalEmulator (real calibration noise)")
    print(f"  Transpilation: qiskit-braket-provider")
    print(f"  Shots per angle: {num_shots}")
    print(f"  Physical pairs: {physical_pairs}")

    test_angles = [-60, -45, -30, -25, -15, 15, 25, 30, 45, 60]
    print(f"\nTest angles: {test_angles}")

    results = []

    for angle in test_angles:
        result = run_single_angle(angle, num_shots, emulator, ankaa3_device, physical_pairs)
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
    print(f"{'Angle':>8} {'L₃(exp)':>10} {'L₃(th)':>10} {'Bound':>10} {'Δ(exp-th)':>12} {'Status':>12}")
    print("-"*70)

    for r in results:
        status = "VIOLATION" if r['violated'] else "No violation"
        delta = r['L3'] - r['L3_theory']
        print(f"{r['phi_deg']:>+8.1f}° {r['L3']:>10.4f} {r['L3_theory']:>10.4f} {r['bound']:>10.4f} {delta:>+12.4f} {status:>12}")

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Done!")


if __name__ == "__main__":
    main()
