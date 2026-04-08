import os
#!/usr/bin/env python3
"""
Leggett Inequality Test with BBPSSW Entanglement Purification

This implements the proper BBPSSW protocol for entanglement distillation:
1. Create 2 noisy singlet pairs (4 qubits)
2. Apply bilateral CNOTs (Alice's control->target, Bob's control->target)
3. Measure target pair (ancilla)
4. Post-select on ancilla outcomes matching (both 0 or both 1)
5. The control pair now has higher fidelity - measure correlation on this pair

For 12-qubit simulation:
- Run 2 purification rounds per correlation (4 qubits each = 8 qubits)
- Plus direct measurement for comparison (4 qubits)
- Total: 12 qubits

Based on arXiv:0801.2241v2 and BBPSSW protocol
"""

import numpy as np
import json
from datetime import datetime

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles (theta, phi)."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def measure_polarization(qc, qubit, theta, phi_angle):
    """Measure polarization at angle specified by Poincaré sphere vector."""
    qc.rz(-phi_angle, qubit)
    qc.ry(-theta, qubit)


def create_singlet(qc, qubit_a, qubit_b):
    """Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 on qubits a and b."""
    qc.x(qubit_b)
    qc.h(qubit_a)
    qc.cx(qubit_a, qubit_b)
    qc.z(qubit_b)


def create_bbpssw_circuit(phi_rad, alice_idx):
    """
    Create circuit with BBPSSW purification for Leggett test.

    Layout (12 qubits):
    - Qubits 0-3: Purification for C(a_i, b_i)
      - 0,1: Control pair (will be measured for correlation)
      - 2,3: Target pair (ancilla, measured for post-selection)
    - Qubits 4-7: Purification for C(a_i, b_i')
      - 4,5: Control pair
      - 6,7: Target pair (ancilla)
    - Qubits 8-11: Direct measurement (no purification) for comparison
      - 8,9: C(a_i, b_i) direct
      - 10,11: C(a_i, b_i') direct

    Args:
        phi_rad: Angle φ in radians
        alice_idx: Which Alice direction (0=a1, 1=a2, 2=a3)

    Returns:
        QuantumCircuit
    """
    # 12 qubits: 4 for purified ab, 4 for purified ab', 4 for direct comparison
    qr = QuantumRegister(12, 'q')
    # Classical registers:
    # cr_purified: 4 bits (control pairs after purification)
    # cr_ancilla: 4 bits (target pairs for post-selection)
    # cr_direct: 4 bits (direct measurement)
    cr_purified = ClassicalRegister(4, 'purified')
    cr_ancilla = ClassicalRegister(4, 'ancilla')
    cr_direct = ClassicalRegister(4, 'direct')

    qc = QuantumCircuit(qr, cr_purified, cr_ancilla, cr_direct)

    # Alice's measurement directions
    alice_dirs = [
        np.array([1, 0, 0]),  # a1
        np.array([0, 1, 0]),  # a2
        np.array([0, 0, 1])   # a3
    ]
    a = alice_dirs[alice_idx]

    # Bob's measurement directions
    if alice_idx == 0:  # a1 = [1,0,0]
        b = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
        b_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    elif alice_idx == 1:  # a2 = [0,1,0]
        b = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
        b_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    else:  # a3 = [0,0,1]
        b = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
        b_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    # Get measurement angles
    theta_a, phi_a = poincare_to_angles(a)
    theta_b, phi_b = poincare_to_angles(b)
    theta_bp, phi_bp = poincare_to_angles(b_prime)

    # === PURIFICATION FOR C(a_i, b_i) - Qubits 0-3 ===
    # Create two singlet pairs
    create_singlet(qc, 0, 1)  # Control pair: Alice=0, Bob=1
    create_singlet(qc, 2, 3)  # Target pair: Alice=2, Bob=3

    # BBPSSW: Bilateral CNOT (Alice: 0->2, Bob: 1->3)
    qc.cx(0, 2)  # Alice's CNOT
    qc.cx(1, 3)  # Bob's CNOT

    # Measure target pair (ancilla) for post-selection
    qc.measure(2, cr_ancilla[0])
    qc.measure(3, cr_ancilla[1])

    # Apply measurement rotations to control pair and measure
    measure_polarization(qc, 0, theta_a, phi_a)
    measure_polarization(qc, 1, theta_b, phi_b)
    qc.measure(0, cr_purified[0])
    qc.measure(1, cr_purified[1])

    # === PURIFICATION FOR C(a_i, b_i') - Qubits 4-7 ===
    create_singlet(qc, 4, 5)  # Control pair
    create_singlet(qc, 6, 7)  # Target pair

    # BBPSSW: Bilateral CNOT
    qc.cx(4, 6)
    qc.cx(5, 7)

    # Measure ancilla
    qc.measure(6, cr_ancilla[2])
    qc.measure(7, cr_ancilla[3])

    # Measure control pair
    measure_polarization(qc, 4, theta_a, phi_a)
    measure_polarization(qc, 5, theta_bp, phi_bp)
    qc.measure(4, cr_purified[2])
    qc.measure(5, cr_purified[3])

    # === DIRECT MEASUREMENT (no purification) - Qubits 8-11 ===
    create_singlet(qc, 8, 9)
    measure_polarization(qc, 8, theta_a, phi_a)
    measure_polarization(qc, 9, theta_b, phi_b)
    qc.measure(8, cr_direct[0])
    qc.measure(9, cr_direct[1])

    create_singlet(qc, 10, 11)
    measure_polarization(qc, 10, theta_a, phi_a)
    measure_polarization(qc, 11, theta_bp, phi_bp)
    qc.measure(10, cr_direct[2])
    qc.measure(11, cr_direct[3])

    return qc


def extract_correlations_bbpssw(counts, num_shots):
    """
    Extract correlations from BBPSSW purification results.

    Post-selection criterion: ancilla qubits match (both 0 or both 1)

    Args:
        counts: Dictionary of bitstring counts
        num_shots: Total shots

    Returns:
        Tuple of:
        - (C_ab_purified, C_ab_prime_purified): Purified correlations
        - (n_ab, n_ab_prime): Number of retained shots
        - (C_ab_direct, C_ab_prime_direct): Direct (unpurified) correlations
    """
    # Purified correlations (post-selected on ancilla match)
    purified_ab = 0
    purified_ab_prime = 0
    n_purified_ab = 0
    n_purified_ab_prime = 0

    # Direct correlations (no post-selection)
    direct_ab = 0
    direct_ab_prime = 0

    for bitstring, count in counts.items():
        # Bitstring format: "direct ancilla purified" (space-separated)
        # Each register is MSB first within its section
        parts = bitstring.split()
        direct_bits = parts[0]    # 4 bits
        ancilla_bits = parts[1]   # 4 bits
        purified_bits = parts[2]  # 4 bits

        # === Purified C(a_i, b_i) ===
        # purified register: bit 0 = Alice ctrl (rightmost), bit 1 = Bob ctrl
        # ancilla register: bit 0 = Alice tgt, bit 1 = Bob tgt
        alice_ctrl_ab = int(purified_bits[3])  # bit 0 (rightmost)
        bob_ctrl_ab = int(purified_bits[2])    # bit 1
        alice_tgt_ab = int(ancilla_bits[3])    # bit 0
        bob_tgt_ab = int(ancilla_bits[2])      # bit 1

        # Post-select: ancilla match (both 0 or both 1)
        if alice_tgt_ab == bob_tgt_ab:
            corr = 1 if alice_ctrl_ab == bob_ctrl_ab else -1
            purified_ab += corr * count
            n_purified_ab += count

        # === Purified C(a_i, b_i') ===
        # purified register: bit 2 = Alice ctrl', bit 3 = Bob ctrl'
        # ancilla register: bit 2 = Alice tgt', bit 3 = Bob tgt'
        alice_ctrl_abp = int(purified_bits[1])  # bit 2
        bob_ctrl_abp = int(purified_bits[0])    # bit 3 (leftmost)
        alice_tgt_abp = int(ancilla_bits[1])    # bit 2
        bob_tgt_abp = int(ancilla_bits[0])      # bit 3

        if alice_tgt_abp == bob_tgt_abp:
            corr = 1 if alice_ctrl_abp == bob_ctrl_abp else -1
            purified_ab_prime += corr * count
            n_purified_ab_prime += count

        # === Direct correlations ===
        alice_direct_ab = int(direct_bits[3])   # bit 0
        bob_direct_ab = int(direct_bits[2])     # bit 1
        corr = 1 if alice_direct_ab == bob_direct_ab else -1
        direct_ab += corr * count

        alice_direct_abp = int(direct_bits[1])  # bit 2
        bob_direct_abp = int(direct_bits[0])    # bit 3
        corr = 1 if alice_direct_abp == bob_direct_abp else -1
        direct_ab_prime += corr * count

    # Normalize
    C_ab_purified = purified_ab / n_purified_ab if n_purified_ab > 0 else 0
    C_ab_prime_purified = purified_ab_prime / n_purified_ab_prime if n_purified_ab_prime > 0 else 0
    C_ab_direct = direct_ab / num_shots
    C_ab_prime_direct = direct_ab_prime / num_shots

    return (C_ab_purified, C_ab_prime_purified), \
           (n_purified_ab, n_purified_ab_prime), \
           (C_ab_direct, C_ab_prime_direct)


def calc_leggett_L3(correlations, phi_rad):
    """Calculate L3 from 6 correlation values."""
    C_a1b1, C_a1b1p, C_a2b2, C_a2b2p, C_a3b3, C_a3b3p = correlations

    L3 = (1/3) * (abs(C_a1b1 + C_a1b1p) +
                  abs(C_a2b2 + C_a2b2p) +
                  abs(C_a3b3 + C_a3b3p))

    L3_theory = 2 * abs(np.cos(phi_rad / 2))
    L_bound = 2 / (1 + abs(np.sin(phi_rad / 2)))

    return L3, L3_theory, L_bound


def run_bbpssw_test(phi_deg, num_shots, simulator, backend=None):
    """Run BBPSSW purification test for one angle."""
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running BBPSSW test for φ = {phi_deg}° ({num_shots} shots per job)")
    print(f"{'='*70}")

    purified_correlations = []
    direct_correlations = []
    n_purified_counts = []

    # Run 3 jobs for a1, a2, a3
    for alice_idx in range(3):
        alice_name = ['a₁=[1,0,0]', 'a₂=[0,1,0]', 'a₃=[0,0,1]'][alice_idx]
        print(f"\n  Job {alice_idx + 1}/3: Testing {alice_name}")

        qc = create_bbpssw_circuit(phi_rad, alice_idx)

        if backend is not None:
            qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
            result = simulator.run(qc_transpiled, shots=num_shots).result()
        else:
            result = simulator.run(qc, shots=num_shots).result()

        counts = result.get_counts()

        (C_ab_pur, C_abp_pur), (n_ab, n_abp), (C_ab_dir, C_abp_dir) = \
            extract_correlations_bbpssw(counts, num_shots)

        purified_correlations.extend([C_ab_pur, C_abp_pur])
        direct_correlations.extend([C_ab_dir, C_abp_dir])
        n_purified_counts.extend([n_ab, n_abp])

        retention_ab = 100 * n_ab / num_shots
        retention_abp = 100 * n_abp / num_shots

        print(f"    C({alice_name}, b): purified={C_ab_pur:.4f}, direct={C_ab_dir:.4f}, retention={retention_ab:.1f}%")
        print(f"    C({alice_name}, b'): purified={C_abp_pur:.4f}, direct={C_abp_dir:.4f}, retention={retention_abp:.1f}%")

    # Calculate L3
    L3_purified, L3_theory, L_bound = calc_leggett_L3(purified_correlations, phi_rad)
    L3_direct, _, _ = calc_leggett_L3(direct_correlations, phi_rad)

    avg_retention = np.mean(n_purified_counts) / num_shots * 100

    print(f"\n  SUMMARY:")
    print(f"    L₃ (purified):  {L3_purified:.4f}")
    print(f"    L₃ (direct):    {L3_direct:.4f}")
    print(f"    L₃ (theory):    {L3_theory:.4f}")
    print(f"    L₃ bound:       {L_bound:.4f}")
    print(f"    Avg retention:  {avg_retention:.1f}%")
    print(f"    Violation:      {'YES' if L3_purified > L_bound else 'No'}")

    return {
        'phi_deg': phi_deg,
        'phi_rad': phi_rad,
        'L3_purified': L3_purified,
        'L3_direct': L3_direct,
        'L3_theory': L3_theory,
        'bound': L_bound,
        'violated_purified': L3_purified > L_bound,
        'violated_direct': L3_direct > L_bound,
        'correlations_purified': purified_correlations,
        'correlations_direct': direct_correlations,
        'n_purified': n_purified_counts,
        'avg_retention_pct': avg_retention,
        'num_shots': num_shots,
        'timestamp': datetime.now().isoformat()
    }


def main():
    import sys

    use_noise_model = '--noise-model' in sys.argv
    backend_name = "ibm_pittsburgh"
    num_shots = 10000
    output_file = None

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    if output_file is None:
        if use_noise_model:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_BBPSSW_{backend_name}_NOISE.json')
        else:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'leggett_results_BBPSSW_noiseless.json')

    print("="*70)
    print("LEGGETT INEQUALITY TEST: BBPSSW Entanglement Purification")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Method: BBPSSW protocol (bilateral CNOT + ancilla post-selection)")
    print(f"  Circuit size: 12 qubits per job (3 jobs total)")
    print(f"  Shots per job: {num_shots}")
    if use_noise_model:
        print(f"  Noise model: {backend_name}")
    else:
        print(f"  Simulator: Noiseless")
    print(f"  Output: {output_file}")

    # Set up simulator ONCE
    backend = None
    if use_noise_model:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_aer.noise import NoiseModel

        print(f"\nLoading noise model from {backend_name}...")
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        noise_model = NoiseModel.from_backend(backend)
        simulator = AerSimulator(noise_model=noise_model)
        print("  Noise model loaded.")
    else:
        simulator = AerSimulator(method='matrix_product_state')

    test_angles = [15, 25, 30, 45, 60]
    results = []

    for phi_deg in test_angles:
        result = run_bbpssw_test(phi_deg, num_shots, simulator, backend)
        results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Angle':>8} {'L3_pur':>10} {'L3_dir':>10} {'L3_theory':>10} {'Bound':>10} {'Retain%':>10} {'Viol?':>8}")
    print("-"*80)

    for r in results:
        status = "YES" if r['violated_purified'] else "No"
        print(f"{r['phi_deg']:>8}° {r['L3_purified']:>10.4f} {r['L3_direct']:>10.4f} {r['L3_theory']:>10.4f} {r['bound']:>10.4f} {r['avg_retention_pct']:>10.1f} {status:>8}")

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Done!")


if __name__ == "__main__":
    main()
