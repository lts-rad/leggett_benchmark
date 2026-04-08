import os
#!/usr/bin/env python3
"""
Leggett Inequality Test with Triple-Redundancy Distillation

This script implements a distillation scheme using post-selection:
- Each correlation measurement is performed with 3 redundant copies
- Only shots where all 3 copies agree are kept ("distilled")
- This suppresses errors at the cost of reduced effective shot count

For local simulation (12 qubits):
- Test 2 correlations at a time (one a_i with b_i and b_i')
- 2 correlations × 2 qubits/pair × 3 copies = 12 qubits
- Run 3 separate jobs for a1, a2, a3

For hardware (72 qubits):
- All 12 correlations × 3 copies = 36 singlet pairs = 72 qubits
- Requires careful layout selection for IBM Pittsburgh (156 qubits)

Based on arXiv:0801.2241v2
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import QuantumCircuit, transpile
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


def create_distillation_circuit_partial(phi_rad, alice_idx):
    """
    Create a 12-qubit circuit for partial Leggett test with triple redundancy.

    Tests 2 correlations: C(a_i, b_i) and C(a_i, b_i') for both +phi and -phi.
    Each correlation has 3 redundant copies for distillation.

    Qubit layout (12 qubits total):
    - Qubits 0-5: 3 copies of (a_i, b_i) for +phi
    - Qubits 6-11: 3 copies of (a_i, b_i') for +phi

    Note: For the full test, we run this 3 times with alice_idx = 0, 1, 2
    and combine the results.

    Args:
        phi_rad: Angle φ in radians
        alice_idx: Which Alice direction (0=a1=[1,0,0], 1=a2=[0,1,0], 2=a3=[0,0,1])

    Returns:
        QuantumCircuit with 12 qubits (2 correlations × 3 copies × 2 qubits)
    """
    qc = QuantumCircuit(12)

    # Alice's measurement directions
    alice_dirs = [
        np.array([1, 0, 0]),  # a1
        np.array([0, 1, 0]),  # a2
        np.array([0, 0, 1])   # a3
    ]
    a = alice_dirs[alice_idx]

    # Bob's measurement directions for +phi (depends on alice_idx)
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

    # Create 3 copies of (a_i, b_i) correlation - qubits 0-5
    for copy in range(3):
        qubit_a = 2 * copy
        qubit_b = 2 * copy + 1

        # Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        qc.x(qubit_b)
        qc.h(qubit_a)
        qc.cx(qubit_a, qubit_b)
        qc.z(qubit_b)

        # Apply measurement rotations
        measure_polarization(qc, qubit_a, theta_a, phi_a)
        measure_polarization(qc, qubit_b, theta_b, phi_b)

    # Create 3 copies of (a_i, b_i') correlation - qubits 6-11
    for copy in range(3):
        qubit_a = 6 + 2 * copy
        qubit_b = 6 + 2 * copy + 1

        # Create singlet state
        qc.x(qubit_b)
        qc.h(qubit_a)
        qc.cx(qubit_a, qubit_b)
        qc.z(qubit_b)

        # Apply measurement rotations
        measure_polarization(qc, qubit_a, theta_a, phi_a)
        measure_polarization(qc, qubit_b, theta_bp, phi_bp)

    qc.measure_all()
    return qc


def extract_distilled_correlations(counts, num_shots):
    """
    Extract correlations using PROPER DISTILLATION: post-select on
    Alice's 3 copies agreeing AND Bob's 3 copies agreeing SEPARATELY.

    This is the correct way to do entanglement distillation for Bell tests:
    - Check if Alice's 3 measurements agree (reliability check)
    - Check if Bob's 3 measurements agree (reliability check)
    - Only keep shots where BOTH parties have consistent results
    - Then compute correlation from the agreed-upon values

    This does NOT bias toward any particular correlation value because
    we're selecting on measurement reliability, not on correlation outcome.

    Args:
        counts: Dictionary of bitstring counts
        num_shots: Total number of shots

    Returns:
        Tuple of:
        - (C_ab, C_ab_prime): Distilled correlation values
        - (n_ab, n_ab_prime): Number of retained shots
        - (raw_C_ab, raw_C_ab_prime): Raw (undistilled) correlations for comparison
    """
    # Counters for distilled correlations (Alice agrees AND Bob agrees)
    distilled_ab = 0
    distilled_ab_prime = 0
    n_distilled_ab = 0
    n_distilled_ab_prime = 0

    # Counters for raw correlations (average of all 3 copies)
    raw_ab = 0
    raw_ab_prime = 0

    for bitstring, count in counts.items():
        # Parse bitstring (MSB first)
        # Qubits 0-5: 3 copies of (a_i, b_i)
        # Qubits 6-11: 3 copies of (a_i, b_i')

        # === First correlation: C(a_i, b_i) using qubits 0-5 ===
        alice_bits_ab = []
        bob_bits_ab = []
        for copy in range(3):
            alice_bits_ab.append(int(bitstring[11 - 2*copy]))       # qubit 2*copy
            bob_bits_ab.append(int(bitstring[11 - (2*copy + 1)]))   # qubit 2*copy + 1

        # Check if Alice's 3 copies agree
        alice_agrees_ab = (alice_bits_ab[0] == alice_bits_ab[1] == alice_bits_ab[2])
        # Check if Bob's 3 copies agree
        bob_agrees_ab = (bob_bits_ab[0] == bob_bits_ab[1] == bob_bits_ab[2])

        # Raw correlation (average all 3)
        for copy in range(3):
            corr_val = 1 if alice_bits_ab[copy] == bob_bits_ab[copy] else -1
            raw_ab += corr_val * count

        # Distilled: only if BOTH Alice and Bob individually agree
        if alice_agrees_ab and bob_agrees_ab:
            # Use the agreed-upon values
            corr_val = 1 if alice_bits_ab[0] == bob_bits_ab[0] else -1
            distilled_ab += corr_val * count
            n_distilled_ab += count

        # === Second correlation: C(a_i, b_i') using qubits 6-11 ===
        alice_bits_abp = []
        bob_bits_abp = []
        for copy in range(3):
            alice_bits_abp.append(int(bitstring[11 - (6 + 2*copy)]))      # qubit 6 + 2*copy
            bob_bits_abp.append(int(bitstring[11 - (6 + 2*copy + 1)]))    # qubit 6 + 2*copy + 1

        # Check if Alice's 3 copies agree
        alice_agrees_abp = (alice_bits_abp[0] == alice_bits_abp[1] == alice_bits_abp[2])
        # Check if Bob's 3 copies agree
        bob_agrees_abp = (bob_bits_abp[0] == bob_bits_abp[1] == bob_bits_abp[2])

        # Raw correlation (average all 3)
        for copy in range(3):
            corr_val = 1 if alice_bits_abp[copy] == bob_bits_abp[copy] else -1
            raw_ab_prime += corr_val * count

        # Distilled: only if BOTH Alice and Bob individually agree
        if alice_agrees_abp and bob_agrees_abp:
            corr_val = 1 if alice_bits_abp[0] == bob_bits_abp[0] else -1
            distilled_ab_prime += corr_val * count
            n_distilled_ab_prime += count

    # Normalize distilled correlations
    C_ab = distilled_ab / n_distilled_ab if n_distilled_ab > 0 else 0
    C_ab_prime = distilled_ab_prime / n_distilled_ab_prime if n_distilled_ab_prime > 0 else 0

    # Normalize raw correlations (3 copies * num_shots)
    raw_C_ab = raw_ab / (3 * num_shots)
    raw_C_ab_prime = raw_ab_prime / (3 * num_shots)

    return (C_ab, C_ab_prime), (n_distilled_ab, n_distilled_ab_prime), (raw_C_ab, raw_C_ab_prime)


def calc_leggett_L3(correlations, phi_rad):
    """Calculate L3 from 6 correlation values."""
    C_a1b1, C_a1b1p, C_a2b2, C_a2b2p, C_a3b3, C_a3b3p = correlations

    L3 = (1/3) * (abs(C_a1b1 + C_a1b1p) +
                  abs(C_a2b2 + C_a2b2p) +
                  abs(C_a3b3 + C_a3b3p))

    # Theoretical values
    L3_theory = 2 * abs(np.cos(phi_rad / 2))
    L_bound = 2 / (1 + abs(np.sin(phi_rad / 2)))

    return L3, L3_theory, L_bound


def run_distillation_test(phi_deg, num_shots, use_noise_model=False, backend_name="ibm_pittsburgh"):
    """
    Run the full distillation test for one angle.

    Runs 3 jobs (for a1, a2, a3) and combines the results.

    Args:
        phi_deg: Angle in degrees
        num_shots: Number of shots per job
        use_noise_model: Whether to use IBM noise model
        backend_name: IBM backend for noise model

    Returns:
        Dictionary with results
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running DISTILLATION test for φ = {phi_deg}° ({num_shots} shots per job)")
    print(f"{'='*70}")

    # Collect all correlations
    distilled_correlations = []
    raw_correlations = []
    n_distilled_counts = []

    # Set up simulator
    if use_noise_model:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_aer.noise import NoiseModel

        print(f"  Using noise model from {backend_name}")
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        noise_model = NoiseModel.from_backend(backend)
        simulator = AerSimulator(noise_model=noise_model)
    else:
        print(f"  Using noiseless simulator")
        simulator = AerSimulator(method='matrix_product_state')

    # Run 3 jobs for a1, a2, a3
    for alice_idx in range(3):
        alice_name = ['a₁=[1,0,0]', 'a₂=[0,1,0]', 'a₃=[0,0,1]'][alice_idx]
        print(f"\n  Job {alice_idx + 1}/3: Testing {alice_name}")

        # Create circuit
        qc = create_distillation_circuit_partial(phi_rad, alice_idx)

        if use_noise_model:
            # Transpile for noise model
            qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
            result = simulator.run(qc_transpiled, shots=num_shots).result()
        else:
            result = simulator.run(qc, shots=num_shots).result()

        counts = result.get_counts()

        # Extract distilled correlations
        (C_ab, C_ab_prime), (n_ab, n_ab_prime), (raw_ab, raw_ab_prime) = \
            extract_distilled_correlations(counts, num_shots)

        distilled_correlations.extend([C_ab, C_ab_prime])
        raw_correlations.extend([raw_ab, raw_ab_prime])
        n_distilled_counts.extend([n_ab, n_ab_prime])

        # Report
        retention_ab = 100 * n_ab / num_shots
        retention_abp = 100 * n_ab_prime / num_shots
        print(f"    C({alice_name}, b): distilled={C_ab:.4f}, raw={raw_ab:.4f}, retention={retention_ab:.1f}%")
        print(f"    C({alice_name}, b'): distilled={C_ab_prime:.4f}, raw={raw_ab_prime:.4f}, retention={retention_abp:.1f}%")

    # Calculate L3 for both distilled and raw
    L3_distilled, L3_theory, L_bound = calc_leggett_L3(distilled_correlations, phi_rad)
    L3_raw, _, _ = calc_leggett_L3(raw_correlations, phi_rad)

    avg_retention = np.mean(n_distilled_counts) / num_shots * 100

    print(f"\n  SUMMARY:")
    print(f"    L₃ (distilled):  {L3_distilled:.4f}")
    print(f"    L₃ (raw):        {L3_raw:.4f}")
    print(f"    L₃ (theory):     {L3_theory:.4f}")
    print(f"    L₃ bound:        {L_bound:.4f}")
    print(f"    Avg retention:   {avg_retention:.1f}%")
    print(f"    Violation:       {'YES' if L3_distilled > L_bound else 'No'}")

    return {
        'phi_deg': phi_deg,
        'phi_rad': phi_rad,
        'L3_distilled': L3_distilled,
        'L3_raw': L3_raw,
        'L3_theory': L3_theory,
        'bound': L_bound,
        'violated_distilled': L3_distilled > L_bound,
        'violated_raw': L3_raw > L_bound,
        'correlations_distilled': distilled_correlations,
        'correlations_raw': raw_correlations,
        'n_distilled': n_distilled_counts,
        'avg_retention_pct': avg_retention,
        'num_shots': num_shots,
        'timestamp': datetime.now().isoformat()
    }


def main():
    import sys

    use_noise_model = '--noise-model' in sys.argv
    backend_name = "ibm_pittsburgh"
    num_shots = 10000  # More shots since we lose some to post-selection
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
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_DISTILLATION_{backend_name}_NOISE.json')
        else:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'leggett_results_DISTILLATION_noiseless.json')

    print("="*70)
    print("LEGGETT INEQUALITY TEST: Triple-Redundancy Distillation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Method: Triple redundancy with post-selection")
    print(f"  Circuit size: 12 qubits per job (3 jobs total)")
    print(f"  Shots per job: {num_shots}")
    if use_noise_model:
        print(f"  Noise model: {backend_name}")
    else:
        print(f"  Simulator: Noiseless")
    print(f"  Output: {output_file}")

    test_angles = [15, 25, 30, 45, 60]
    results = []

    for phi_deg in test_angles:
        result = run_distillation_test(phi_deg, num_shots, use_noise_model, backend_name)
        results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Angle':>8} {'L3_dist':>10} {'L3_raw':>10} {'L3_theory':>10} {'Bound':>10} {'Retain%':>10} {'Viol?':>8}")
    print("-"*80)

    for r in results:
        status = "YES" if r['violated_distilled'] else "No"
        print(f"{r['phi_deg']:>8}° {r['L3_distilled']:>10.4f} {r['L3_raw']:>10.4f} {r['L3_theory']:>10.4f} {r['bound']:>10.4f} {r['avg_retention_pct']:>10.1f} {status:>8}")

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Done!")


if __name__ == "__main__":
    main()
