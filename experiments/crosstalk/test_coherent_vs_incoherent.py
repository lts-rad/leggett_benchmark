#!/usr/bin/env python3
"""
Test whether Leggett inequality measurements can distinguish coherent vs incoherent errors.

Coherent errors: Systematic over/under-rotation of gates
Incoherent errors: Depolarizing noise

We simulate both and compare the resulting correlation patterns.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles (theta, phi)."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def create_leggett_circuit(phi_rad, rotation_error=0.0, rotation_error_axis='all'):
    """
    Create Leggett circuit with optional coherent rotation error.

    Args:
        phi_rad: Measurement angle
        rotation_error: Fractional over-rotation (e.g., 0.05 = 5% over-rotation)
        rotation_error_axis: 'all', 'ry', 'rz', or 'asymmetric'
    """
    qc = QuantumCircuit(12)  # 6 pairs

    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    measurement_pairs = [
        (a1, b1), (a1, b1_prime),
        (a2, b2), (a2, b2_prime),
        (a3, b3), (a3, b3_prime),
    ]

    for i, (a_vec, b_vec) in enumerate(measurement_pairs):
        qubit_a = 2 * i
        qubit_b = 2 * i + 1

        # Create singlet |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        qc.x(qubit_b)
        qc.h(qubit_a)
        qc.cx(qubit_a, qubit_b)
        qc.z(qubit_b)

        # Measurement rotations with optional coherent error
        theta_a, phi_a = poincare_to_angles(a_vec)
        theta_b, phi_b = poincare_to_angles(b_vec)

        # Apply rotation error based on axis setting
        if rotation_error_axis == 'all':
            ry_error = rotation_error
            rz_error = rotation_error
        elif rotation_error_axis == 'ry':
            ry_error = rotation_error
            rz_error = 0.0
        elif rotation_error_axis == 'rz':
            ry_error = 0.0
            rz_error = rotation_error
        elif rotation_error_axis == 'asymmetric':
            # Different error for different pairs (simulates qubit-dependent calibration)
            ry_error = rotation_error * (1 + 0.5 * np.sin(i))
            rz_error = rotation_error * (1 - 0.5 * np.sin(i))
        else:
            ry_error = rotation_error
            rz_error = rotation_error

        # Alice's rotation
        qc.rz(-phi_a * (1 + rz_error), qubit_a)
        qc.ry(-theta_a * (1 + ry_error), qubit_a)

        # Bob's rotation
        qc.rz(-phi_b * (1 + rz_error), qubit_b)
        qc.ry(-theta_b * (1 + ry_error), qubit_b)

    qc.measure_all()
    return qc


def extract_correlations(counts, num_shots):
    """Extract 6 correlations from measurement counts."""
    correlations = [0.0] * 6

    for bitstring, count in counts.items():
        for pair_idx in range(6):
            alice_bit = int(bitstring[11 - 2*pair_idx])
            bob_bit = int(bitstring[10 - 2*pair_idx])

            if alice_bit == bob_bit:
                correlations[pair_idx] += count
            else:
                correlations[pair_idx] -= count

    return [c / num_shots for c in correlations]


def theoretical_correlations(phi_rad):
    """Calculate theoretical correlations for singlet state."""
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    # C(a,b) = -a·b for singlet
    return [
        -np.dot(a1, b1), -np.dot(a1, b1_prime),
        -np.dot(a2, b2), -np.dot(a2, b2_prime),
        -np.dot(a3, b3), -np.dot(a3, b3_prime),
    ]


def calc_L3(correlations):
    """Calculate L3 parameter."""
    return (1/3) * (
        abs(correlations[0] + correlations[1]) +
        abs(correlations[2] + correlations[3]) +
        abs(correlations[4] + correlations[5])
    )


def run_simulation(phi_deg, rotation_error=0.0, rotation_error_axis='all',
                   depolarizing_rate=0.0, num_shots=10000):
    """
    Run simulation with specified error model.

    Args:
        phi_deg: Measurement angle in degrees
        rotation_error: Coherent over-rotation fraction
        depolarizing_rate: Incoherent depolarizing error rate per gate
        num_shots: Number of measurement shots
    """
    phi_rad = np.radians(phi_deg)
    qc = create_leggett_circuit(phi_rad, rotation_error, rotation_error_axis)

    sim = AerSimulator()

    if depolarizing_rate > 0:
        noise_model = NoiseModel()
        # Add depolarizing error to single-qubit gates
        error_1q = depolarizing_error(depolarizing_rate, 1)
        error_2q = depolarizing_error(depolarizing_rate * 2, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h', 'x', 'z'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        result = sim.run(qc, shots=num_shots, noise_model=noise_model).result()
    else:
        result = sim.run(qc, shots=num_shots).result()

    counts = result.get_counts()
    correlations = extract_correlations(counts, num_shots)
    theory = theoretical_correlations(phi_rad)

    return {
        'correlations': correlations,
        'theory': theory,
        'L3': calc_L3(correlations),
        'L3_theory': calc_L3(theory),
        'residuals': [correlations[i] - theory[i] for i in range(6)],
    }


def main():
    phi_deg = 30  # Test at 30 degrees
    num_shots = 10000

    print("=" * 70)
    print(f"Testing Coherent vs Incoherent Error Detection (φ = {phi_deg}°)")
    print("=" * 70)

    # 1. Ideal (no errors)
    print("\n1. IDEAL (no errors)")
    ideal = run_simulation(phi_deg, num_shots=num_shots)
    print(f"   L3 = {ideal['L3']:.4f} (theory: {ideal['L3_theory']:.4f})")
    print(f"   Correlations: {[f'{c:.4f}' for c in ideal['correlations']]}")
    print(f"   Residuals:    {[f'{r:+.4f}' for r in ideal['residuals']]}")

    # 2. Coherent error: 15% over-rotation on all axes
    print("\n2. COHERENT: 15% over-rotation (all axes)")
    coherent_all = run_simulation(phi_deg, rotation_error=0.15, num_shots=num_shots)
    print(f"   L3 = {coherent_all['L3']:.4f} (theory: {coherent_all['L3_theory']:.4f})")
    print(f"   Correlations: {[f'{c:.4f}' for c in coherent_all['correlations']]}")
    print(f"   Residuals:    {[f'{r:+.4f}' for r in coherent_all['residuals']]}")

    # 3. Coherent error: 15% over-rotation on RY only
    print("\n3. COHERENT: 15% over-rotation (RY only)")
    coherent_ry = run_simulation(phi_deg, rotation_error=0.15, rotation_error_axis='ry', num_shots=num_shots)
    print(f"   L3 = {coherent_ry['L3']:.4f} (theory: {coherent_ry['L3_theory']:.4f})")
    print(f"   Correlations: {[f'{c:.4f}' for c in coherent_ry['correlations']]}")
    print(f"   Residuals:    {[f'{r:+.4f}' for r in coherent_ry['residuals']]}")

    # 4. Coherent error: 15% over-rotation on RZ only
    print("\n4. COHERENT: 15% over-rotation (RZ only)")
    coherent_rz = run_simulation(phi_deg, rotation_error=0.15, rotation_error_axis='rz', num_shots=num_shots)
    print(f"   L3 = {coherent_rz['L3']:.4f} (theory: {coherent_rz['L3_theory']:.4f})")
    print(f"   Correlations: {[f'{c:.4f}' for c in coherent_rz['correlations']]}")
    print(f"   Residuals:    {[f'{r:+.4f}' for r in coherent_rz['residuals']]}")

    # 5. Coherent error: Asymmetric (different error per qubit pair)
    print("\n5. COHERENT: Asymmetric (qubit-dependent calibration error)")
    coherent_asym = run_simulation(phi_deg, rotation_error=0.15, rotation_error_axis='asymmetric', num_shots=num_shots)
    print(f"   L3 = {coherent_asym['L3']:.4f} (theory: {coherent_asym['L3_theory']:.4f})")
    print(f"   Correlations: {[f'{c:.4f}' for c in coherent_asym['correlations']]}")
    print(f"   Residuals:    {[f'{r:+.4f}' for r in coherent_asym['residuals']]}")

    # 6. Incoherent error: Depolarizing noise
    print("\n6. INCOHERENT: Depolarizing noise (1% per gate)")
    incoherent = run_simulation(phi_deg, depolarizing_rate=0.01, num_shots=num_shots)
    print(f"   L3 = {incoherent['L3']:.4f} (theory: {incoherent['L3_theory']:.4f})")
    print(f"   Correlations: {[f'{c:.4f}' for c in incoherent['correlations']]}")
    print(f"   Residuals:    {[f'{r:+.4f}' for r in incoherent['residuals']]}")

    # 7. Incoherent error: Higher depolarizing noise
    print("\n7. INCOHERENT: Depolarizing noise (3% per gate)")
    incoherent_high = run_simulation(phi_deg, depolarizing_rate=0.03, num_shots=num_shots)
    print(f"   L3 = {incoherent_high['L3']:.4f} (theory: {incoherent_high['L3_theory']:.4f})")
    print(f"   Correlations: {[f'{c:.4f}' for c in incoherent_high['correlations']]}")
    print(f"   Residuals:    {[f'{r:+.4f}' for r in incoherent_high['residuals']]}")

    # 8. Combined: Both coherent and incoherent
    print("\n8. COMBINED: 5% over-rotation + 1% depolarizing")
    combined = run_simulation(phi_deg, rotation_error=0.05, depolarizing_rate=0.01, num_shots=num_shots)
    print(f"   L3 = {combined['L3']:.4f} (theory: {combined['L3_theory']:.4f})")
    print(f"   Correlations: {[f'{c:.4f}' for c in combined['correlations']]}")
    print(f"   Residuals:    {[f'{r:+.4f}' for r in combined['residuals']]}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Distinguishing Signatures")
    print("=" * 70)

    def analyze_residuals(residuals, name):
        mean_r = np.mean(residuals)
        std_r = np.std(residuals)
        max_r = max(residuals)
        min_r = min(residuals)
        spread = max_r - min_r
        # Check if residuals have consistent sign (all positive or all negative)
        all_positive = all(r > 0 for r in residuals)
        all_negative = all(r < 0 for r in residuals)
        mixed = not (all_positive or all_negative)

        print(f"\n{name}:")
        print(f"   Mean residual: {mean_r:+.4f}")
        print(f"   Std residual:  {std_r:.4f}")
        print(f"   Range:         [{min_r:+.4f}, {max_r:+.4f}] (spread: {spread:.4f})")
        print(f"   Pattern:       {'MIXED (+/-)' if mixed else 'ALL SAME SIGN'}")

        return {'mean': mean_r, 'std': std_r, 'spread': spread, 'mixed': mixed}

    stats = {}
    stats['ideal'] = analyze_residuals(ideal['residuals'], "Ideal")
    stats['coherent_all'] = analyze_residuals(coherent_all['residuals'], "Coherent (all axes)")
    stats['coherent_ry'] = analyze_residuals(coherent_ry['residuals'], "Coherent (RY only)")
    stats['coherent_rz'] = analyze_residuals(coherent_rz['residuals'], "Coherent (RZ only)")
    stats['coherent_asym'] = analyze_residuals(coherent_asym['residuals'], "Coherent (asymmetric)")
    stats['incoherent'] = analyze_residuals(incoherent['residuals'], "Incoherent (1%)")
    stats['incoherent_high'] = analyze_residuals(incoherent_high['residuals'], "Incoherent (3%)")
    stats['combined'] = analyze_residuals(combined['residuals'], "Combined")

    print("\n" + "=" * 70)
    print("KEY DISTINGUISHING FEATURES:")
    print("=" * 70)
    print("""
    COHERENT errors tend to show:
    - Large spread in residuals (different correlations affected differently)
    - Mixed signs (some correlations above theory, some below)
    - Axis-dependent patterns

    INCOHERENT errors tend to show:
    - Small spread in residuals (all correlations reduced similarly)
    - Same sign residuals (all below theory for depolarizing)
    - Uniform reduction pattern
    """)

    # Print correlation table
    print("\n" + "=" * 70)
    print("CORRELATION TABLE")
    print("=" * 70)
    labels = ['C(a1,b1)', 'C(a1,b1\')', 'C(a2,b2)', 'C(a2,b2\')', 'C(a3,b3)', 'C(a3,b3\')']

    print(f"\n{'Error Type':<25} | " + " | ".join([f'{l:>9}' for l in labels]) + " | L3")
    print("-" * 100)
    print(f"{'Theory':<25} | " + " | ".join([f'{c:>9.4f}' for c in ideal['theory']]) + f" | {ideal['L3_theory']:.4f}")
    print("-" * 100)

    all_data = [
        ('Ideal', ideal),
        ('Coherent 15% (all)', coherent_all),
        ('Coherent 15% (RY)', coherent_ry),
        ('Coherent 15% (RZ)', coherent_rz),
        ('Coherent 15% (asym)', coherent_asym),
        ('Incoherent 1%', incoherent),
        ('Incoherent 3%', incoherent_high),
        ('Combined', combined),
    ]

    for name, data in all_data:
        print(f"{name:<25} | " + " | ".join([f'{c:>9.4f}' for c in data['correlations']]) + f" | {data['L3']:.4f}")

    # Print residuals table
    print("\n" + "=" * 70)
    print("RESIDUALS TABLE (measured - theory)")
    print("=" * 70)
    print(f"\n{'Error Type':<25} | " + " | ".join([f'{l:>9}' for l in labels]) + " | Mean | Std | Spread")
    print("-" * 120)

    for name, data in all_data:
        r = data['residuals']
        mean_r = np.mean(r)
        std_r = np.std(r)
        spread = max(r) - min(r)
        print(f"{name:<25} | " + " | ".join([f'{x:>+9.4f}' for x in r]) + f" | {mean_r:+.3f} | {std_r:.3f} | {spread:.4f}")


if __name__ == "__main__":
    main()
