#!/usr/bin/env python3
"""
Rigorous test to determine the correct rotation sequence and signs.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def poincare_to_angles(vec):
    """Convert Poincaré sphere vector to spherical angles."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def test_single_correlation(a_vec, b_vec, ry_rz_order, use_negative_signs, shots=100000):
    """
    Test a single correlation measurement.

    Args:
        a_vec, b_vec: Measurement directions
        ry_rz_order: "ry_rz" or "rz_ry"
        use_negative_signs: True to use negative signs in rotations
    """
    qc = QuantumCircuit(2)

    # Create singlet state |ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    qc.x(1)  # Start with |01⟩
    qc.h(0)
    qc.cx(0, 1)
    qc.z(1)

    # Get angles
    theta_a, phi_a = poincare_to_angles(a_vec)
    theta_b, phi_b = poincare_to_angles(b_vec)

    # Apply rotations with specified order and signs
    sign = -1 if use_negative_signs else 1

    if ry_rz_order == "ry_rz":
        qc.ry(sign * theta_a, 0)
        qc.rz(sign * phi_a, 0)
        qc.ry(sign * theta_b, 1)
        qc.rz(sign * phi_b, 1)
    else:  # "rz_ry"
        qc.rz(sign * phi_a, 0)
        qc.ry(sign * theta_a, 0)
        qc.rz(sign * phi_b, 1)
        qc.ry(sign * theta_b, 1)

    qc.measure_all()

    # Simulate
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Calculate correlation
    correlation = 0.0
    for bitstring, count in counts.items():
        alice_bit = int(bitstring[1])
        bob_bit = int(bitstring[0])

        # For singlet: same bits → +1, different bits → -1
        # This gives C = -a·b
        if alice_bit == bob_bit:
            correlation += count
        else:
            correlation -= count

    return correlation / shots


if __name__ == "__main__":
    print("="*80)
    print("TESTING ALL COMBINATIONS OF ROTATION ORDER AND SIGNS")
    print("="*80)

    # Test with φ = 30°
    phi_deg = 30
    phi = np.radians(phi_deg)

    # Alice and Bob measurement directions
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])

    b1 = np.array([np.cos(phi/2), np.sin(phi/2), 0])
    b1_prime = np.array([np.cos(phi/2), -np.sin(phi/2), 0])
    b2 = np.array([0, np.cos(phi/2), np.sin(phi/2)])
    b2_prime = np.array([0, np.cos(phi/2), -np.sin(phi/2)])

    # Theoretical correlations for singlet: C = -a·b
    theory = {
        'C(a1,b1)': -np.dot(a1, b1),
        'C(a1,b1\')': -np.dot(a1, b1_prime),
        'C(a2,b2)': -np.dot(a2, b2),
        'C(a2,b2\')': -np.dot(a2, b2_prime),
    }

    print(f"\nTesting with φ = {phi_deg}°")
    print(f"\nTheoretical correlations (singlet C = -a·b):")
    for name, val in theory.items():
        print(f"  {name:12} = {val:+.6f}")

    # Test all 4 combinations
    combinations = [
        ("RY then RZ", "ry_rz", True,  "RY(-θ)RZ(-φ)"),
        ("RY then RZ", "ry_rz", False, "RY(+θ)RZ(+φ)"),
        ("RZ then RY", "rz_ry", True,  "RZ(-φ)RY(-θ)"),
        ("RZ then RY", "rz_ry", False, "RZ(+φ)RY(+θ)"),
    ]

    print("\n" + "="*80)
    print(f"{'Order':12} {'Formula':15} {'C(a1,b1)':>12} {'C(a1,b1\')':>12} {'C(a2,b2)':>12} {'C(a2,b2\')':>12} {'Total Err':>12}")
    print("-"*80)

    best_error = float('inf')
    best_combo = None

    for name, order, neg_signs, formula in combinations:
        c1 = test_single_correlation(a1, b1, order, neg_signs)
        c2 = test_single_correlation(a1, b1_prime, order, neg_signs)
        c3 = test_single_correlation(a2, b2, order, neg_signs)
        c4 = test_single_correlation(a2, b2_prime, order, neg_signs)

        error = (abs(c1 - theory['C(a1,b1)']) +
                 abs(c2 - theory['C(a1,b1\')']) +
                 abs(c3 - theory['C(a2,b2)']) +
                 abs(c4 - theory['C(a2,b2\')']))

        print(f"{name:12} {formula:15} {c1:>+12.6f} {c2:>+12.6f} {c3:>+12.6f} {c4:>+12.6f} {error:>12.6f}")

        if error < best_error:
            best_error = error
            best_combo = (name, formula)

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"\n✓ CORRECT: {best_combo[0]} with formula {best_combo[1]}")
    print(f"  Total error: {best_error:.6f}")

    # Detailed explanation
    print("\n" + "="*80)
    print("EXPLANATION")
    print("="*80)
    print("""
The correct sequence is the one that minimizes error vs theoretical predictions.

For measuring along direction n̂ = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)):

The state along n̂ is created by: |ψ(θ,φ)⟩ = RZ(φ)RY(θ)|0⟩

To MEASURE in that basis, we need to rotate that direction BACK to the Z-axis,
which means applying the INVERSE rotation: RY⁻¹(θ)RZ⁻¹(φ) = RY(-θ)RZ(-φ)

But we apply these in REVERSE order of the original sequence:
  Original: RZ(φ)RY(θ)
  Inverse:  RY(-θ)RZ(-φ)  [apply RY first to undo the last operation]

Wait, that's RY then RZ... Let me recalculate which actually works.
""")
