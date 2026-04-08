import numpy as np
from qiskit import QuantumCircuit


def poincare_to_angles(vec):
    """
    Convert Poincaré sphere vector to spherical angles (theta, phi).

    Args:
        vec: 3D unit vector [x, y, z]

    Returns:
        theta: Polar angle (0 to π)
        phi: Azimuthal angle (0 to 2π)
    """
    x, y, z = vec

    # Polar angle from z-component
    theta = np.arccos(np.clip(z, -1, 1))

    # Azimuthal angle from x, y components
    phi = np.arctan2(y, x)

    return theta, phi


def measure_polarization(qc, qubit, theta, phi_angle):
    """
    Measure polarization at angle specified by Poincaré sphere vector.

    Args:
        qc: QuantumCircuit
        qubit: Which qubit to measure
        theta: Polar angle
        phi_angle: Azimuthal angle
    """
    # Rotate to measurement basis
    qc.rz(-phi_angle, qubit)
    qc.ry(-theta, qubit)


def create_leggett_circuit_for_angle(phi_rad):
    """
    Create a circuit with 12 independent singlet pairs, one for each of the
    6 correlations needed for BOTH +phi and -phi angles.

    Correlations for +phi: C(a₁,b₁), C(a₁,b₁'), C(a₂,b₂), C(a₂,b₂'), C(a₃,b₃), C(a₃,b₃')
    Correlations for -phi: C(a₁,b₁), C(a₁,b₁'), C(a₂,b₂), C(a₂,b₂'), C(a₃,b₃), C(a₃,b₃')

    Args:
        phi_rad: Angle φ in radians (will test both +phi and -phi)

    Returns:
        QuantumCircuit with 24 qubits (12 singlet pairs)
    """
    # 12 singlet pairs = 24 qubits (6 pairs for +phi, 6 pairs for -phi)
    qc = QuantumCircuit(24)

    # Alice's measurement directions (same for all)
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    # Bob's measurement directions for +phi
    b1_pos = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime_pos = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2_pos = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime_pos = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3_pos = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime_pos = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    # Bob's measurement directions for -phi
    b1_neg = np.array([np.cos(-phi_rad/2), np.sin(-phi_rad/2), 0])
    b1_prime_neg = np.array([np.cos(-phi_rad/2), -np.sin(-phi_rad/2), 0])
    b2_neg = np.array([0, np.cos(-phi_rad/2), np.sin(-phi_rad/2)])
    b2_prime_neg = np.array([0, np.cos(-phi_rad/2), -np.sin(-phi_rad/2)])
    b3_neg = np.array([np.sin(-phi_rad/2), 0, np.cos(-phi_rad/2)])
    b3_prime_neg = np.array([-np.sin(-phi_rad/2), 0, np.cos(-phi_rad/2)])

    # 12 correlations: 6 for +phi, then 6 for -phi
    measurement_pairs = [
        # Pairs 0-5 for +phi (qubits 0-11)
        (a1, b1_pos),       # Pair 0: qubits 0-1
        (a1, b1_prime_pos), # Pair 1: qubits 2-3
        (a2, b2_pos),       # Pair 2: qubits 4-5
        (a2, b2_prime_pos), # Pair 3: qubits 6-7
        (a3, b3_pos),       # Pair 4: qubits 8-9
        (a3, b3_prime_pos), # Pair 5: qubits 10-11
        # Pairs 6-11 for -phi (qubits 12-23)
        (a1, b1_neg),       # Pair 6: qubits 12-13
        (a1, b1_prime_neg), # Pair 7: qubits 14-15
        (a2, b2_neg),       # Pair 8: qubits 16-17
        (a2, b2_prime_neg), # Pair 9: qubits 18-19
        (a3, b3_neg),       # Pair 10: qubits 20-21
        (a3, b3_prime_neg)  # Pair 11: qubits 22-23
    ]

    # Create 12 independent singlet states and apply measurement rotations
    for i, (a_vec, b_vec) in enumerate(measurement_pairs):
        qubit_a = 2 * i      # Alice's qubit
        qubit_b = 2 * i + 1  # Bob's qubit

        # Create singlet state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        qc.x(qubit_b)
        qc.h(qubit_a)
        qc.cx(qubit_a, qubit_b)
        qc.z(qubit_b)

        # Apply measurement rotations
        theta_a, phi_a = poincare_to_angles(a_vec)
        theta_b, phi_b = poincare_to_angles(b_vec)

        measure_polarization(qc, qubit_a, theta_a, phi_a)
        measure_polarization(qc, qubit_b, theta_b, phi_b)

    # Measure all qubits
    qc.measure_all()

    return qc

def extract_correlation_from_counts(counts, num_shots):
    """
    Extract correlation from 2-qubit measurement counts.

    For singlet: C(a,b) = -a·b
    Convention: same bits → +1, different bits → -1
    """
    correlation = 0.0

    for bitstring, count in counts.items():
        alice_bit = int(bitstring[1])  # qubit 0
        bob_bit = int(bitstring[0])    # qubit 1

        # Same bits → +1, different bits → -1
        if alice_bit == bob_bit:
            correlation += count
        else:
            correlation -= count

    return correlation / num_shots

def extract_correlations_from_counts(counts, num_shots):
    """
    Extract 12 correlation values from measurement counts (6 for +phi, 6 for -phi).

    Args:
        counts: Dictionary of bitstring counts
        num_shots: Total number of shots

    Returns:
        Tuple of two lists:
          - correlations_pos: 6 correlation values for +phi
          - correlations_neg: 6 correlation values for -phi
    """
    correlations = [0.0] * 12

    for bitstring, count in counts.items():
        # bitstring format: q23 q22 ... q1 q0
        # Pairs 0-5 (+phi): (q0,q1), (q2,q3), (q4,q5), (q6,q7), (q8,q9), (q10,q11)
        # Pairs 6-11 (-phi): (q12,q13), (q14,q15), (q16,q17), (q18,q19), (q20,q21), (q22,q23)

        for pair_idx in range(12):
            alice_bit = int(bitstring[23 - 2*pair_idx])  # qubit 2*pair_idx
            bob_bit = int(bitstring[22 - 2*pair_idx])    # qubit 2*pair_idx + 1

            # Correlation for singlet state: same bits → +1, different bits → -1
            # This gives C(a,b) = -a·b as expected for singlet |Ψ⁻⟩
            if alice_bit == bob_bit:
                correlations[pair_idx] += count
            else:
                correlations[pair_idx] -= count

    # Normalize
    correlations = [c / num_shots for c in correlations]

    # Split into +phi (first 6) and -phi (last 6)
    correlations_pos = correlations[0:6]
    correlations_neg = correlations[6:12]

    return correlations_pos, correlations_neg


# Helper function to calculate L3 for one angle
def calc_leggett_for_angle(correlations, phi_rad):
    C_a1b1, C_a1b1p, C_a2b2, C_a2b2p, C_a3b3, C_a3b3p = correlations

    # Calculate theoretical correlations
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    C_a1b1_th = -np.dot(a1, b1)
    C_a1b1p_th = -np.dot(a1, b1_prime)
    C_a2b2_th = -np.dot(a2, b2)
    C_a2b2p_th = -np.dot(a2, b2_prime)
    C_a3b3_th = -np.dot(a3, b3)
    C_a3b3p_th = -np.dot(a3, b3_prime)

    # Leggett parameter L₃
    L3 = (1/3) * (abs(C_a1b1 + C_a1b1p) +
                  abs(C_a2b2 + C_a2b2p) +
                  abs(C_a3b3 + C_a3b3p))

    L3_th = (1/3) * (abs(C_a1b1_th + C_a1b1p_th) +
                     abs(C_a2b2_th + C_a2b2p_th) +
                     abs(C_a3b3_th + C_a3b3p_th))

    # Leggett bound
    L_bound = 2 - (2/3) * abs(np.sin(phi_rad/2))

    violated = L3 > L_bound

    return {
        'correlations': correlations,
        'correlations_theory': [C_a1b1_th, C_a1b1p_th, C_a2b2_th, C_a2b2p_th, C_a3b3_th, C_a3b3p_th],
        'L3': L3,
        'L3_theory': L3_th,
        'bound': L_bound,
        'violated': violated
    }
