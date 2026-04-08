"""
Two-qubit state tomography module.

Performs 3x3 Pauli basis measurements (XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ)
to reconstruct the density matrix and calculate tangle/concurrence.

Based on James et al., Phys. Rev. A 64, 052312 (2001) for tangle calculation.
"""

import numpy as np
from qiskit import QuantumCircuit


# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def create_singlet_circuit():
    """
    Create a circuit that prepares the singlet state |Psi^-⟩ = (|01⟩ - |10⟩)/sqrt(2)
    """
    qc = QuantumCircuit(2)
    qc.x(1)       # |00⟩ -> |01⟩
    qc.h(0)       # -> (|01⟩ + |11⟩)/sqrt(2)
    qc.cx(0, 1)   # -> (|01⟩ + |10⟩)/sqrt(2)
    qc.z(0)       # -> (|01⟩ - |10⟩)/sqrt(2) = |Psi^-⟩
    return qc


def apply_measurement_rotation(qc, qubit, basis):
    """
    Apply rotation to measure in specified Pauli basis.

    Args:
        qc: QuantumCircuit
        qubit: Which qubit to rotate
        basis: 'X', 'Y', or 'Z'
    """
    if basis == 'X':
        qc.h(qubit)  # Rotate X basis to Z basis
    elif basis == 'Y':
        qc.sdg(qubit)  # S^dagger
        qc.h(qubit)    # Rotate Y basis to Z basis
    # Z basis: no rotation needed


def create_tomography_circuit(basis_a, basis_b):
    """
    Create a circuit for tomography measurement in given bases.

    Args:
        basis_a: Measurement basis for qubit A ('X', 'Y', or 'Z')
        basis_b: Measurement basis for qubit B ('X', 'Y', or 'Z')

    Returns:
        QuantumCircuit measuring in the specified bases
    """
    qc = create_singlet_circuit()

    # Apply measurement rotations
    apply_measurement_rotation(qc, 0, basis_a)
    apply_measurement_rotation(qc, 1, basis_b)

    # Measure
    qc.measure_all()

    return qc


def create_tomography_circuit_9pairs():
    """
    Create a circuit with 9 independent singlet pairs for all tomography measurements.

    Measures all 9 Pauli basis combinations simultaneously:
    XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ

    Returns:
        QuantumCircuit with 18 qubits (9 singlet pairs)
    """
    qc = QuantumCircuit(18)

    # 9 basis combinations
    bases = [
        ('X', 'X'),  # Pair 0: qubits 0-1
        ('X', 'Y'),  # Pair 1: qubits 2-3
        ('X', 'Z'),  # Pair 2: qubits 4-5
        ('Y', 'X'),  # Pair 3: qubits 6-7
        ('Y', 'Y'),  # Pair 4: qubits 8-9
        ('Y', 'Z'),  # Pair 5: qubits 10-11
        ('Z', 'X'),  # Pair 6: qubits 12-13
        ('Z', 'Y'),  # Pair 7: qubits 14-15
        ('Z', 'Z'),  # Pair 8: qubits 16-17
    ]

    for i, (basis_a, basis_b) in enumerate(bases):
        qubit_a = 2 * i
        qubit_b = 2 * i + 1

        # Create singlet state |Psi^-⟩ = (|01⟩ - |10⟩)/sqrt(2)
        qc.x(qubit_b)
        qc.h(qubit_a)
        qc.cx(qubit_a, qubit_b)
        qc.z(qubit_a)

        # Apply measurement rotations
        apply_measurement_rotation(qc, qubit_a, basis_a)
        apply_measurement_rotation(qc, qubit_b, basis_b)

    qc.measure_all()
    return qc


def extract_expectation_values(counts, num_shots):
    """
    Extract all 9 Pauli expectation values from 18-qubit tomography circuit.

    Args:
        counts: Dictionary of bitstring counts from 18-qubit measurement
        num_shots: Total number of shots

    Returns:
        Dictionary mapping basis pairs to expectation values
    """
    bases = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    expectations = {b: 0.0 for b in bases}

    for bitstring, count in counts.items():
        for pair_idx, basis in enumerate(bases):
            # Extract bits for this pair (bitstring is MSB first)
            # Pair i uses qubits 2i and 2i+1
            alice_bit = int(bitstring[17 - 2*pair_idx])
            bob_bit = int(bitstring[16 - 2*pair_idx])

            # Expectation value: (+1 if same parity, -1 if different)
            # For Pauli measurements: eigenvalue is (-1)^bit
            alice_val = (-1) ** alice_bit
            bob_val = (-1) ** bob_bit

            expectations[basis] += alice_val * bob_val * count

    # Normalize
    for basis in bases:
        expectations[basis] /= num_shots

    return expectations


def extract_single_qubit_expectations(counts, num_shots):
    """
    Extract single-qubit expectation values from tomography data.

    Uses ZZ measurement to get <ZI> and <IZ>
    Uses XZ measurement to get <XI>
    Uses ZX measurement to get <IX>
    Uses YZ measurement to get <YI>
    Uses ZY measurement to get <IY>

    Returns:
        Dictionary with 'XI', 'YI', 'ZI', 'IX', 'IY', 'IZ' expectations
    """
    single_exp = {
        'XI': 0.0, 'YI': 0.0, 'ZI': 0.0,
        'IX': 0.0, 'IY': 0.0, 'IZ': 0.0
    }

    # Map which measurement gives which single-qubit expectation
    # XZ measurement (pair 2, qubits 4-5): qubit A in X basis, qubit B in Z basis
    # ZX measurement (pair 6, qubits 12-13): qubit A in Z basis, qubit B in X basis

    for bitstring, count in counts.items():
        # XZ (pair 2): get <XI> from qubit A
        alice_bit_xz = int(bitstring[17 - 4])  # qubit 4
        single_exp['XI'] += ((-1) ** alice_bit_xz) * count

        # YZ (pair 5): get <YI> from qubit A
        alice_bit_yz = int(bitstring[17 - 10])  # qubit 10
        single_exp['YI'] += ((-1) ** alice_bit_yz) * count

        # ZZ (pair 8): get <ZI> and <IZ> from both qubits
        alice_bit_zz = int(bitstring[17 - 16])  # qubit 16
        bob_bit_zz = int(bitstring[17 - 17])    # qubit 17
        single_exp['ZI'] += ((-1) ** alice_bit_zz) * count
        single_exp['IZ'] += ((-1) ** bob_bit_zz) * count

        # ZX (pair 6): get <IX> from qubit B
        bob_bit_zx = int(bitstring[17 - 13])  # qubit 13
        single_exp['IX'] += ((-1) ** bob_bit_zx) * count

        # ZY (pair 7): get <IY> from qubit B
        bob_bit_zy = int(bitstring[17 - 15])  # qubit 15
        single_exp['IY'] += ((-1) ** bob_bit_zy) * count

    # Normalize
    for key in single_exp:
        single_exp[key] /= num_shots

    return single_exp


def reconstruct_density_matrix(two_qubit_exp, single_exp=None):
    """
    Reconstruct 2-qubit density matrix from Pauli expectation values.

    The density matrix is:
        rho = (1/4) sum_{i,j} r_ij (sigma_i tensor sigma_j)

    where r_ij = <sigma_i tensor sigma_j> and i,j in {I, X, Y, Z}

    Args:
        two_qubit_exp: Dictionary with 'XX', 'XY', etc. expectations
        single_exp: Optional dictionary with 'XI', 'IX', etc. expectations
                   If None, assumes maximally mixed marginals (singlet case)

    Returns:
        4x4 complex numpy array representing the density matrix
    """
    # Build full expectation dictionary
    exp = {'II': 1.0}  # Normalization

    # Two-qubit correlations
    for basis in ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']:
        exp[basis] = two_qubit_exp.get(basis, 0.0)

    # Single-qubit expectations
    if single_exp is not None:
        for key in ['XI', 'YI', 'ZI', 'IX', 'IY', 'IZ']:
            exp[key] = single_exp.get(key, 0.0)
    else:
        # For singlet state, marginals are maximally mixed
        for key in ['XI', 'YI', 'ZI', 'IX', 'IY', 'IZ']:
            exp[key] = 0.0

    # Reconstruct density matrix
    rho = np.zeros((4, 4), dtype=complex)

    paulis_list = ['I', 'X', 'Y', 'Z']

    for i, p_a in enumerate(paulis_list):
        for j, p_b in enumerate(paulis_list):
            key = p_a + p_b
            r_ij = exp.get(key, 0.0)

            # Tensor product of Pauli matrices
            sigma_ij = np.kron(PAULIS[p_a], PAULIS[p_b])

            rho += r_ij * sigma_ij

    rho /= 4.0

    return rho


def calculate_tangle(rho):
    """
    Calculate the tangle of a 2-qubit density matrix.

    Following James et al., Phys. Rev. A 64, 052312 (2001):
        R = rho * Sigma * rho^T * Sigma
        where Sigma is the spin-flip matrix

    Concurrence: C = max(0, sqrt(r1) - sqrt(r2) - sqrt(r3) - sqrt(r4))
    Tangle: T = C^2

    Args:
        rho: 4x4 density matrix (numpy array)

    Returns:
        tuple: (tangle, concurrence)
    """
    rho_array = np.array(rho)

    # Spin-flip matrix Sigma = sigma_y tensor sigma_y
    Sigma = np.array([
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0]
    ], dtype=complex)

    # R = rho * Sigma * rho^T * Sigma
    rho_T = rho_array.T  # Transpose (not conjugate transpose)
    R = rho_array @ Sigma @ rho_T @ Sigma

    # Get eigenvalues of R
    eigenvalues = np.linalg.eigvals(R)

    # Take real parts (should be real for valid density matrices)
    eigenvalues = np.real(eigenvalues)

    # Ensure non-negative
    eigenvalues = np.maximum(eigenvalues, 0)

    # Sort in decreasing order
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Concurrence: C = max(0, sqrt(r1) - sqrt(r2) - sqrt(r3) - sqrt(r4))
    sqrt_eigs = np.sqrt(eigenvalues)
    concurrence = max(0, sqrt_eigs[0] - sqrt_eigs[1] - sqrt_eigs[2] - sqrt_eigs[3])

    # Tangle: T = C^2
    tangle = concurrence ** 2

    return tangle, concurrence


def calculate_purity(rho):
    """
    Calculate purity of density matrix: Tr(rho^2)

    Pure state: purity = 1
    Maximally mixed 2-qubit: purity = 0.25
    """
    return np.real(np.trace(rho @ rho))


def calculate_fidelity_singlet(rho):
    """
    Calculate fidelity with ideal singlet state.

    F = <Psi^-|rho|Psi^->
    """
    # Singlet state vector: (|01⟩ - |10⟩)/sqrt(2)
    # In computational basis: [0, 1, -1, 0]/sqrt(2)
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)

    fidelity = np.real(singlet.conj() @ rho @ singlet)
    return fidelity


def get_theoretical_singlet_expectations():
    """
    Return theoretical expectation values for ideal singlet state.

    Singlet |Psi^-⟩ = (|01⟩ - |10⟩)/sqrt(2)
    """
    two_qubit = {
        'XX': -1.0,
        'XY': 0.0,
        'XZ': 0.0,
        'YX': 0.0,
        'YY': -1.0,
        'YZ': 0.0,
        'ZX': 0.0,
        'ZY': 0.0,
        'ZZ': -1.0
    }

    single = {
        'XI': 0.0, 'YI': 0.0, 'ZI': 0.0,
        'IX': 0.0, 'IY': 0.0, 'IZ': 0.0
    }

    return two_qubit, single


def print_density_matrix(rho, title="Density Matrix"):
    """Pretty print a density matrix."""
    print(f"\n{title}:")
    print("-" * 50)
    for i in range(4):
        row = "  ".join(f"{rho[i,j].real:+.4f}{rho[i,j].imag:+.4f}j" for j in range(4))
        print(f"  [{row}]")


def analyze_tomography_results(two_qubit_exp, single_exp=None, verbose=True):
    """
    Complete analysis of tomography results.

    Args:
        two_qubit_exp: Dictionary of two-qubit Pauli expectations
        single_exp: Optional dictionary of single-qubit expectations
        verbose: Print detailed results

    Returns:
        Dictionary with analysis results
    """
    # Reconstruct density matrix
    rho = reconstruct_density_matrix(two_qubit_exp, single_exp)

    # Calculate quantities
    tangle, concurrence = calculate_tangle(rho)
    purity = calculate_purity(rho)
    fidelity = calculate_fidelity_singlet(rho)

    # Get theoretical values for comparison
    th_two, th_single = get_theoretical_singlet_expectations()
    rho_ideal = reconstruct_density_matrix(th_two, th_single)
    tangle_ideal, concurrence_ideal = calculate_tangle(rho_ideal)

    if verbose:
        print("\n" + "=" * 60)
        print("TOMOGRAPHY ANALYSIS RESULTS")
        print("=" * 60)

        print("\nMeasured Pauli Expectations:")
        print("-" * 40)
        for basis in ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']:
            exp_val = two_qubit_exp.get(basis, 0.0)
            th_val = th_two.get(basis, 0.0)
            print(f"  <{basis}> = {exp_val:+.4f}  (theory: {th_val:+.4f}, delta: {exp_val - th_val:+.4f})")

        if single_exp:
            print("\nSingle-qubit expectations:")
            for key in ['XI', 'YI', 'ZI', 'IX', 'IY', 'IZ']:
                val = single_exp.get(key, 0.0)
                print(f"  <{key}> = {val:+.4f}")

        print_density_matrix(rho, "Reconstructed Density Matrix")
        print_density_matrix(rho_ideal, "Ideal Singlet Density Matrix")

        print("\n" + "-" * 40)
        print("ENTANGLEMENT MEASURES")
        print("-" * 40)
        print(f"  Concurrence:    {concurrence:.6f}  (ideal: {concurrence_ideal:.6f})")
        print(f"  Tangle:         {tangle:.6f}  (ideal: {tangle_ideal:.6f})")
        print(f"  Purity:         {purity:.6f}  (ideal: 1.000000)")
        print(f"  Fidelity:       {fidelity:.6f}  (with singlet)")

        # Visibility estimate from concurrence
        # For Werner state: C = max(0, (3V-1)/2), so V = (2C+1)/3
        visibility_est = (2 * concurrence + 1) / 3
        print(f"\n  Estimated visibility (from concurrence): {visibility_est:.4f}")

        # Key thresholds from Branciard paper
        V_ENTANGLED = 1/3
        V_CHSH = 1/np.sqrt(2)
        V_LEGGETT_COMPAT = (1 + 1/np.sqrt(2)) / 2  # ~0.854: Leggett model can reproduce
        V_LEGGETT_VIOLATE = np.sqrt(3) / 2         # ~0.866: Solidly violates Leggett

        print("\n  Visibility thresholds:")
        print(f"    V > {V_ENTANGLED:.4f}: Entangled  {'[YES]' if visibility_est > V_ENTANGLED else '[NO]'}")
        print(f"    V > {V_CHSH:.4f}: CHSH violation  {'[YES]' if visibility_est > V_CHSH else '[NO]'}")
        print(f"    V <= {V_LEGGETT_COMPAT:.4f}: Leggett compatible  {'[YES]' if visibility_est <= V_LEGGETT_COMPAT else '[NO]'}")
        print(f"    V > {V_LEGGETT_VIOLATE:.4f}: Leggett violation  {'[YES]' if visibility_est > V_LEGGETT_VIOLATE else '[NO]'}")

    return {
        'density_matrix': rho,
        'tangle': tangle,
        'concurrence': concurrence,
        'purity': purity,
        'fidelity': fidelity,
        'visibility_estimate': (2 * concurrence + 1) / 3,
        'expectations': two_qubit_exp,
        'single_expectations': single_exp
    }
