#!/usr/bin/env python3
"""
Local Hidden Variable (LHV) model that reproduces Pauli tomography of a singlet.

This demonstrates that standard 9-basis Pauli tomography CANNOT distinguish
a true quantum singlet from a classical LHV model.

The LHV model:
- Each "particle pair" carries hidden variables (a_x, a_y, a_z) ∈ {-1, +1}³
- Qubit A returns a_i when measured in basis i
- Qubit B returns -a_i when measured in basis i (anti-correlated)

This reproduces:
- ⟨XX⟩ = ⟨YY⟩ = ⟨ZZ⟩ = -1 (perfect anti-correlation)
- ⟨XY⟩ = ⟨XZ⟩ = ⟨YZ⟩ = ... = 0 (no cross-correlations)
- ⟨XI⟩ = ⟨YI⟩ = ⟨ZI⟩ = 0 (no marginals)

BUT this model FAILS for non-Pauli measurements (Bell/CHSH/Leggett tests).
"""

import numpy as np


class LHVFakeSinglet:
    """
    Local Hidden Variable fake singlet.

    Each instance represents a "particle pair" with predetermined outcomes.
    """

    def __init__(self):
        # Hidden variables: predetermined outcomes for X, Y, Z measurements
        # Each is randomly ±1
        self.a_x = np.random.choice([-1, 1])
        self.a_y = np.random.choice([-1, 1])
        self.a_z = np.random.choice([-1, 1])

        # Qubit B is anti-correlated (this is the "hidden" correlation)
        self.b_x = -self.a_x
        self.b_y = -self.a_y
        self.b_z = -self.a_z

    def measure_A(self, basis):
        """Measure qubit A in given basis."""
        if basis == 'X':
            return self.a_x
        elif basis == 'Y':
            return self.a_y
        elif basis == 'Z':
            return self.a_z
        else:
            raise ValueError(f"Unknown basis: {basis}")

    def measure_B(self, basis):
        """Measure qubit B in given basis."""
        if basis == 'X':
            return self.b_x
        elif basis == 'Y':
            return self.b_y
        elif basis == 'Z':
            return self.b_z
        else:
            raise ValueError(f"Unknown basis: {basis}")


def run_pauli_tomography_lhv(n_shots=1000):
    """
    Run 9-basis Pauli tomography on LHV fake singlets.

    Returns correlation matrix.
    """
    bases = ['X', 'Y', 'Z']
    correlations = {}

    for basis_a in bases:
        for basis_b in bases:
            results = []
            for _ in range(n_shots):
                pair = LHVFakeSinglet()
                outcome_a = pair.measure_A(basis_a)
                outcome_b = pair.measure_B(basis_b)
                results.append(outcome_a * outcome_b)

            correlations[f'{basis_a}{basis_b}'] = np.mean(results)

    # Also compute marginals
    marginals_a = {}
    marginals_b = {}

    for basis in bases:
        results_a = []
        results_b = []
        for _ in range(n_shots):
            pair = LHVFakeSinglet()
            results_a.append(pair.measure_A(basis))
            results_b.append(pair.measure_B(basis))
        marginals_a[basis] = np.mean(results_a)
        marginals_b[basis] = np.mean(results_b)

    return correlations, marginals_a, marginals_b


def ideal_singlet_correlations():
    """Return ideal quantum singlet correlations for comparison."""
    return {
        'XX': -1.0, 'XY': 0.0, 'XZ': 0.0,
        'YX': 0.0, 'YY': -1.0, 'YZ': 0.0,
        'ZX': 0.0, 'ZY': 0.0, 'ZZ': -1.0,
    }


def main():
    print("=" * 60)
    print("LOCAL HIDDEN VARIABLE FAKE SINGLET")
    print("=" * 60)
    print()
    print("This LHV model reproduces Pauli tomography of a singlet,")
    print("demonstrating that tomography alone cannot verify entanglement.")
    print()

    n_shots = 10000
    print(f"Running {n_shots} shots...")

    corr_lhv, marg_a, marg_b = run_pauli_tomography_lhv(n_shots)
    corr_qm = ideal_singlet_correlations()

    print()
    print("=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)
    print()
    print(f"{'Basis':<8} {'LHV':<12} {'QM Singlet':<12} {'Diff':<12}")
    print("-" * 44)

    for key in ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']:
        lhv = corr_lhv[key]
        qm = corr_qm[key]
        diff = abs(lhv - qm)
        print(f"{key:<8} {lhv:<12.4f} {qm:<12.4f} {diff:<12.4f}")

    print()
    print("=" * 60)
    print("MARGINALS (should be 0 for singlet)")
    print("=" * 60)
    print()
    print(f"Qubit A: ⟨X⟩={marg_a['X']:.4f}, ⟨Y⟩={marg_a['Y']:.4f}, ⟨Z⟩={marg_a['Z']:.4f}")
    print(f"Qubit B: ⟨X⟩={marg_b['X']:.4f}, ⟨Y⟩={marg_b['Y']:.4f}, ⟨Z⟩={marg_b['Z']:.4f}")

    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print()
    print("The LHV model PERFECTLY reproduces Pauli tomography!")
    print()
    print("This is why we need Bell/CHSH/Leggett tests:")
    print("- They use NON-ORTHOGONAL measurement bases")
    print("- LHV models cannot reproduce those correlations")
    print()

    # Demonstrate where LHV fails: non-Pauli measurement
    print("=" * 60)
    print("WHERE LHV FAILS: Non-Pauli measurements")
    print("=" * 60)
    print()

    # For singlet, correlation along any direction n is: C(a,b) = -a·b
    # For a=X and b at 45° between X and Z: b = (1,0,1)/√2
    # QM predicts: C = -cos(45°) = -0.707

    # But LHV with predetermined X,Y,Z values cannot reproduce this!
    # LHV would give: C = ⟨a_x · (b_x + b_z)/√2⟩ but b is fixed per-shot

    print("QM prediction for 45° measurement: C = -cos(45°) = -0.707")
    print()
    print("LHV cannot reproduce correlations for arbitrary angles.")
    print("This is the essence of Bell's theorem.")


if __name__ == "__main__":
    main()
