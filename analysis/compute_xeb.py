#!/usr/bin/env python3
"""
Compute Linear Cross-Entropy Benchmarking (XEB) from experimental results.

XEB = 2^n * <p_ideal(x)>_samples - 1

Where:
- n = number of qubits
- p_ideal(x) = ideal probability of bitstring x from statevector simulation
- <...>_samples = average over experimental samples weighted by frequency

For perfect quantum computer: XEB = 1
For random noise (uniform distribution): XEB ≈ 0
"""

import json
import numpy as np
from pathlib import Path
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# Import circuit creation from leggett module
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from leggett import create_leggett_circuit_for_angle, create_leggett_circuit_for_angle_six


def compute_ideal_probabilities(phi_rad, num_pairs=12):
    """
    Compute ideal probability distribution from statevector simulation.

    Args:
        phi_rad: Angle φ in radians
        num_pairs: Number of qubit pairs (6 or 12)

    Returns:
        Dictionary mapping bitstrings to ideal probabilities
    """
    # Create the same circuit used in experiment (without measurements)
    if num_pairs == 12:
        qc = create_leggett_circuit_for_angle(phi_rad)
    else:
        qc = create_leggett_circuit_for_angle_six(phi_rad)

    # Remove measurements to get statevector
    qc_no_meas = qc.remove_final_measurements(inplace=False)

    # Get statevector
    sv = Statevector.from_instruction(qc_no_meas)
    probs = sv.probabilities_dict()

    return probs


def compute_xeb(counts, ideal_probs, num_qubits):
    """
    Compute linear XEB from experimental counts and ideal probabilities.

    XEB = 2^n * sum_x(f_exp(x) * p_ideal(x)) - 1

    Args:
        counts: Dictionary of experimental bitstring counts
        ideal_probs: Dictionary of ideal probabilities
        num_qubits: Number of qubits in the circuit

    Returns:
        XEB fidelity value
    """
    total_shots = sum(counts.values())

    # Calculate weighted average of ideal probabilities
    weighted_sum = 0.0
    for bitstring, count in counts.items():
        freq = count / total_shots
        p_ideal = ideal_probs.get(bitstring, 0.0)
        weighted_sum += freq * p_ideal

    # XEB = 2^n * <p_ideal> - 1
    xeb = (2 ** num_qubits) * weighted_sum - 1

    return xeb


def compute_xeb_per_pair(counts, ideal_probs, num_pairs=None):
    """
    Compute XEB for each qubit pair independently.

    This marginalizes over all other qubits to compute 2-qubit XEB per pair.

    Args:
        counts: Dictionary of experimental bitstring counts
        ideal_probs: Dictionary of ideal probabilities
        num_pairs: Number of qubit pairs (auto-detected if None)

    Returns:
        List of XEB values, one per pair
    """
    total_shots = sum(counts.values())

    # Auto-detect number of qubits from bitstring length
    sample_bitstring = next(iter(counts.keys()))
    num_qubits = len(sample_bitstring)

    if num_pairs is None:
        num_pairs = num_qubits // 2

    xeb_per_pair = []

    for pair_idx in range(num_pairs):
        # Marginalize to get 2-qubit distribution for this pair
        exp_marginal = {'00': 0, '01': 0, '10': 0, '11': 0}
        ideal_marginal = {'00': 0.0, '01': 0.0, '10': 0.0, '11': 0.0}

        for bitstring, count in counts.items():
            # Extract bits for this pair (bitstring is MSB first)
            alice_bit = bitstring[num_qubits - 1 - 2*pair_idx]
            bob_bit = bitstring[num_qubits - 2 - 2*pair_idx]
            pair_bits = alice_bit + bob_bit
            exp_marginal[pair_bits] += count

        for bitstring, prob in ideal_probs.items():
            alice_bit = bitstring[num_qubits - 1 - 2*pair_idx]
            bob_bit = bitstring[num_qubits - 2 - 2*pair_idx]
            pair_bits = alice_bit + bob_bit
            ideal_marginal[pair_bits] += prob

        # Compute 2-qubit XEB for this pair
        weighted_sum = 0.0
        for pair_bits in ['00', '01', '10', '11']:
            freq = exp_marginal[pair_bits] / total_shots
            p_ideal = ideal_marginal[pair_bits]
            weighted_sum += freq * p_ideal

        xeb_pair = 4 * weighted_sum - 1  # 2^2 = 4 for 2 qubits
        xeb_per_pair.append(xeb_pair)

    return xeb_per_pair


def load_job_data(job_dir, platform='ibm'):
    """
    Load all job data from a directory.

    Args:
        job_dir: Path to directory containing job JSON files
        platform: 'ibm' or 'ionq'

    Returns:
        List of (phi_deg, counts, num_shots) tuples
    """
    job_dir = Path(job_dir)
    results = []

    pattern = 'job_*.json' if platform == 'ibm' else 'task_*.json'

    for json_file in sorted(job_dir.glob(pattern)):
        with open(json_file) as f:
            data = json.load(f)

        phi_deg = data['phi_deg']
        counts = data['counts']
        num_shots = data['num_shots']

        results.append((phi_deg, counts, num_shots))

    return results


def main():
    base = Path('/Users/adc/qsim/relative_phase_variant/geometry/simulator_gnuradio')

    print("=" * 70)
    print("Cross-Entropy Benchmarking (XEB) Analysis")
    print("=" * 70)

    # Cache ideal probabilities (same for same |phi| and num_pairs)
    ideal_probs_cache = {}

    def get_ideal_probs(phi_deg, num_pairs=6):
        key = (abs(phi_deg), num_pairs)
        if key not in ideal_probs_cache:
            print(f"  Computing ideal probabilities for |φ| = {abs(phi_deg)}° ({num_pairs} pairs)...")
            phi_rad = np.radians(abs(phi_deg))
            ideal_probs_cache[key] = compute_ideal_probabilities(phi_rad, num_pairs=num_pairs)
        return ideal_probs_cache[key]

    results_summary = {}

    # IBM Pittsburgh results (12 qubits = 6 pairs from job files)
    print("\n" + "-" * 70)
    print("IBM Pittsburgh (12-qubit, 6 pairs)")
    print("-" * 70)

    ibm_jobs = load_job_data(base / 'ibm_jobs', platform='ibm')
    ibm_xeb_global = []
    ibm_xeb_pairs = []

    for phi_deg, counts, num_shots in ibm_jobs:
        # Detect number of qubits from bitstring length
        num_qubits = len(next(iter(counts.keys())))
        num_pairs = num_qubits // 2

        ideal_probs = get_ideal_probs(phi_deg, num_pairs=num_pairs)

        # Global XEB
        xeb_global = compute_xeb(counts, ideal_probs, num_qubits=num_qubits)
        ibm_xeb_global.append((phi_deg, xeb_global))

        # Per-pair 2-qubit XEB
        xeb_pairs = compute_xeb_per_pair(counts, ideal_probs)
        ibm_xeb_pairs.append((phi_deg, xeb_pairs))

        print(f"  φ = {phi_deg:+.0f}°: XEB_global = {xeb_global:.4f}, "
              f"XEB_pair_avg = {np.mean(xeb_pairs):.4f} ± {np.std(xeb_pairs):.4f}")

    ibm_avg_global = np.mean([x[1] for x in ibm_xeb_global])
    ibm_avg_pair = np.mean([np.mean(x[1]) for x in ibm_xeb_pairs])

    results_summary['IBM'] = {
        'global': ibm_xeb_global,
        'per_pair': ibm_xeb_pairs,
        'avg_global': ibm_avg_global,
        'avg_pair': ibm_avg_pair
    }

    print(f"\n  IBM Average XEB (global {num_qubits}Q): {ibm_avg_global:.4f}")
    print(f"  IBM Average XEB (per pair):   {ibm_avg_pair:.4f}")

    # IonQ Forte results (12 qubits = 6 pairs from job files)
    print("\n" + "-" * 70)
    print("IonQ Forte-1 (12-qubit, 6 pairs)")
    print("-" * 70)

    ionq_jobs = load_job_data(base / 'ionq_jobs', platform='ionq')
    ionq_xeb_global = []
    ionq_xeb_pairs = []

    for phi_deg, counts, num_shots in ionq_jobs:
        # Detect number of qubits from bitstring length
        num_qubits = len(next(iter(counts.keys())))
        num_pairs = num_qubits // 2

        ideal_probs = get_ideal_probs(phi_deg, num_pairs=num_pairs)

        # Global XEB
        xeb_global = compute_xeb(counts, ideal_probs, num_qubits=num_qubits)
        ionq_xeb_global.append((phi_deg, xeb_global))

        # Per-pair 2-qubit XEB
        xeb_pairs = compute_xeb_per_pair(counts, ideal_probs)
        ionq_xeb_pairs.append((phi_deg, xeb_pairs))

        print(f"  φ = {phi_deg:+.0f}°: XEB_global = {xeb_global:.4f}, "
              f"XEB_pair_avg = {np.mean(xeb_pairs):.4f} ± {np.std(xeb_pairs):.4f}")

    ionq_avg_global = np.mean([x[1] for x in ionq_xeb_global])
    ionq_avg_pair = np.mean([np.mean(x[1]) for x in ionq_xeb_pairs])

    results_summary['IonQ'] = {
        'global': ionq_xeb_global,
        'per_pair': ionq_xeb_pairs,
        'avg_global': ionq_avg_global,
        'avg_pair': ionq_avg_pair
    }

    print(f"\n  IonQ Average XEB (global {num_qubits}Q): {ionq_avg_global:.4f}")
    print(f"  IonQ Average XEB (per pair):   {ionq_avg_pair:.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("XEB SUMMARY")
    print("=" * 70)
    print(f"\n{'Platform':<20} {'XEB (24Q global)':<20} {'XEB (per-pair avg)':<20}")
    print("-" * 60)
    print(f"{'IBM Pittsburgh':<20} {ibm_avg_global:<20.4f} {ibm_avg_pair:<20.4f}")
    print(f"{'IonQ Forte-1':<20} {ionq_avg_global:<20.4f} {ionq_avg_pair:<20.4f}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  XEB = 1.0 : Perfect fidelity (ideal quantum computer)")
    print("  XEB = 0.0 : Random noise (no quantum signal)")
    print("  XEB < 0   : Worse than random (systematic errors)")
    print("=" * 70)

    # Generate LaTeX table
    print("\n" + "=" * 70)
    print("LaTeX Table")
    print("=" * 70)

    print(r"""
\begin{table}[H]
\centering
\caption{Cross-Entropy Benchmarking (XEB) Results}
\label{tab:xeb_results}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Platform} & \textbf{XEB (24Q)} & \textbf{XEB (per-pair)} \\
\hline""")
    print(f"IBM Pittsburgh & {ibm_avg_global:.3f} & {ibm_avg_pair:.3f} \\\\")
    print(f"IonQ Forte-1 & {ionq_avg_global:.3f} & {ionq_avg_pair:.3f} \\\\")
    print(r"""\hline
\end{tabular}
\end{table}
""")

    # Save detailed results
    output_file = base / 'paper/ibm/xeb_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'IBM': {
                'global_xeb': [(phi, float(xeb)) for phi, xeb in ibm_xeb_global],
                'pair_xeb': [(phi, [float(x) for x in xeb]) for phi, xeb in ibm_xeb_pairs],
                'avg_global': float(ibm_avg_global),
                'avg_pair': float(ibm_avg_pair)
            },
            'IonQ': {
                'global_xeb': [(phi, float(xeb)) for phi, xeb in ionq_xeb_global],
                'pair_xeb': [(phi, [float(x) for x in xeb]) for phi, xeb in ionq_xeb_pairs],
                'avg_global': float(ionq_avg_global),
                'avg_pair': float(ionq_avg_pair)
            }
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
