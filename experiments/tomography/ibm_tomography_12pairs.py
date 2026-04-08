import os
#!/usr/bin/env python3
"""
12-Pair State Tomography on IBM Quantum

Performs 3x3 Pauli basis tomography on 12 qubit pairs simultaneously.
Compares tomography-derived visibility/tangle with Leggett experiment results.

Strategy: Run 9 circuits (one per basis), each with 12 singlet pairs.
Total: 9 circuits × 24 qubits = 216 logical qubits across all circuits.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from tomography import (
    apply_measurement_rotation,
    reconstruct_density_matrix,
    calculate_tangle,
    calculate_purity,
    calculate_fidelity_singlet,
    get_theoretical_singlet_expectations,
    PAULIS
)


# Best 12 non-overlapping pairs from current calibration
BEST_LAYOUT_12PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

# 9 Pauli basis combinations for tomography
TOMOGRAPHY_BASES = [
    ('X', 'X'), ('X', 'Y'), ('X', 'Z'),
    ('Y', 'X'), ('Y', 'Y'), ('Y', 'Z'),
    ('Z', 'X'), ('Z', 'Y'), ('Z', 'Z')
]


def create_12pair_singlet_circuit(basis_a, basis_b):
    """
    Create a circuit with 12 singlet pairs, all measured in the same basis.

    Args:
        basis_a: Measurement basis for Alice ('X', 'Y', or 'Z')
        basis_b: Measurement basis for Bob ('X', 'Y', or 'Z')

    Returns:
        QuantumCircuit with 24 qubits (12 pairs)
    """
    qc = QuantumCircuit(24, 24)

    for pair_idx in range(12):
        qubit_a = 2 * pair_idx      # Alice qubit (even)
        qubit_b = 2 * pair_idx + 1  # Bob qubit (odd)

        # Create singlet |Psi^-⟩ = (|01⟩ - |10⟩)/sqrt(2)
        qc.x(qubit_b)
        qc.h(qubit_a)
        qc.cx(qubit_a, qubit_b)
        qc.z(qubit_a)

        # Apply measurement rotations
        apply_measurement_rotation(qc, qubit_a, basis_a)
        apply_measurement_rotation(qc, qubit_b, basis_b)

    # Measure all qubits
    qc.measure(range(24), range(24))

    return qc


def extract_pair_correlations(counts, num_shots, num_pairs=12):
    """
    Extract correlation for each pair from measurement counts.

    Args:
        counts: Dictionary of bitstring counts
        num_shots: Total shots
        num_pairs: Number of qubit pairs

    Returns:
        List of correlations for each pair
    """
    correlations = [0.0] * num_pairs

    for bitstring, count in counts.items():
        # Bitstring is in reverse order (qubit 23 is leftmost)
        for pair_idx in range(num_pairs):
            # Alice is qubit 2*pair_idx, Bob is qubit 2*pair_idx + 1
            alice_pos = 23 - 2 * pair_idx
            bob_pos = 23 - (2 * pair_idx + 1)

            alice_bit = int(bitstring[alice_pos])
            bob_bit = int(bitstring[bob_pos])

            # Correlation: +1 if same, -1 if different
            alice_val = (-1) ** alice_bit
            bob_val = (-1) ** bob_bit

            correlations[pair_idx] += alice_val * bob_val * count

    # Normalize
    correlations = [c / num_shots for c in correlations]

    return correlations


def run_tomography_12pairs(num_shots, use_ibm, backend_name, dry_run=False,
                           enable_error_mitigation=True, use_noise_model=False,
                           qubit_pairs=None):
    """
    Run full 3x3 tomography on 12 qubit pairs.

    Returns:
        Dictionary with per-pair tomography results
    """
    if qubit_pairs is None:
        qubit_pairs = BEST_LAYOUT_12PAIRS

    print(f"\n{'='*70}")
    print(f"12-PAIR STATE TOMOGRAPHY ({num_shots} shots per basis)")
    print(f"{'='*70}")
    print(f"\nQubit pairs: {qubit_pairs}")

    # Flatten pairs for initial_layout
    initial_layout = []
    for q1, q2 in qubit_pairs:
        initial_layout.extend([q1, q2])

    print(f"Initial layout (24 qubits): {initial_layout}")

    # Connect to backend
    if use_ibm or use_noise_model:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        print(f"Backend: {backend_name}, {backend.num_qubits} qubits")
    else:
        backend = None

    # Store results for each basis
    all_basis_results = {}

    if use_ibm:
        # PARALLEL SUBMISSION: Submit all 9 jobs at once
        print(f"\n--- Submitting all 9 basis measurements in parallel ---")

        # Configure options
        options = SamplerOptions()
        if enable_error_mitigation:
            options.dynamical_decoupling.enable = True
            options.dynamical_decoupling.sequence_type = "XX"
            options.twirling.enable_gates = True
            options.twirling.enable_measure = True

        # Transpile and submit all circuits
        jobs = {}
        for basis_idx, (basis_a, basis_b) in enumerate(TOMOGRAPHY_BASES):
            basis_name = f"{basis_a}{basis_b}"

            qc = create_12pair_singlet_circuit(basis_a, basis_b)
            qc_transpiled = transpile(qc, backend=backend, optimization_level=3,
                                     initial_layout=initial_layout)

            print(f"  {basis_name}: depth={qc_transpiled.depth()}", end="")

            if dry_run:
                print(" (dry run)")
                continue

            sampler = Sampler(backend, options=options)
            job = sampler.run([qc_transpiled], shots=num_shots)
            jobs[basis_name] = job
            print(f" -> job={job.job_id()}")

        if dry_run:
            return None

        # Wait for all jobs to complete
        print(f"\n--- Waiting for all 9 jobs to complete ---")
        start_time = time.time()

        while True:
            all_done = True
            status_line = []

            for basis_name, job in jobs.items():
                status = str(job.status())
                if status not in ['DONE', 'COMPLETED', 'ERROR', 'CANCELLED']:
                    all_done = False
                status_line.append(f"{basis_name}:{status[:4]}")

            elapsed = time.time() - start_time
            print(f"  [{elapsed/60:.1f}m] {' '.join(status_line)}", end='\r')

            if all_done:
                print()
                break

            time.sleep(15)

        # Collect results
        print(f"\n--- Collecting results ---")
        for basis_name, job in jobs.items():
            status = str(job.status())
            if status in ['ERROR', 'CANCELLED']:
                print(f"  {basis_name}: FAILED ({status})")
                continue

            result = job.result()
            pub_result = result[0]
            # Handle different Qiskit versions
            if hasattr(pub_result.data, 'meas'):
                counts = pub_result.data.meas.get_counts()
            elif hasattr(pub_result.data, 'c'):
                counts = pub_result.data.c.get_counts()
            else:
                # Try first available BitArray
                for attr in dir(pub_result.data):
                    if not attr.startswith('_'):
                        obj = getattr(pub_result.data, attr)
                        if hasattr(obj, 'get_counts'):
                            counts = obj.get_counts()
                            break
            correlations = extract_pair_correlations(counts, num_shots)
            all_basis_results[basis_name] = correlations
            print(f"  {basis_name}: {[f'{c:.3f}' for c in correlations[:3]]}...")

    else:
        # Sequential for simulator (noise model or noiseless)
        for basis_idx, (basis_a, basis_b) in enumerate(TOMOGRAPHY_BASES):
            basis_name = f"{basis_a}{basis_b}"
            print(f"\n--- Basis {basis_idx+1}/9: {basis_name} ---")

            qc = create_12pair_singlet_circuit(basis_a, basis_b)

            if use_noise_model:
                from qiskit_aer.noise import NoiseModel
                noise_model = NoiseModel.from_backend(backend)
                qc_transpiled = transpile(qc, backend=backend, optimization_level=3,
                                         initial_layout=initial_layout)
                simulator = AerSimulator(noise_model=noise_model)
                result = simulator.run(qc_transpiled, shots=num_shots).result()
                counts = result.get_counts()
            else:
                # Noiseless simulation
                simulator = AerSimulator()
                result = simulator.run(qc, shots=num_shots).result()
                counts = result.get_counts()

            correlations = extract_pair_correlations(counts, num_shots)
            all_basis_results[basis_name] = correlations
            print(f"  Correlations: {[f'{c:.4f}' for c in correlations]}")

    if len(all_basis_results) < 9:
        print(f"\nWARNING: Only {len(all_basis_results)}/9 bases completed!")
        if len(all_basis_results) == 0:
            return None

    # Now analyze results per pair
    print(f"\n{'='*70}")
    print("PER-PAIR TOMOGRAPHY ANALYSIS")
    print(f"{'='*70}")

    pair_results = []

    for pair_idx, (q1, q2) in enumerate(qubit_pairs):
        print(f"\n--- Pair {pair_idx+1}: ({q1}, {q2}) ---")

        # Build expectation dictionary for this pair
        two_qubit_exp = {}
        for basis_name, correlations in all_basis_results.items():
            two_qubit_exp[basis_name] = correlations[pair_idx]

        print(f"  Expectations: XX={two_qubit_exp['XX']:.4f}, YY={two_qubit_exp['YY']:.4f}, ZZ={two_qubit_exp['ZZ']:.4f}")

        # Reconstruct density matrix (assume maximally mixed marginals for singlet)
        rho = reconstruct_density_matrix(two_qubit_exp, single_exp=None)

        # Calculate entanglement measures
        tangle, concurrence = calculate_tangle(rho)
        purity = calculate_purity(rho)
        fidelity = calculate_fidelity_singlet(rho)

        # Visibility from concurrence (Werner state formula)
        visibility = (2 * concurrence + 1) / 3

        print(f"  Concurrence: {concurrence:.4f}")
        print(f"  Tangle:      {tangle:.4f}")
        print(f"  Purity:      {purity:.4f}")
        print(f"  Fidelity:    {fidelity:.4f}")
        print(f"  Visibility:  {visibility:.4f}")

        pair_results.append({
            'pair': (q1, q2),
            'pair_idx': pair_idx,
            'expectations': two_qubit_exp,
            'concurrence': float(concurrence),
            'tangle': float(tangle),
            'purity': float(purity),
            'fidelity': float(fidelity),
            'visibility': float(visibility),
            'density_matrix_real': rho.real.tolist(),
            'density_matrix_imag': rho.imag.tolist()
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    visibilities = [p['visibility'] for p in pair_results]
    concurrences = [p['concurrence'] for p in pair_results]
    purities = [p['purity'] for p in pair_results]
    fidelities = [p['fidelity'] for p in pair_results]

    print(f"\n{'Pair':>12} {'Visibility':>12} {'Concurrence':>12} {'Purity':>10} {'Fidelity':>10}")
    print("-" * 60)
    for p in pair_results:
        q1, q2 = p['pair']
        print(f"({q1:>3},{q2:>3})   {p['visibility']:>10.4f}   {p['concurrence']:>10.4f}   {p['purity']:>8.4f}   {p['fidelity']:>8.4f}")
    print("-" * 60)
    print(f"{'Average':>12} {np.mean(visibilities):>10.4f}   {np.mean(concurrences):>10.4f}   {np.mean(purities):>8.4f}   {np.mean(fidelities):>8.4f}")
    print(f"{'Std':>12} {np.std(visibilities):>10.4f}   {np.std(concurrences):>10.4f}   {np.std(purities):>8.4f}   {np.std(fidelities):>8.4f}")

    return {
        'qubit_pairs': [(q1, q2) for q1, q2 in qubit_pairs],
        'num_shots': num_shots,
        'timestamp': datetime.now().isoformat(),
        'pair_results': pair_results,
        'summary': {
            'avg_visibility': float(np.mean(visibilities)),
            'std_visibility': float(np.std(visibilities)),
            'avg_concurrence': float(np.mean(concurrences)),
            'avg_purity': float(np.mean(purities)),
            'avg_fidelity': float(np.mean(fidelities))
        },
        'raw_basis_results': {k: v for k, v in all_basis_results.items()}
    }


def main():
    import sys

    # Parse arguments
    use_ibm = '--ibm' in sys.argv
    dry_run = '--dry-run' in sys.argv
    use_noise_model = '--noise-model' in sys.argv
    backend_name = "ibm_pittsburgh"
    num_shots = 4096
    output_file = None
    enable_error_mitigation = '--no-error-mitigation' not in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    if output_file is None:
        if use_ibm:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'tomography_12pairs_{backend_name}.json')
        elif use_noise_model:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'tomography_12pairs_{backend_name}_NOISE_MODEL.json')
        else:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'tomography_12pairs_SIM_noiseless.json')

    print("=" * 70)
    print("12-PAIR STATE TOMOGRAPHY")
    print("=" * 70)
    print(f"\nConfiguration:")
    if use_ibm:
        print(f"  Backend: {backend_name} (REAL HARDWARE)")
    elif use_noise_model:
        print(f"  Backend: {backend_name} noise model (SIMULATOR)")
    else:
        print(f"  Backend: Noiseless simulator")
    print(f"  Shots per basis: {num_shots}")
    print(f"  Total shots: {num_shots * 9} (9 bases)")
    print(f"  Error mitigation: {'enabled' if enable_error_mitigation else 'disabled'}")
    print(f"  Output: {output_file}")

    # Run tomography
    results = run_tomography_12pairs(
        num_shots=num_shots,
        use_ibm=use_ibm,
        backend_name=backend_name,
        dry_run=dry_run,
        enable_error_mitigation=enable_error_mitigation,
        use_noise_model=use_noise_model
    )

    if results is None:
        print("\nDry run completed.")
        return

    # Save results
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("Done!")

    return results


if __name__ == "__main__":
    main()
