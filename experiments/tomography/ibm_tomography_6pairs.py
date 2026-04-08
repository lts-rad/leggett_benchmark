import os
#!/usr/bin/env python3
"""
6-Pair State Tomography on IBM Quantum

Performs 3x3 Pauli basis tomography on 6 qubit pairs.
Compares tomography-derived visibility/tangle with Leggett experiment results.

Usage:
    python ibm_tomography_6pairs.py                    # Noiseless simulator
    python ibm_tomography_6pairs.py --noise-model      # Noise model simulator
    python ibm_tomography_6pairs.py --ibm              # Real hardware
    python ibm_tomography_6pairs.py --ibm --dry-run    # Transpile only
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from tomography import (
    apply_measurement_rotation,
    reconstruct_density_matrix,
    calculate_tangle,
    calculate_purity,
    calculate_fidelity_singlet,
)

# Best 6 non-overlapping pairs from current calibration
BEST_PAIRS_6 = [
    (86, 87), (113, 114), (99, 115),
    (60, 61), (37, 45), (46, 47)
]

# 9 Pauli basis combinations
TOMOGRAPHY_BASES = [
    ('X', 'X'), ('X', 'Y'), ('X', 'Z'),
    ('Y', 'X'), ('Y', 'Y'), ('Y', 'Z'),
    ('Z', 'X'), ('Z', 'Y'), ('Z', 'Z')
]


def create_npair_circuit(basis_a, basis_b, num_pairs):
    """Create circuit with num_pairs singlets measured in given basis."""
    qc = QuantumCircuit(2 * num_pairs, 2 * num_pairs)

    for pair_idx in range(num_pairs):
        qa = 2 * pair_idx
        qb = 2 * pair_idx + 1

        # Create singlet |Psi^-⟩
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

        # Measurement rotations
        apply_measurement_rotation(qc, qa, basis_a)
        apply_measurement_rotation(qc, qb, basis_b)

    qc.measure(range(2 * num_pairs), range(2 * num_pairs))
    return qc


def extract_correlations(counts, num_shots, num_pairs):
    """Extract correlation for each pair from counts."""
    corrs = [0.0] * num_pairs
    n_qubits = 2 * num_pairs

    for bitstring, count in counts.items():
        for pair_idx in range(num_pairs):
            a_pos = (n_qubits - 1) - 2 * pair_idx
            b_pos = (n_qubits - 1) - (2 * pair_idx + 1)
            a_val = (-1) ** int(bitstring[a_pos])
            b_val = (-1) ** int(bitstring[b_pos])
            corrs[pair_idx] += a_val * b_val * count

    return [c / num_shots for c in corrs]


def run_tomography(num_shots=8192, use_ibm=False, use_noise_model=False,
                   backend_name="ibm_pittsburgh", dry_run=False,
                   qubit_pairs=None):
    """Run 3x3 tomography on 6 qubit pairs."""

    if qubit_pairs is None:
        qubit_pairs = BEST_PAIRS_6

    num_pairs = len(qubit_pairs)

    # Flatten for initial_layout
    initial_layout = []
    for q1, q2 in qubit_pairs:
        initial_layout.extend([q1, q2])

    print(f"\n{'='*60}")
    print(f"6-PAIR TOMOGRAPHY ({num_shots} shots/basis)")
    print(f"{'='*60}")
    print(f"Pairs: {qubit_pairs}")
    print(f"Layout: {initial_layout}")

    # Setup backend
    backend = None
    noise_model = None

    if use_ibm or use_noise_model:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        print(f"Backend: {backend_name}")

        if use_noise_model:
            from qiskit_aer.noise import NoiseModel
            print("Loading noise model...")
            noise_model = NoiseModel.from_backend(backend)

    all_results = {}

    if use_ibm:
        # PARALLEL SUBMISSION: Submit all 9 jobs at once
        from qiskit_ibm_runtime import SamplerV2 as Sampler, SamplerOptions

        print(f"\n--- Submitting all 9 basis measurements in parallel ---")

        options = SamplerOptions()
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = "XX"
        options.twirling.enable_gates = True

        jobs = {}
        for basis_a, basis_b in TOMOGRAPHY_BASES:
            basis_name = f"{basis_a}{basis_b}"
            qc = create_npair_circuit(basis_a, basis_b, num_pairs)
            qc_t = transpile(qc, backend=backend, optimization_level=3,
                            initial_layout=initial_layout)

            print(f"  {basis_name}: depth={qc_t.depth()}", end="")

            if dry_run:
                print(" (dry run)")
                continue

            sampler = Sampler(backend, options=options)
            job = sampler.run([qc_t], shots=num_shots)
            jobs[basis_name] = job
            print(f" -> job={job.job_id()}")

        if dry_run:
            return None

        # Wait for all jobs
        print(f"\n--- Waiting for all 9 jobs ---")
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
                print(f"  {basis_name}: FAILED")
                continue

            counts = job.result()[0].data.meas.get_counts()
            corrs = extract_correlations(counts, num_shots, num_pairs)
            all_results[basis_name] = corrs
            print(f"  {basis_name}: {[f'{c:.3f}' for c in corrs[:3]]}...")

    else:
        # Sequential for simulator
        for basis_idx, (basis_a, basis_b) in enumerate(TOMOGRAPHY_BASES):
            basis_name = f"{basis_a}{basis_b}"
            print(f"[{basis_idx+1}/9] {basis_name}: ", end="", flush=True)

            qc = create_npair_circuit(basis_a, basis_b, num_pairs)

            if use_noise_model:
                qc_t = transpile(qc, backend=backend, optimization_level=3,
                                initial_layout=initial_layout)
                sim = AerSimulator(noise_model=noise_model)
                result = sim.run(qc_t, shots=num_shots).result()
                counts = result.get_counts()
            else:
                sim = AerSimulator()
                result = sim.run(qc, shots=num_shots).result()
                counts = result.get_counts()

            corrs = extract_correlations(counts, num_shots, num_pairs)
            all_results[basis_name] = corrs
            print(f"done (pair0={corrs[0]:.3f})")

    if len(all_results) < 9:
        print(f"\nWARNING: Only {len(all_results)}/9 bases completed!")
        if len(all_results) == 0:
            return None

    # Analyze per pair
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Pair':>12} {'Vis':>8} {'Conc':>8} {'Purity':>8} {'Fidelity':>8}")
    print("-" * 50)

    pair_results = []

    for pair_idx, (q1, q2) in enumerate(qubit_pairs):
        exp = {b: all_results[b][pair_idx] for b in all_results}
        rho = reconstruct_density_matrix(exp, single_exp=None)
        tangle, conc = calculate_tangle(rho)
        purity = calculate_purity(rho)
        fidelity = calculate_fidelity_singlet(rho)
        vis = (2 * conc + 1) / 3

        print(f"({q1:>3},{q2:>3})  {vis:>8.4f} {conc:>8.4f} {purity:>8.4f} {fidelity:>8.4f}")

        pair_results.append({
            'pair': (q1, q2),
            'visibility': float(vis),
            'concurrence': float(conc),
            'tangle': float(tangle),
            'purity': float(purity),
            'fidelity': float(fidelity),
            'expectations': exp
        })

    vis_list = [p['visibility'] for p in pair_results]
    conc_list = [p['concurrence'] for p in pair_results]

    print("-" * 50)
    print(f"{'Average':>12} {np.mean(vis_list):>8.4f} {np.mean(conc_list):>8.4f}")
    print(f"{'Std':>12} {np.std(vis_list):>8.4f} {np.std(conc_list):>8.4f}")

    return {
        'qubit_pairs': qubit_pairs,
        'num_shots': num_shots,
        'timestamp': datetime.now().isoformat(),
        'pair_results': pair_results,
        'raw_basis_results': all_results,
        'summary': {
            'avg_visibility': float(np.mean(vis_list)),
            'std_visibility': float(np.std(vis_list)),
            'avg_concurrence': float(np.mean(conc_list))
        }
    }


def main():
    import sys

    use_ibm = '--ibm' in sys.argv
    use_noise_model = '--noise-model' in sys.argv
    dry_run = '--dry-run' in sys.argv
    backend_name = "ibm_pittsburgh"
    num_shots = 8192

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])

    # Output filename
    if use_ibm:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'tomography_6pairs_{backend_name}.json')
    elif use_noise_model:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'tomography_6pairs_{backend_name}_NOISE_MODEL.json')
    else:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'tomography_6pairs_SIM_noiseless.json')

    print("=" * 60)
    print("6-PAIR STATE TOMOGRAPHY")
    print("=" * 60)
    if use_ibm:
        print(f"Mode: IBM Hardware ({backend_name})")
    elif use_noise_model:
        print(f"Mode: Noise model ({backend_name})")
    else:
        print("Mode: Noiseless simulator")
    print(f"Shots: {num_shots} per basis ({num_shots * 9} total)")
    print(f"Output: {output_file}")

    results = run_tomography(
        num_shots=num_shots,
        use_ibm=use_ibm,
        use_noise_model=use_noise_model,
        backend_name=backend_name,
        dry_run=dry_run
    )

    if results:
        print(f"\nSaving to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print("Done!")


if __name__ == "__main__":
    main()
