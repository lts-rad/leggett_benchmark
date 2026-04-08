import os
#!/usr/bin/env python3
"""
Single-pair state tomography on IBM Quantum.
Run 9 Pauli basis measurements on a single qubit pair in isolation.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from tomography import (
    apply_measurement_rotation,
    reconstruct_density_matrix,
    calculate_tangle,
    calculate_purity,
    calculate_fidelity_singlet,
)

TOMOGRAPHY_BASES = [
    ('X', 'X'), ('X', 'Y'), ('X', 'Z'),
    ('Y', 'X'), ('Y', 'Y'), ('Y', 'Z'),
    ('Z', 'X'), ('Z', 'Y'), ('Z', 'Z')
]


def create_singlet_circuit(basis_a, basis_b):
    """Create singlet state with measurement in specified bases."""
    qc = QuantumCircuit(2, 2)

    # Create singlet |Psi^-⟩ = (|01⟩ - |10⟩)/sqrt(2)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)

    # Measurement rotations
    apply_measurement_rotation(qc, 0, basis_a)
    apply_measurement_rotation(qc, 1, basis_b)

    qc.measure([0, 1], [0, 1])
    return qc


def extract_correlation(counts, num_shots):
    """Extract correlation from counts."""
    corr = 0.0
    for bitstring, count in counts.items():
        # Bitstring is reversed: bit 0 is rightmost
        a_bit = int(bitstring[-1])  # qubit 0
        b_bit = int(bitstring[-2])  # qubit 1
        a_val = (-1) ** a_bit
        b_val = (-1) ** b_bit
        corr += a_val * b_val * count
    return corr / num_shots


def run_single_pair_tomography(q1, q2, num_shots=8192, backend_name="ibm_pittsburgh",
                                dry_run=False):
    """Run full tomography on a single qubit pair."""

    print(f"\n{'='*60}")
    print(f"SINGLE-PAIR TOMOGRAPHY: ({q1}, {q2})")
    print(f"{'='*60}")
    print(f"Backend: {backend_name}")
    print(f"Shots per basis: {num_shots}")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    initial_layout = [q1, q2]
    print(f"Initial layout: {initial_layout}")

    # Configure options
    options = SamplerOptions()
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XX"
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True

    # Submit all 9 jobs in parallel
    print(f"\n--- Submitting 9 basis measurements ---")
    jobs = {}

    for basis_a, basis_b in TOMOGRAPHY_BASES:
        basis_name = f"{basis_a}{basis_b}"

        qc = create_singlet_circuit(basis_a, basis_b)
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

    # Wait for completion
    print(f"\n--- Waiting for jobs ---")
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

        time.sleep(10)

    # Collect results
    print(f"\n--- Results ---")
    expectations = {}
    job_ids = {}

    for basis_name, job in jobs.items():
        job_ids[basis_name] = job.job_id()
        status = str(job.status())

        if status in ['ERROR', 'CANCELLED']:
            print(f"  {basis_name}: FAILED ({status})")
            continue

        result = job.result()
        pub_result = result[0]

        # Get counts
        if hasattr(pub_result.data, 'meas'):
            counts = pub_result.data.meas.get_counts()
        elif hasattr(pub_result.data, 'c'):
            counts = pub_result.data.c.get_counts()
        else:
            for attr in dir(pub_result.data):
                if not attr.startswith('_'):
                    obj = getattr(pub_result.data, attr)
                    if hasattr(obj, 'get_counts'):
                        counts = obj.get_counts()
                        break

        corr = extract_correlation(counts, num_shots)
        expectations[basis_name] = corr
        print(f"  {basis_name}: {corr:+.4f}")

    if len(expectations) < 9:
        print(f"\nWARNING: Only {len(expectations)}/9 bases completed!")

    # Reconstruct density matrix
    print(f"\n--- Analysis ---")
    rho = reconstruct_density_matrix(expectations, single_exp=None)
    tangle, conc = calculate_tangle(rho)
    purity = calculate_purity(rho)
    fidelity = calculate_fidelity_singlet(rho)
    visibility = (2 * conc + 1) / 3

    print(f"  XX: {expectations.get('XX', 'N/A'):+.4f}  (ideal: -1)")
    print(f"  YY: {expectations.get('YY', 'N/A'):+.4f}  (ideal: -1)")
    print(f"  ZZ: {expectations.get('ZZ', 'N/A'):+.4f}  (ideal: -1)")
    print()
    print(f"  Concurrence:  {conc:.4f}")
    print(f"  Tangle:       {tangle:.4f}")
    print(f"  Purity:       {purity:.4f}")
    print(f"  Fidelity:     {fidelity:.4f}")
    print(f"  Visibility:   {visibility:.4f}")

    return {
        'pair': (q1, q2),
        'backend': backend_name,
        'num_shots': num_shots,
        'timestamp': datetime.now().isoformat(),
        'job_ids': job_ids,
        'expectations': {k: float(v) for k, v in expectations.items()},
        'concurrence': float(conc),
        'tangle': float(tangle),
        'purity': float(purity),
        'fidelity': float(fidelity),
        'visibility': float(visibility),
        'density_matrix_real': rho.real.tolist(),
        'density_matrix_imag': rho.imag.tolist()
    }


def main():
    import sys

    # Default: pair (11, 18)
    q1, q2 = 11, 18
    backend_name = "ibm_pittsburgh"
    num_shots = 8192
    dry_run = '--dry-run' in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == '--pair' and i + 2 < len(sys.argv):
            q1 = int(sys.argv[i + 1])
            q2 = int(sys.argv[i + 2])
        elif arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'tomography_pair_{q1}_{q2}_{backend_name}.json')

    results = run_single_pair_tomography(
        q1, q2,
        num_shots=num_shots,
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
