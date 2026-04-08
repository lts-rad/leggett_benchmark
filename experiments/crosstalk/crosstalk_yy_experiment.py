import os
#!/usr/bin/env python3
"""
Crosstalk experiment: Run YY measurement on (11,18) with varying group sizes.
Test if adding other pairs degrades (11,18) performance.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions

# All 12 pairs
ALL_PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

# (11, 18) always included, split others into 3 groups of ~4
TARGET_PAIR = (11, 18)
OTHER_PAIRS = [p for p in ALL_PAIRS if p != TARGET_PAIR]

# 4 experiments: (11,18) + 3 others each (last group has 2 others)
GROUPS = [
    [TARGET_PAIR, (86, 87), (113, 114), (99, 115)],      # Group A: 4 pairs
    [TARGET_PAIR, (60, 61), (37, 45), (46, 47)],         # Group B: 4 pairs
    [TARGET_PAIR, (147, 148), (100, 101), (40, 41)],     # Group C: 4 pairs
    [TARGET_PAIR, (97, 107), (84, 85)],                  # Group D: 3 pairs
]


def create_yy_circuit(pairs):
    """Create circuit with singlets measured in YY basis."""
    n_pairs = len(pairs)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1

        # Singlet
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

        # Y-basis measurement rotation
        qc.sdg(qa)
        qc.h(qa)
        qc.sdg(qb)
        qc.h(qb)

    qc.measure(range(2 * n_pairs), range(2 * n_pairs))
    return qc


def extract_correlations(counts, num_shots, num_pairs):
    """Extract YY correlation for each pair."""
    corrs = [0.0] * num_pairs
    n_qubits = 2 * num_pairs

    for bitstring, count in counts.items():
        for i in range(num_pairs):
            a_pos = (n_qubits - 1) - 2 * i
            b_pos = (n_qubits - 1) - (2 * i + 1)
            a_val = (-1) ** int(bitstring[a_pos])
            b_val = (-1) ** int(bitstring[b_pos])
            corrs[i] += a_val * b_val * count

    return [c / num_shots for c in corrs]


def run_experiment(num_shots=1000, backend_name="ibm_pittsburgh"):
    """Run YY measurement on 3 groups, each containing (11,18)."""

    print(f"\n{'='*60}")
    print("CROSSTALK EXPERIMENT: YY basis only")
    print(f"{'='*60}")
    print(f"Target pair: {TARGET_PAIR}")
    print(f"Shots per group: {num_shots}")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    options = SamplerOptions()
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XX"
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True

    results = {}
    jobs = {}

    # Submit all 3 jobs
    print(f"\n--- Submitting 3 group experiments ---")

    for group_idx, pairs in enumerate(GROUPS):
        group_name = f"group_{chr(65 + group_idx)}"  # A, B, C

        # Build layout
        layout = []
        for q1, q2 in pairs:
            layout.extend([q1, q2])

        qc = create_yy_circuit(pairs)
        qc_t = transpile(qc, backend=backend, optimization_level=3, initial_layout=layout)

        print(f"  {group_name}: pairs={pairs}, depth={qc_t.depth()}", end="")

        sampler = Sampler(backend, options=options)
        job = sampler.run([qc_t], shots=num_shots)
        jobs[group_name] = {'job': job, 'pairs': pairs}
        print(f" -> job={job.job_id()}")

    # Wait for completion
    print(f"\n--- Waiting for jobs ---")
    start_time = time.time()

    while True:
        all_done = True
        status_line = []

        for name, info in jobs.items():
            status = str(info['job'].status())
            if status not in ['DONE', 'COMPLETED', 'ERROR', 'CANCELLED']:
                all_done = False
            status_line.append(f"{name}:{status[:4]}")

        elapsed = time.time() - start_time
        print(f"  [{elapsed/60:.1f}m] {' '.join(status_line)}", end='\r')

        if all_done:
            print()
            break
        time.sleep(10)

    # Collect results
    print(f"\n--- Results ---")
    print(f"\n{'Group':<10} {'Pairs':<40} {'(11,18) YY':>12}")
    print("-" * 65)

    target_yy_values = []

    for group_name, info in jobs.items():
        job = info['job']
        pairs = info['pairs']

        if str(job.status()) in ['ERROR', 'CANCELLED']:
            print(f"  {group_name}: FAILED")
            continue

        result = job.result()[0]

        # Get counts
        if hasattr(result.data, 'meas'):
            counts = result.data.meas.get_counts()
        elif hasattr(result.data, 'c'):
            counts = result.data.c.get_counts()
        else:
            for attr in dir(result.data):
                if not attr.startswith('_'):
                    obj = getattr(result.data, attr)
                    if hasattr(obj, 'get_counts'):
                        counts = obj.get_counts()
                        break

        corrs = extract_correlations(counts, num_shots, len(pairs))

        # Find (11,18) index
        target_idx = pairs.index(TARGET_PAIR)
        target_yy = corrs[target_idx]
        target_yy_values.append(target_yy)

        pairs_str = str(pairs)[:38]
        print(f"{group_name:<10} {pairs_str:<40} {target_yy:>+12.4f}")

        results[group_name] = {
            'pairs': pairs,
            'correlations': {str(p): c for p, c in zip(pairs, corrs)},
            'target_yy': target_yy,
            'job_id': job.job_id()
        }

    # Summary
    print("-" * 65)
    print(f"\n(11,18) YY correlation across groups:")
    print(f"  Mean:  {np.mean(target_yy_values):+.4f}")
    print(f"  Std:   {np.std(target_yy_values):.4f}")
    print(f"  Range: {min(target_yy_values):+.4f} to {max(target_yy_values):+.4f}")
    print(f"\nCompare to:")
    print(f"  Isolated (11,18): -0.952")
    print(f"  12-pair run:      -0.900")

    return {
        'timestamp': datetime.now().isoformat(),
        'target_pair': TARGET_PAIR,
        'num_shots': num_shots,
        'groups': results,
        'target_yy_values': target_yy_values,
        'summary': {
            'mean': float(np.mean(target_yy_values)),
            'std': float(np.std(target_yy_values)),
        }
    }


if __name__ == "__main__":
    results = run_experiment(num_shots=1000)

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'crosstalk_yy_experiment.json')
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Done!")
