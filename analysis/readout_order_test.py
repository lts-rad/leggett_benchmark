#!/usr/bin/env python3
"""
Test readout feedline crosstalk hypothesis.
Measure (11,18) FIRST, before other pairs, to see if it improves.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import json
from datetime import datetime

# All 12 pairs
ALL_PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

TARGET_PAIR = (11, 18)
TARGET_IDX = ALL_PAIRS.index(TARGET_PAIR)  # Index 6


def create_sequential_readout_circuit():
    """
    Create circuit with all 12 pairs, YY basis.
    Measure (11,18) FIRST, then the rest.
    """
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    # Step 1: Prepare all singlets and apply YY rotations
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1

        # Singlet
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

        # Y-basis rotation
        qc.sdg(qa)
        qc.h(qa)
        qc.sdg(qb)
        qc.h(qb)

    # Step 2: Barrier before any measurement
    qc.barrier()

    # Step 3: Measure (11,18) FIRST
    # Target pair is at index TARGET_IDX, so qubits are 2*TARGET_IDX and 2*TARGET_IDX+1
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1
    qc.measure(target_qa, target_qa)
    qc.measure(target_qb, target_qb)

    # Step 4: Barrier
    qc.barrier()

    # Step 5: Measure all other pairs
    for i in range(n_pairs):
        if i != TARGET_IDX:
            qa, qb = 2 * i, 2 * i + 1
            qc.measure(qa, qa)
            qc.measure(qb, qb)

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


def main():
    print("=" * 60)
    print("READOUT ORDER TEST")
    print("=" * 60)
    print("Hypothesis: Readout feedline crosstalk hurts (11,18)")
    print("Test: Measure (11,18) FIRST, before other pairs")
    print()

    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')

    # Build layout
    layout = []
    for q1, q2 in ALL_PAIRS:
        layout.extend([q1, q2])

    print(f"Pairs: {len(ALL_PAIRS)}")
    print(f"Target: {TARGET_PAIR} (measured first)")
    print()

    # Create circuit
    qc = create_sequential_readout_circuit()
    qc_t = transpile(qc, backend=backend, optimization_level=3, initial_layout=layout)

    print(f"Circuit depth: {qc_t.depth()}")
    print(f"Gate counts: {dict(qc_t.count_ops())}")
    print()

    # Submit
    options = SamplerOptions()
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XY4"
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True

    sampler = Sampler(backend, options=options)
    job = sampler.run([qc_t], shots=1000)
    print(f"Submitted: {job.job_id()}")
    print("Waiting for results...")

    result = job.result()[0]

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

    num_shots = sum(counts.values())
    corrs = extract_correlations(counts, num_shots, len(ALL_PAIRS))

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"{'Pair':<15} {'YY Correlation':<15} {'Notes'}")
    print("-" * 50)

    for i, (pair, corr) in enumerate(zip(ALL_PAIRS, corrs)):
        note = "<-- TARGET (measured first)" if pair == TARGET_PAIR else ""
        print(f"{str(pair):<15} {corr:+.4f}          {note}")

    target_yy = corrs[TARGET_IDX]
    print("-" * 50)
    print()
    print("Comparison for (11,18):")
    print(f"  This test (measured first): {target_yy:+.4f}")
    print(f"  12-pair simultaneous:       -0.9000")
    print(f"  Isolated:                   -0.9520")
    print()

    if target_yy < -0.94:
        print("RESULT: Significant improvement! Readout crosstalk likely culprit.")
    elif target_yy < -0.92:
        print("RESULT: Modest improvement. Readout may be partial factor.")
    else:
        print("RESULT: No improvement. Readout order not the issue.")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'job_id': job.job_id(),
        'experiment': 'readout_order_test',
        'target_pair': TARGET_PAIR,
        'target_yy': float(target_yy),
        'all_correlations': {str(p): float(c) for p, c in zip(ALL_PAIRS, corrs)},
        'comparison': {
            'measured_first': float(target_yy),
            'simultaneous_12pair': -0.900,
            'isolated': -0.952
        }
    }

    with open('readout_order_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to readout_order_test.json")


if __name__ == "__main__":
    main()
