#!/usr/bin/env python3
"""
Serialized singlet preparation test.

Prepare each singlet one at a time with barriers between them.
Only measure (11,18) in YY basis.

If this shows good results -> simultaneous CZ gates are the problem
If this shows bad results -> cumulative ZZ crosstalk even when serialized
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import json
import time
from datetime import datetime

# All 12 pairs
ALL_PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

TARGET_PAIR = (11, 18)
TARGET_IDX = ALL_PAIRS.index(TARGET_PAIR)


def create_serialized_circuit():
    """
    Create circuit with serialized singlet preparation.
    Each pair prepared one at a time with barriers.
    Only (11,18) measured in YY basis.
    """
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2)  # Only 2 classical bits for target

    # Serialize singlet preparation: one pair at a time
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1

        # Singlet state
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

        # Barrier after each singlet
        qc.barrier()

    # YY rotation only on target pair
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1
    qc.sdg(target_qa)
    qc.h(target_qa)
    qc.sdg(target_qb)
    qc.h(target_qb)

    # Only measure target
    qc.measure(target_qa, 0)
    qc.measure(target_qb, 1)

    return qc


def extract_yy_correlation(counts, num_shots):
    """Extract YY correlation from 2-bit measurement."""
    corr = 0.0
    for bitstring, count in counts.items():
        # bitstring is 2 bits: c1 c0 where c0=qa, c1=qb
        a_val = (-1) ** int(bitstring[1])  # c0
        b_val = (-1) ** int(bitstring[0])  # c1
        corr += a_val * b_val * count
    return corr / num_shots


def main():
    print("=" * 60)
    print("SERIALIZED SINGLET PREPARATION TEST")
    print("=" * 60)
    print()
    print("All 12 singlets prepared ONE AT A TIME with barriers")
    print("Only (11,18) measured in YY basis")
    print()

    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')

    # Build layout
    layout = []
    for q1, q2 in ALL_PAIRS:
        layout.extend([q1, q2])

    options = SamplerOptions()
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XY4"
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True

    # Create and transpile
    qc = create_serialized_circuit()
    qc_t = transpile(qc, backend=backend, optimization_level=3, initial_layout=layout)

    print(f"Circuit depth: {qc_t.depth()}")
    print(f"CZ count: {qc_t.count_ops().get('cz', 0)}")
    print()

    # Submit
    sampler = Sampler(backend, options=options)

    print("Submitting job...")
    job = sampler.run([qc_t], shots=1000)
    print(f"Job ID: {job.job_id()}")

    # Wait
    print("\nWaiting for completion...")
    while True:
        status = str(job.status())
        print(f"  Status: {status}", end='\r')

        if status in ['DONE', 'ERROR', 'CANCELLED']:
            print()
            break
        time.sleep(10)

    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if str(job.status()) != 'DONE':
        print(f"Job failed: {job.status()}")
        return

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

    yy = extract_yy_correlation(counts, sum(counts.values()))
    print(f"\n(11,18) YY correlation: {yy:+.4f}")

    print()
    print("-" * 40)
    print("Comparison:")
    print(f"  Isolated (11,18):          -0.952")
    print(f"  Simultaneous 12 pairs:     -0.900")
    print(f"  Serialized (this test):    {yy:+.4f}")

    print()
    if yy < -0.94:
        print("CONCLUSION: Serializing helps! Simultaneous CZ gates are the problem.")
    elif yy < -0.92:
        print("CONCLUSION: Partial improvement. Some effect from serialization.")
    else:
        print("CONCLUSION: No improvement. Crosstalk persists even when serialized.")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'job_id': job.job_id(),
        'experiment': 'serialized_singlet',
        'target_pair': TARGET_PAIR,
        'yy_correlation': float(yy),
        'circuit_depth': qc_t.depth(),
        'comparison': {
            'isolated': -0.952,
            'simultaneous_12': -0.900,
            'serialized': float(yy)
        }
    }

    with open('serialized_singlet_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to serialized_singlet_test.json")


if __name__ == "__main__":
    main()
