#!/usr/bin/env python3
"""
Test readout feedline crosstalk hypothesis with 2 experiments.

Experiment A: Measure (11,18) FIRST, then measure the rest
Experiment B: Prepare all 12 singlets, but ONLY measure (11,18)

If B shows good results → interference is from readout
If B shows bad results → interference is from state prep/gates
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


def create_experiment_a():
    """
    Experiment A: All 12 pairs YY, measure (11,18) FIRST, then rest.
    """
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    # Prepare all singlets + YY rotations
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)
        qc.sdg(qa)
        qc.h(qa)
        qc.sdg(qb)
        qc.h(qb)

    qc.barrier()

    # Measure (11,18) first
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1
    qc.measure(target_qa, target_qa)
    qc.measure(target_qb, target_qb)

    qc.barrier()

    # Measure rest
    for i in range(n_pairs):
        if i != TARGET_IDX:
            qa, qb = 2 * i, 2 * i + 1
            qc.measure(qa, qa)
            qc.measure(qb, qb)

    return qc


def create_experiment_b():
    """
    Experiment B: Prepare all 12 singlets, but ONLY measure (11,18).
    Other pairs just sit there unmeasured.
    """
    n_pairs = len(ALL_PAIRS)
    # Only need 2 classical bits for the target pair
    qc = QuantumCircuit(2 * n_pairs, 2)

    # Prepare ALL singlets (no measurement rotations for others)
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

    qc.barrier()

    # Only apply YY rotation to (11,18)
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1
    qc.sdg(target_qa)
    qc.h(target_qa)
    qc.sdg(target_qb)
    qc.h(target_qb)

    # Only measure (11,18)
    qc.measure(target_qa, 0)
    qc.measure(target_qb, 1)

    return qc


def extract_target_correlation_a(counts, num_shots):
    """Extract (11,18) correlation from Experiment A (24 classical bits)."""
    n_qubits = 24
    corr = 0.0
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1

    for bitstring, count in counts.items():
        a_pos = (n_qubits - 1) - target_qa
        b_pos = (n_qubits - 1) - target_qb
        a_val = (-1) ** int(bitstring[a_pos])
        b_val = (-1) ** int(bitstring[b_pos])
        corr += a_val * b_val * count

    return corr / num_shots


def extract_target_correlation_b(counts, num_shots):
    """Extract (11,18) correlation from Experiment B (2 classical bits)."""
    corr = 0.0
    for bitstring, count in counts.items():
        # bitstring is 2 bits: c1 c0 where c0=qa, c1=qb
        a_val = (-1) ** int(bitstring[1])  # c0
        b_val = (-1) ** int(bitstring[0])  # c1
        corr += a_val * b_val * count
    return corr / num_shots


def main():
    print("=" * 60)
    print("READOUT HYPOTHESIS TEST")
    print("=" * 60)
    print()
    print("Experiment A: All 12 YY, measure (11,18) FIRST")
    print("Experiment B: All 12 singlets, ONLY measure (11,18)")
    print()

    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')

    layout = []
    for q1, q2 in ALL_PAIRS:
        layout.extend([q1, q2])

    options = SamplerOptions()
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XY4"
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True

    # Create circuits
    qc_a = create_experiment_a()
    qc_b = create_experiment_b()

    qc_a_t = transpile(qc_a, backend=backend, optimization_level=3, initial_layout=layout)
    qc_b_t = transpile(qc_b, backend=backend, optimization_level=3, initial_layout=layout)

    print(f"Experiment A: depth={qc_a_t.depth()}")
    print(f"Experiment B: depth={qc_b_t.depth()}")
    print()

    # Submit both
    sampler = Sampler(backend, options=options)

    print("Submitting jobs...")
    job_a = sampler.run([qc_a_t], shots=1000)
    print(f"  Exp A: {job_a.job_id()}")

    job_b = sampler.run([qc_b_t], shots=1000)
    print(f"  Exp B: {job_b.job_id()}")

    # Wait
    print("\nWaiting for completion...")
    while True:
        status_a = str(job_a.status())
        status_b = str(job_b.status())
        print(f"  A: {status_a}, B: {status_b}", end='\r')

        if status_a in ['DONE', 'ERROR'] and status_b in ['DONE', 'ERROR']:
            print()
            break
        time.sleep(10)

    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    results = {}

    # Experiment A
    if str(job_a.status()) == 'DONE':
        res_a = job_a.result()[0]
        if hasattr(res_a.data, 'meas'):
            counts_a = res_a.data.meas.get_counts()
        elif hasattr(res_a.data, 'c'):
            counts_a = res_a.data.c.get_counts()
        else:
            for attr in dir(res_a.data):
                if not attr.startswith('_'):
                    obj = getattr(res_a.data, attr)
                    if hasattr(obj, 'get_counts'):
                        counts_a = obj.get_counts()
                        break

        yy_a = extract_target_correlation_a(counts_a, sum(counts_a.values()))
        results['exp_a'] = yy_a
        print(f"\nExperiment A (measure first): {yy_a:+.4f}")
    else:
        print(f"\nExperiment A: FAILED ({job_a.status()})")

    # Experiment B
    if str(job_b.status()) == 'DONE':
        res_b = job_b.result()[0]
        if hasattr(res_b.data, 'meas'):
            counts_b = res_b.data.meas.get_counts()
        elif hasattr(res_b.data, 'c'):
            counts_b = res_b.data.c.get_counts()
        else:
            for attr in dir(res_b.data):
                if not attr.startswith('_'):
                    obj = getattr(res_b.data, attr)
                    if hasattr(obj, 'get_counts'):
                        counts_b = obj.get_counts()
                        break

        yy_b = extract_target_correlation_b(counts_b, sum(counts_b.values()))
        results['exp_b'] = yy_b
        print(f"Experiment B (only measure):  {yy_b:+.4f}")
    else:
        print(f"Experiment B: FAILED ({job_b.status()})")

    # Comparison
    print()
    print("-" * 40)
    print("Comparison for (11,18) YY:")
    print(f"  Isolated:              -0.9520")
    print(f"  12-pair simultaneous:  -0.9000")
    if 'exp_a' in results:
        print(f"  Exp A (measure first): {results['exp_a']:+.4f}")
    if 'exp_b' in results:
        print(f"  Exp B (only measure):  {results['exp_b']:+.4f}")

    print()
    if 'exp_b' in results:
        if results['exp_b'] < -0.94:
            print("CONCLUSION: Exp B good → READOUT CROSSTALK is the culprit")
        elif results['exp_b'] < -0.92:
            print("CONCLUSION: Exp B moderate → Mixed effects")
        else:
            print("CONCLUSION: Exp B bad → Interference from STATE PREP/GATES")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'job_ids': {'exp_a': job_a.job_id(), 'exp_b': job_b.job_id()},
        'results': results,
        'comparison': {
            'isolated': -0.952,
            'simultaneous_12pair': -0.900,
        }
    }

    with open('readout_hypothesis_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to readout_hypothesis_test.json")


if __name__ == "__main__":
    main()
