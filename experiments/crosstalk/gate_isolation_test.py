#!/usr/bin/env python3
"""
Isolate which gate causes crosstalk to (11,18).

Exp 1: Others do only X
Exp 2: Others do X + H
Exp 3: Others do X + H + CX

All experiments: (11,18) does full singlet + YY measurement
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import time

ALL_PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

TARGET_PAIR = (11, 18)
TARGET_IDX = ALL_PAIRS.index(TARGET_PAIR)


def create_exp1_circuit():
    """Others: X only"""
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1

        if i == TARGET_IDX:
            # Full singlet for (11,18)
            qc.x(qb)
            qc.h(qa)
            qc.cx(qa, qb)
            qc.z(qa)
        else:
            # Others: only X
            qc.x(qb)

    # YY measurement on all
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.sdg(qa)
        qc.h(qa)
        qc.sdg(qb)
        qc.h(qb)

    qc.measure(range(2 * n_pairs), range(2 * n_pairs))
    return qc


def create_exp2_circuit():
    """Others: X + H"""
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1

        if i == TARGET_IDX:
            # Full singlet for (11,18)
            qc.x(qb)
            qc.h(qa)
            qc.cx(qa, qb)
            qc.z(qa)
        else:
            # Others: X + H
            qc.x(qb)
            qc.h(qa)

    # YY measurement on all
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.sdg(qa)
        qc.h(qa)
        qc.sdg(qb)
        qc.h(qb)

    qc.measure(range(2 * n_pairs), range(2 * n_pairs))
    return qc


def create_exp3_circuit():
    """Others: X + H + CX (no Z)"""
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1

        if i == TARGET_IDX:
            # Full singlet for (11,18)
            qc.x(qb)
            qc.h(qa)
            qc.cx(qa, qb)
            qc.z(qa)
        else:
            # Others: X + H + CX
            qc.x(qb)
            qc.h(qa)
            qc.cx(qa, qb)

    # YY measurement on all
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.sdg(qa)
        qc.h(qa)
        qc.sdg(qb)
        qc.h(qb)

    qc.measure(range(2 * n_pairs), range(2 * n_pairs))
    return qc


def main():
    print("=" * 60)
    print("GATE ISOLATION TEST")
    print("=" * 60)
    print()
    print("Exp 1: Others do X only")
    print("Exp 2: Others do X + H")
    print("Exp 3: Others do X + H + CX")
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

    # Create and transpile circuits
    qc1 = create_exp1_circuit()
    qc2 = create_exp2_circuit()
    qc3 = create_exp3_circuit()

    qc1_t = transpile(qc1, backend=backend, optimization_level=3, initial_layout=layout)
    qc2_t = transpile(qc2, backend=backend, optimization_level=3, initial_layout=layout)
    qc3_t = transpile(qc3, backend=backend, optimization_level=3, initial_layout=layout)

    print(f"Exp 1 depth: {qc1_t.depth()}, CZ count: {qc1_t.count_ops().get('cz', 0)}")
    print(f"Exp 2 depth: {qc2_t.depth()}, CZ count: {qc2_t.count_ops().get('cz', 0)}")
    print(f"Exp 3 depth: {qc3_t.depth()}, CZ count: {qc3_t.count_ops().get('cz', 0)}")
    print()

    # Submit all 3
    sampler = Sampler(backend, options=options)

    print("Submitting jobs...")
    job1 = sampler.run([qc1_t], shots=1000)
    print(f"  Exp 1 (X only):    {job1.job_id()}")

    job2 = sampler.run([qc2_t], shots=1000)
    print(f"  Exp 2 (X + H):     {job2.job_id()}")

    job3 = sampler.run([qc3_t], shots=1000)
    print(f"  Exp 3 (X + H + CX): {job3.job_id()}")

    print("\nWaiting for completion...")
    jobs = [('Exp1', job1), ('Exp2', job2), ('Exp3', job3)]

    while True:
        statuses = [(name, str(j.status())) for name, j in jobs]
        print(f"  {statuses}", end='\r')

        if all(s in ['DONE', 'ERROR', 'CANCELLED'] for _, s in statuses):
            print()
            break
        time.sleep(10)

    # Extract results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    def extract_target_yy(counts, num_shots):
        corr = 0.0
        n_qubits = 24
        qa = 2 * TARGET_IDX
        qb = 2 * TARGET_IDX + 1
        for bitstring, count in counts.items():
            a_pos = (n_qubits - 1) - qa
            b_pos = (n_qubits - 1) - qb
            a_val = (-1) ** int(bitstring[a_pos])
            b_val = (-1) ** int(bitstring[b_pos])
            corr += a_val * b_val * count
        return corr / num_shots

    for name, job in jobs:
        if str(job.status()) != 'DONE':
            print(f"{name}: FAILED")
            continue

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

        yy = extract_target_yy(counts, sum(counts.values()))
        print(f"{name}: (11,18) YY = {yy:+.4f}")

    print()
    print("Comparison:")
    print("  Isolated:     -0.952")
    print("  Full 12-pair: -0.900")


if __name__ == "__main__":
    main()
