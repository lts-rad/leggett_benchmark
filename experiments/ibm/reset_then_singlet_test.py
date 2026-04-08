#!/usr/bin/env python3
"""
Reset test: Create 11 other singlets, barrier, reset (11,18), then make fresh singlet.

If good results -> interference is from simultaneous state prep
If bad results -> other singlets' states interfere with (11,18) measurement
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import json
import time
from datetime import datetime

ALL_PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

TARGET_PAIR = (11, 18)
TARGET_IDX = ALL_PAIRS.index(TARGET_PAIR)


def create_reset_circuit():
    """
    1. Create 11 other singlets (not 11,18)
    2. Barrier
    3. Reset qubits 11 and 18
    4. Create fresh singlet on (11,18)
    5. YY measurement on (11,18) only
    """
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2)

    # Step 1: Create all OTHER singlets
    for i in range(n_pairs):
        if i == TARGET_IDX:
            continue  # Skip (11,18) for now
        qa, qb = 2 * i, 2 * i + 1
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

    qc.barrier()

    # Step 2: Reset (11,18)
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1
    qc.reset(target_qa)
    qc.reset(target_qb)

    qc.barrier()

    # Step 3: Fresh singlet on (11,18)
    qc.x(target_qb)
    qc.h(target_qa)
    qc.cx(target_qa, target_qb)
    qc.z(target_qa)

    qc.barrier()

    # Step 4: YY measurement on (11,18) only
    qc.sdg(target_qa)
    qc.h(target_qa)
    qc.sdg(target_qb)
    qc.h(target_qb)

    qc.measure(target_qa, 0)
    qc.measure(target_qb, 1)

    return qc


def extract_yy(counts, num_shots):
    corr = 0.0
    for bitstring, count in counts.items():
        a_val = (-1) ** int(bitstring[1])
        b_val = (-1) ** int(bitstring[0])
        corr += a_val * b_val * count
    return corr / num_shots


def main():
    print("=" * 60)
    print("RESET THEN SINGLET TEST")
    print("=" * 60)
    print()
    print("1. Create 11 other singlets")
    print("2. Barrier")
    print("3. Reset (11,18)")
    print("4. Fresh singlet on (11,18)")
    print("5. YY measurement on (11,18)")
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

    qc = create_reset_circuit()
    qc_t = transpile(qc, backend=backend, optimization_level=3, initial_layout=layout)

    print(f"Circuit depth: {qc_t.depth()}")
    print(f"Operations: {dict(qc_t.count_ops())}")
    print()

    sampler = Sampler(backend, options=options)

    print("Submitting job...")
    job = sampler.run([qc_t], shots=1000)
    print(f"Job ID: {job.job_id()}")

    print("\nWaiting...")
    while True:
        status = str(job.status())
        print(f"  {status}", end='\r')
        if status in ['DONE', 'ERROR', 'CANCELLED']:
            print()
            break
        time.sleep(10)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if str(job.status()) != 'DONE':
        print(f"Failed: {job.status()}")
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

    yy = extract_yy(counts, sum(counts.values()))
    print(f"\n(11,18) YY: {yy:+.4f}")

    print()
    print("-" * 40)
    print("Comparison:")
    print(f"  Isolated:              -0.952")
    print(f"  Simultaneous 12:       -0.900")
    print(f"  Serialized:            -0.912")
    print(f"  Reset then fresh:      {yy:+.4f}")

    print()
    if yy < -0.94:
        print("CONCLUSION: Reset helps! Interference is from simultaneous prep.")
    else:
        print("CONCLUSION: Still degraded. Other singlets' states interfere.")

    output = {
        'timestamp': datetime.now().isoformat(),
        'job_id': job.job_id(),
        'yy': float(yy),
    }
    with open('reset_then_singlet_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to reset_then_singlet_test.json")


if __name__ == "__main__":
    main()
