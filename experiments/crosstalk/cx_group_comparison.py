#!/usr/bin/env python3
"""
CX-only experiment: Compare which group of pairs' CX gates interfere with (11,18).

Exp 1: (11,18) full singlet + first 6 pairs do CX only
Exp 2: (11,18) full singlet + other 5 pairs do CX only

This isolates which specific pairs' CZ gates cause interference.
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

# Split other pairs into two groups
OTHER_PAIRS = [p for p in ALL_PAIRS if p != TARGET_PAIR]
GROUP_1 = OTHER_PAIRS[:6]  # First 6: (86,87), (113,114), (99,115), (60,61), (37,45), (46,47)
GROUP_2 = OTHER_PAIRS[6:]  # Remaining 5: (147,148), (100,101), (40,41), (97,107), (84,85)


def create_cx_only_circuit(target_pair, other_pairs):
    """
    Create circuit where:
    - target_pair does full singlet + YY measurement
    - other_pairs do ONLY X + H + CX (no Z, just entangling gate)
    """
    all_pairs = [target_pair] + list(other_pairs)
    n_pairs = len(all_pairs)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    # State preparation
    for i, pair in enumerate(all_pairs):
        qa, qb = 2 * i, 2 * i + 1

        if pair == target_pair:
            # Full singlet for target
            qc.x(qb)
            qc.h(qa)
            qc.cx(qa, qb)
            qc.z(qa)
        else:
            # Others: X + H + CX only (Bell state, not singlet)
            qc.x(qb)
            qc.h(qa)
            qc.cx(qa, qb)

    qc.barrier()

    # YY measurement on all pairs
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.sdg(qa)
        qc.h(qa)
        qc.sdg(qb)
        qc.h(qb)

    qc.measure(range(2 * n_pairs), range(2 * n_pairs))
    return qc, all_pairs


def extract_target_yy(counts, num_shots):
    """Extract YY correlation for target pair (always first in layout)."""
    corr = 0.0
    n_qubits = len(list(counts.keys())[0])

    for bitstring, count in counts.items():
        # Target is at index 0: qubits 0 and 1
        a_pos = (n_qubits - 1) - 0
        b_pos = (n_qubits - 1) - 1
        a_val = (-1) ** int(bitstring[a_pos])
        b_val = (-1) ** int(bitstring[b_pos])
        corr += a_val * b_val * count

    return corr / num_shots


def main():
    print("=" * 60)
    print("CX GROUP COMPARISON TEST")
    print("=" * 60)
    print()
    print(f"Target: {TARGET_PAIR}")
    print(f"Group 1 (6 pairs): {GROUP_1}")
    print(f"Group 2 (5 pairs): {GROUP_2}")
    print()
    print("Exp 1: Target + Group 1 do CX")
    print("Exp 2: Target + Group 2 do CX")
    print()

    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')

    options = SamplerOptions()
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XY4"
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True

    # Create circuits
    qc1, pairs1 = create_cx_only_circuit(TARGET_PAIR, GROUP_1)
    qc2, pairs2 = create_cx_only_circuit(TARGET_PAIR, GROUP_2)

    # Build layouts
    layout1 = []
    for q1, q2 in pairs1:
        layout1.extend([q1, q2])

    layout2 = []
    for q1, q2 in pairs2:
        layout2.extend([q1, q2])

    qc1_t = transpile(qc1, backend=backend, optimization_level=3, initial_layout=layout1)
    qc2_t = transpile(qc2, backend=backend, optimization_level=3, initial_layout=layout2)

    print(f"Exp 1: {len(pairs1)} pairs, depth={qc1_t.depth()}, CZ count={qc1_t.count_ops().get('cz', 0)}")
    print(f"Exp 2: {len(pairs2)} pairs, depth={qc2_t.depth()}, CZ count={qc2_t.count_ops().get('cz', 0)}")
    print()

    # Submit
    sampler = Sampler(backend, options=options)

    print("Submitting jobs...")
    job1 = sampler.run([qc1_t], shots=1000)
    print(f"  Exp 1 (Group 1): {job1.job_id()}")

    job2 = sampler.run([qc2_t], shots=1000)
    print(f"  Exp 2 (Group 2): {job2.job_id()}")

    # Wait
    print("\nWaiting for completion...")
    jobs = [('Exp1', job1), ('Exp2', job2)]

    while True:
        statuses = [(name, str(j.status())) for name, j in jobs]
        print(f"  {statuses}", end='\r')

        if all(s in ['DONE', 'ERROR', 'CANCELLED'] for _, s in statuses):
            print()
            break
        time.sleep(10)

    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    results = {}

    for name, job in jobs:
        if str(job.status()) != 'DONE':
            print(f"{name}: FAILED ({job.status()})")
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
        results[name] = yy
        print(f"{name}: (11,18) YY = {yy:+.4f}")

    print()
    print("-" * 40)
    print("Comparison:")
    print(f"  Isolated (11,18):      -0.952")
    print(f"  All 12 pairs:          -0.900")
    print(f"  Exp 3 (X+H+CX on all): -0.916")
    if 'Exp1' in results:
        print(f"  Exp 1 (Group 1 CX):    {results['Exp1']:+.4f}")
    if 'Exp2' in results:
        print(f"  Exp 2 (Group 2 CX):    {results['Exp2']:+.4f}")

    print()
    if 'Exp1' in results and 'Exp2' in results:
        if abs(results['Exp1'] - results['Exp2']) < 0.02:
            print("CONCLUSION: Both groups cause similar interference")
        elif results['Exp1'] < results['Exp2']:
            print(f"CONCLUSION: Group 1 causes MORE interference ({results['Exp1']:+.4f} vs {results['Exp2']:+.4f})")
            print(f"  Problematic pairs: {GROUP_1}")
        else:
            print(f"CONCLUSION: Group 2 causes MORE interference ({results['Exp2']:+.4f} vs {results['Exp1']:+.4f})")
            print(f"  Problematic pairs: {GROUP_2}")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'job_ids': {'exp1': job1.job_id(), 'exp2': job2.job_id()},
        'group_1': [str(p) for p in GROUP_1],
        'group_2': [str(p) for p in GROUP_2],
        'results': results,
        'comparison': {
            'isolated': -0.952,
            'all_12_pairs': -0.900,
            'exp3_cx_all': -0.916,
        }
    }

    with open('cx_group_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to cx_group_comparison.json")


if __name__ == "__main__":
    main()
