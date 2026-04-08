#!/usr/bin/env python3
"""
Single-qubit tomography on (11,18) while other pairs run CZ gates.

1. Prepare 11 other singlets (their CZ gates run)
2. Qubits 11 and 18 stay idle in |00⟩
3. Measure 11 and 18 in X, Y, Z bases

This shows what crosstalk the other CZ gates cause on idle qubits.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import json
import time
from datetime import datetime
import numpy as np

ALL_PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

TARGET_PAIR = (11, 18)
TARGET_IDX = ALL_PAIRS.index(TARGET_PAIR)


def create_circuit(basis):
    """
    Create circuit where other pairs make singlets,
    but (11,18) stays idle, then measured in given basis.

    basis: 'X', 'Y', or 'Z'
    """
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2)

    # Prepare all OTHER singlets
    for i in range(n_pairs):
        if i == TARGET_IDX:
            continue  # Leave (11,18) idle
        qa, qb = 2 * i, 2 * i + 1
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

    qc.barrier()

    # Measurement rotation for (11,18)
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1

    if basis == 'X':
        qc.h(target_qa)
        qc.h(target_qb)
    elif basis == 'Y':
        qc.sdg(target_qa)
        qc.h(target_qa)
        qc.sdg(target_qb)
        qc.h(target_qb)
    # Z basis: no rotation needed

    qc.measure(target_qa, 0)
    qc.measure(target_qb, 1)

    return qc


def extract_expectations(counts, num_shots):
    """Extract single-qubit expectations from 2-bit measurement."""
    # counts are 2-bit strings: 'ba' where a=qubit11, b=qubit18
    exp_11 = 0.0
    exp_18 = 0.0

    for bitstring, count in counts.items():
        val_11 = (-1) ** int(bitstring[1])  # rightmost bit
        val_18 = (-1) ** int(bitstring[0])  # leftmost bit
        exp_11 += val_11 * count
        exp_18 += val_18 * count

    return exp_11 / num_shots, exp_18 / num_shots


def main():
    print("=" * 60)
    print("IDLE QUBIT TOMOGRAPHY")
    print("=" * 60)
    print()
    print("Other 11 pairs make singlets (run CZ gates)")
    print("Qubits 11 and 18 stay idle in |00⟩")
    print("Then measure 11 and 18 in X, Y, Z bases")
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

    # Create circuits for X, Y, Z bases
    circuits = {}
    for basis in ['X', 'Y', 'Z']:
        qc = create_circuit(basis)
        qc_t = transpile(qc, backend=backend, optimization_level=3, initial_layout=layout)
        circuits[basis] = qc_t
        print(f"Basis {basis}: depth={qc_t.depth()}")

    print()

    # Submit jobs
    sampler = Sampler(backend, options=options)
    jobs = {}

    print("Submitting jobs...")
    for basis, qc_t in circuits.items():
        job = sampler.run([qc_t], shots=1000)
        jobs[basis] = job
        print(f"  {basis}: {job.job_id()}")

    # Wait
    print("\nWaiting...")
    while True:
        statuses = {b: str(j.status()) for b, j in jobs.items()}
        print(f"  {statuses}", end='\r')
        if all(s in ['DONE', 'ERROR', 'CANCELLED'] for s in statuses.values()):
            print()
            break
        time.sleep(10)

    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    expectations = {'qubit_11': {}, 'qubit_18': {}}

    for basis, job in jobs.items():
        if str(job.status()) != 'DONE':
            print(f"{basis}: FAILED")
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

        exp_11, exp_18 = extract_expectations(counts, sum(counts.values()))
        expectations['qubit_11'][basis] = exp_11
        expectations['qubit_18'][basis] = exp_18

    print()
    print("Single-qubit Bloch vector components:")
    print()
    print(f"{'Qubit':<10} {'<X>':<10} {'<Y>':<10} {'<Z>':<10} {'|r|':<10}")
    print("-" * 50)

    for qubit in ['qubit_11', 'qubit_18']:
        x = expectations[qubit].get('X', 0)
        y = expectations[qubit].get('Y', 0)
        z = expectations[qubit].get('Z', 0)
        r = np.sqrt(x**2 + y**2 + z**2)
        print(f"{qubit:<10} {x:+.4f}     {y:+.4f}     {z:+.4f}     {r:.4f}")

    print()
    print("Expected for |0⟩: <X>=0, <Y>=0, <Z>=+1, |r|=1")
    print()

    # Check for errors
    z_11 = expectations['qubit_11'].get('Z', 0)
    z_18 = expectations['qubit_18'].get('Z', 0)
    y_11 = expectations['qubit_11'].get('Y', 0)
    y_18 = expectations['qubit_18'].get('Y', 0)

    if z_11 < 0.95 or z_18 < 0.95:
        print("WARNING: <Z> significantly below 1 - bit flip errors or leakage")

    if abs(y_11) > 0.05 or abs(y_18) > 0.05:
        print(f"WARNING: Non-zero <Y> detected - phase error from CZ crosstalk!")
        theta_11 = np.degrees(np.arcsin(y_11)) if abs(y_11) < 1 else 90
        theta_18 = np.degrees(np.arcsin(y_18)) if abs(y_18) < 1 else 90
        print(f"  Estimated phase rotation: qubit 11 = {theta_11:.1f}°, qubit 18 = {theta_18:.1f}°")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'job_ids': {b: j.job_id() for b, j in jobs.items()},
        'expectations': expectations,
        'expected_for_ground_state': {'X': 0, 'Y': 0, 'Z': 1}
    }

    with open('idle_qubit_tomography.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to idle_qubit_tomography.json")


if __name__ == "__main__":
    main()
