#!/usr/bin/env python3
"""
Crosstalk compensation experiment.

Apply Rx/Ry corrections to qubits 11,18 based on idle tomography results.

Experiment 1: YY measurement only on all 12 pairs (compensated)
Experiment 2: Leggett test at phi=+30 and phi=-30 (compensated)

Compensation angles (from idle tomography):
  Qubit 11: Rx(-0.038), Ry(-0.030)
  Qubit 18: Rx(-0.032), Ry(-0.006)
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions

ALL_PAIRS = [
    (86, 87), (113, 114), (99, 115), (60, 61),
    (37, 45), (46, 47), (11, 18), (147, 148),
    (100, 101), (40, 41), (97, 107), (84, 85)
]

TARGET_PAIR = (11, 18)
TARGET_IDX = ALL_PAIRS.index(TARGET_PAIR)

# Compensation angles from idle tomography (radians)
COMP_Q11_RX = -0.038
COMP_Q11_RY = -0.030
COMP_Q18_RX = -0.032
COMP_Q18_RY = -0.006


def bloch_to_angles(vec):
    """Convert Bloch vector to spherical angles (theta, phi)."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def get_leggett_bases(phi_rad):
    """
    Get 6 measurement basis pairs for Leggett test at angle phi.
    Returns list of (name, alice_vec, bob_vec) tuples.
    """
    a1 = np.array([1, 0, 0])  # X
    a2 = np.array([0, 1, 0])  # Y
    a3 = np.array([0, 0, 1])  # Z

    b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1p = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2p = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3p = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    return [
        ('a1_b1', a1, b1),
        ('a1_b1p', a1, b1p),
        ('a2_b2', a2, b2),
        ('a2_b2p', a2, b2p),
        ('a3_b3', a3, b3),
        ('a3_b3p', a3, b3p),
    ]


def create_yy_circuit_compensated():
    """Create 12-pair YY circuit with compensation on (11,18)."""
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    # Prepare all singlets
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

    # Apply compensation to (11,18) AFTER singlet prep
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1
    qc.rx(COMP_Q11_RX, target_qa)
    qc.ry(COMP_Q11_RY, target_qa)
    qc.rx(COMP_Q18_RX, target_qb)
    qc.ry(COMP_Q18_RY, target_qb)

    qc.barrier()

    # YY measurement on all
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.sdg(qa)
        qc.h(qa)
        qc.sdg(qb)
        qc.h(qb)

    qc.measure(range(2 * n_pairs), range(2 * n_pairs))
    return qc


def extract_correlations(counts, num_shots, num_pairs):
    """Extract correlation for each pair."""
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


def get_counts_from_result(result):
    """Extract counts from result object."""
    if hasattr(result.data, 'meas'):
        return result.data.meas.get_counts()
    elif hasattr(result.data, 'c'):
        return result.data.c.get_counts()
    else:
        for attr in dir(result.data):
            if not attr.startswith('_'):
                obj = getattr(result.data, attr)
                if hasattr(obj, 'get_counts'):
                    return obj.get_counts()
    return {}


def run_yy_experiment(backend, sampler, layout):
    """Experiment 1: YY only with compensation."""
    print("=" * 60)
    print("EXPERIMENT 1: YY MEASUREMENT (Compensated)")
    print("=" * 60)
    print()

    qc = create_yy_circuit_compensated()
    qc_t = transpile(qc, backend=backend, optimization_level=3, initial_layout=layout)

    print(f"Circuit depth: {qc_t.depth()}")

    print("Submitting job...")
    job = sampler.run([qc_t], shots=1000)
    print(f"  Job ID: {job.job_id()}")

    print("\nWaiting...")
    while True:
        status = str(job.status())
        print(f"  {status}", end='\r')
        if status in ['DONE', 'ERROR', 'CANCELLED']:
            print()
            break
        time.sleep(10)

    if str(job.status()) != 'DONE':
        print("FAILED")
        return None

    result = job.result()[0]
    counts = get_counts_from_result(result)
    corrs = extract_correlations(counts, sum(counts.values()), len(ALL_PAIRS))

    target_yy = corrs[TARGET_IDX]
    print(f"\n(11,18) YY: {target_yy:+.4f}")
    print(f"  Previous uncompensated: -0.900")
    print(f"  Isolated: -0.952")

    return {'target_yy': target_yy, 'all_corrs': corrs, 'job_id': job.job_id()}


def create_leggett_12pair_circuit(phi_deg=30):
    """
    Create single 12-pair Leggett circuit with compensation on (11,18).

    First 6 pairs: measure 6 bases for phi=+30
    Second 6 pairs: measure 6 bases for phi=-30
    """
    n_pairs = len(ALL_PAIRS)
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    # Prepare all singlets
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1
        qc.x(qb)
        qc.h(qa)
        qc.cx(qa, qb)
        qc.z(qa)

    # Apply compensation to (11,18)
    target_qa = 2 * TARGET_IDX
    target_qb = 2 * TARGET_IDX + 1
    qc.rx(COMP_Q11_RX, target_qa)
    qc.ry(COMP_Q11_RY, target_qa)
    qc.rx(COMP_Q18_RX, target_qb)
    qc.ry(COMP_Q18_RY, target_qb)

    qc.barrier()

    # Get bases for +phi and -phi
    phi_rad = np.radians(phi_deg)
    bases_pos = get_leggett_bases(phi_rad)    # 6 bases for +phi
    bases_neg = get_leggett_bases(-phi_rad)   # 6 bases for -phi

    # First 6 pairs get +phi bases, second 6 pairs get -phi bases
    for i in range(n_pairs):
        qa, qb = 2 * i, 2 * i + 1

        if i < 6:
            # First 6 pairs: +phi
            _, alice_vec, bob_vec = bases_pos[i]
        else:
            # Second 6 pairs: -phi
            _, alice_vec, bob_vec = bases_neg[i - 6]

        theta_a, phi_a = bloch_to_angles(alice_vec)
        theta_b, phi_b = bloch_to_angles(bob_vec)

        qc.rz(-phi_a, qa)
        qc.ry(-theta_a, qa)
        qc.rz(-phi_b, qb)
        qc.ry(-theta_b, qb)

    qc.measure(range(2 * n_pairs), range(2 * n_pairs))
    return qc


def run_leggett_experiment(backend, sampler, layout):
    """Experiment 2: Leggett test at phi=±30° in a single circuit."""
    print()
    print("=" * 60)
    print("EXPERIMENT 2: LEGGETT TEST phi=±30° (Compensated)")
    print("=" * 60)
    print()
    print("First 6 pairs: phi=+30° bases")
    print("Second 6 pairs: phi=-30° bases")
    print(f"(11,18) is pair index {TARGET_IDX}, so measures phi={-30 if TARGET_IDX >= 6 else 30}°")
    print()

    qc = create_leggett_12pair_circuit(phi_deg=30)
    qc_t = transpile(qc, backend=backend, optimization_level=3, initial_layout=layout)

    print(f"Circuit depth: {qc_t.depth()}")

    print("Submitting job...")
    job = sampler.run([qc_t], shots=1000)
    print(f"  Job ID: {job.job_id()}")

    print("\nWaiting...")
    while True:
        status = str(job.status())
        print(f"  {status}", end='\r')
        if status in ['DONE', 'ERROR', 'CANCELLED']:
            print()
            break
        time.sleep(10)

    if str(job.status()) != 'DONE':
        print("FAILED")
        return None

    result = job.result()[0]
    counts = get_counts_from_result(result)
    corrs = extract_correlations(counts, sum(counts.values()), len(ALL_PAIRS))

    # Process results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    phi_rad = np.radians(30)
    bases_pos = get_leggett_bases(phi_rad)
    bases_neg = get_leggett_bases(-phi_rad)

    # Extract correlations by phi group
    results = {30: {}, -30: {}}
    for i in range(6):
        results[30][bases_pos[i][0]] = corrs[i]
        results[-30][bases_neg[i][0]] = corrs[i + 6]

    # Calculate L3 for each phi
    for phi in [30, -30]:
        phi_rad = np.radians(phi)
        c = results[phi]

        term1 = abs(c['a1_b1'] + c['a1_b1p'])
        term2 = abs(c['a2_b2'] + c['a2_b2p'])
        term3 = abs(c['a3_b3'] + c['a3_b3p'])
        L3 = term1 + term2 + term3

        L3_leggett = 2 - (2/3) * abs(np.sin(phi_rad / 2))
        L3_qm = 2 * abs(np.cos(phi_rad / 2))

        print(f"\nphi={phi:+d}°:")
        print(f"  Correlations: {c}")
        print(f"  L3 = {L3:.4f}")
        print(f"  L3_Leggett = {L3_leggett:.4f}")
        print(f"  L3_QM = {L3_qm:.4f}")
        if L3 > L3_leggett:
            print(f"  *** LEGGETT VIOLATION! ***")
        else:
            print(f"  No violation (need L3 > {L3_leggett:.4f})")

    # Note about (11,18)
    print(f"\nNote: (11,18) is pair index {TARGET_IDX}")
    if TARGET_IDX < 6:
        print(f"  Measured phi=+30° basis: {bases_pos[TARGET_IDX][0]}")
        print(f"  Correlation: {corrs[TARGET_IDX]:+.4f}")
    else:
        print(f"  Measured phi=-30° basis: {bases_neg[TARGET_IDX-6][0]}")
        print(f"  Correlation: {corrs[TARGET_IDX]:+.4f}")

    return {'correlations': corrs, 'results': results, 'job_id': job.job_id()}


def main():
    print("=" * 60)
    print("COMPENSATED CROSSTALK TEST")
    print("=" * 60)
    print()
    print("Compensation angles:")
    print(f"  Qubit 11: Rx({np.degrees(COMP_Q11_RX):.2f}°), Ry({np.degrees(COMP_Q11_RY):.2f}°)")
    print(f"  Qubit 18: Rx({np.degrees(COMP_Q18_RX):.2f}°), Ry({np.degrees(COMP_Q18_RY):.2f}°)")
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

    sampler = Sampler(backend, options=options)

    # Create both circuits
    print("Creating circuits...")
    qc_yy = create_yy_circuit_compensated()
    qc_leggett = create_leggett_12pair_circuit(phi_deg=30)

    qc_yy_t = transpile(qc_yy, backend=backend, optimization_level=3, initial_layout=layout)
    qc_leggett_t = transpile(qc_leggett, backend=backend, optimization_level=3, initial_layout=layout)

    print(f"  YY circuit depth: {qc_yy_t.depth()}")
    print(f"  Leggett circuit depth: {qc_leggett_t.depth()}")

    # Submit both jobs upfront
    print("\nSubmitting both jobs...")
    job_yy = sampler.run([qc_yy_t], shots=1000)
    job_leggett = sampler.run([qc_leggett_t], shots=1000)

    print(f"  YY job: {job_yy.job_id()}")
    print(f"  Leggett job: {job_leggett.job_id()}")

    # Wait for both
    print("\nWaiting for completion...")
    jobs = [('YY', job_yy), ('Leggett', job_leggett)]
    while True:
        statuses = [(n, str(j.status())) for n, j in jobs]
        print(f"  {statuses}", end='\r')
        if all(s in ['DONE', 'ERROR', 'CANCELLED'] for _, s in statuses):
            print()
            break
        time.sleep(10)

    # Process YY results
    print()
    print("=" * 60)
    print("EXPERIMENT 1: YY MEASUREMENT (Compensated)")
    print("=" * 60)

    results_yy = None
    if str(job_yy.status()) == 'DONE':
        result = job_yy.result()[0]
        counts = get_counts_from_result(result)
        corrs = extract_correlations(counts, sum(counts.values()), len(ALL_PAIRS))
        target_yy = corrs[TARGET_IDX]
        results_yy = {'target_yy': target_yy, 'all_corrs': corrs, 'job_id': job_yy.job_id()}

        print(f"\n(11,18) YY: {target_yy:+.4f}")
        print(f"  Previous uncompensated: -0.900")
        print(f"  Isolated: -0.952")
    else:
        print(f"YY job failed: {job_yy.status()}")

    # Process Leggett results
    print()
    print("=" * 60)
    print("EXPERIMENT 2: LEGGETT TEST phi=±30° (Compensated)")
    print("=" * 60)
    print()
    print("First 6 pairs: phi=+30° bases")
    print("Second 6 pairs: phi=-30° bases")
    print(f"(11,18) is pair index {TARGET_IDX}, so measures phi={-30 if TARGET_IDX >= 6 else 30}°")

    results_leggett = None
    if str(job_leggett.status()) == 'DONE':
        result = job_leggett.result()[0]
        counts = get_counts_from_result(result)
        corrs = extract_correlations(counts, sum(counts.values()), len(ALL_PAIRS))

        phi_rad = np.radians(30)
        bases_pos = get_leggett_bases(phi_rad)
        bases_neg = get_leggett_bases(-phi_rad)

        # Extract correlations by phi group
        results = {30: {}, -30: {}}
        for i in range(6):
            results[30][bases_pos[i][0]] = corrs[i]
            results[-30][bases_neg[i][0]] = corrs[i + 6]

        # Calculate L3 for each phi
        for phi in [30, -30]:
            phi_r = np.radians(phi)
            c = results[phi]

            term1 = abs(c['a1_b1'] + c['a1_b1p'])
            term2 = abs(c['a2_b2'] + c['a2_b2p'])
            term3 = abs(c['a3_b3'] + c['a3_b3p'])
            L3 = term1 + term2 + term3

            L3_leggett = 2 - (2/3) * abs(np.sin(phi_r / 2))
            L3_qm = 2 * abs(np.cos(phi_r / 2))

            print(f"\nphi={phi:+d}°:")
            print(f"  Correlations: {c}")
            print(f"  L3 = {L3:.4f}")
            print(f"  L3_Leggett = {L3_leggett:.4f}")
            print(f"  L3_QM = {L3_qm:.4f}")
            if L3 > L3_leggett:
                print(f"  *** LEGGETT VIOLATION! ***")
            else:
                print(f"  No violation (need L3 > {L3_leggett:.4f})")

        # Note about (11,18)
        print(f"\nNote: (11,18) is pair index {TARGET_IDX}")
        if TARGET_IDX < 6:
            print(f"  Measured phi=+30° basis: {bases_pos[TARGET_IDX][0]}")
            print(f"  Correlation: {corrs[TARGET_IDX]:+.4f}")
        else:
            print(f"  Measured phi=-30° basis: {bases_neg[TARGET_IDX-6][0]}")
            print(f"  Correlation: {corrs[TARGET_IDX]:+.4f}")

        results_leggett = {'correlations': corrs, 'results': results, 'job_id': job_leggett.job_id()}
    else:
        print(f"Leggett job failed: {job_leggett.status()}")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'compensation': {
            'q11_rx': COMP_Q11_RX,
            'q11_ry': COMP_Q11_RY,
            'q18_rx': COMP_Q18_RX,
            'q18_ry': COMP_Q18_RY,
        },
        'yy_results': results_yy,
        'leggett_results': results_leggett,
    }

    with open('compensated_crosstalk_test.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to compensated_crosstalk_test.json")


if __name__ == "__main__":
    main()
