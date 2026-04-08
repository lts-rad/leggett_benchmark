#!/usr/bin/env python3
"""
RERUN: Rotated layout ±30° only with 10k shots.
Uses the rotated best_layout from the rotate script.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle, extract_correlations_from_counts, calc_leggett_for_angle


def run_30deg_rotated():
    """Run just the ±30° angle with rotated layout, 10k shots."""

    phi_deg = 30
    phi_rad = np.radians(phi_deg)
    num_shots = 10000
    backend_name = "ibm_pittsburgh"

    print(f"\n{'='*70}")
    print(f"RERUN: Rotated layout φ = ±30° (10k shots, 24 qubits)")
    print(f"{'='*70}")

    # Create circuit
    qc = create_leggett_circuit_for_angle(phi_rad)

    # Connect to IBM
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    print(f"  Transpiling for {backend_name}...")
    print(f"  Original circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    # ROTATED layout (same as in rotate script)
    best_layout = [109, 110, 54, 55, 13, 14, 67, 68, 2, 3, 94, 95, 93, 92, 153, 152, 29, 30, 75, 74, 107, 97, 112, 113]

    print(f"  Using ROTATED 24-qubit layout: {best_layout}")

    qc_transpiled = transpile(qc, backend=backend, optimization_level=3, initial_layout=best_layout)
    print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

    # Log physical qubit mapping
    layout = qc_transpiled.layout
    print(f"  Physical qubit mapping:")
    for logical_idx in range(qc.num_qubits):
        physical_qubit = layout.initial_layout._v2p[qc.qubits[logical_idx]]
        pair_idx = logical_idx // 2
        role = 'Alice' if logical_idx % 2 == 0 else 'Bob'
        print(f"    Logical {logical_idx} (pair {pair_idx}, {role}) -> Physical qubit {physical_qubit}")

    # Configure error mitigation
    options = SamplerOptions()
    options.execution.init_qubits = True
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XY4"
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True
    print(f"  Error mitigation: twirling + dynamical decoupling (XY4) enabled")

    # Run with Sampler
    print(f"  Submitting job...")
    sampler = Sampler(backend, options=options)
    job = sampler.run([qc_transpiled], shots=num_shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Waiting for results...")
    print(f"  You can check status at: https://quantum.ibm.com/jobs/{job_id}")

    # Wait with status updates
    start_time = time.time()
    while True:
        status = job.status()
        elapsed = time.time() - start_time
        print(f"  Status: {status} (elapsed: {elapsed/60:.1f} min)", end='\r')

        if status in ['DONE', 'COMPLETED', 'ERROR', 'CANCELLED']:
            print()
            break

        time.sleep(30)

    if status == 'ERROR' or status == 'CANCELLED':
        print(f"  Job {status}!")
        return None

    result = job.result()

    # Extract counts
    pub_result = result[0]
    counts_array = pub_result.data.meas.get_counts()
    counts = {bitstring: count for bitstring, count in counts_array.items()}

    # Extract correlations for both +phi and -phi
    correlations_pos, correlations_neg = extract_correlations_from_counts(counts, num_shots)

    # Calculate results
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
    result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

    print(f"\n  Results for φ = +30°:")
    print(f"    Correlations (exp): {result_pos['correlations']}")
    print(f"    L₃ (exp):      {result_pos['L3']:.4f}")
    print(f"    L₃ (theory):   {result_pos['L3_theory']:.4f}")
    print(f"    Violated:      {result_pos['violated']}")

    print(f"\n  Results for φ = -30°:")
    print(f"    Correlations (exp): {result_neg['correlations']}")
    print(f"    L₃ (exp):      {result_neg['L3']:.4f}")
    print(f"    L₃ (theory):   {result_neg['L3_theory']:.4f}")
    print(f"    Violated:      {result_neg['violated']}")

    # Save results
    results = [
        {
            'phi_deg': -30,
            'phi_rad': -phi_rad,
            **result_neg,
            'job_id': job_id,
            'num_shots': num_shots,
            'unique_bitstrings': len(counts),
            'timestamp': datetime.now().isoformat()
        },
        {
            'phi_deg': 30,
            'phi_rad': phi_rad,
            **result_pos,
            'job_id': job_id,
            'num_shots': num_shots,
            'unique_bitstrings': len(counts),
            'timestamp': datetime.now().isoformat()
        }
    ]

    output_file = f"leggett_rotated_30deg_rerun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Saved to {output_file}")
    print(f"  Job ID: {job_id}")

    return results


if __name__ == "__main__":
    run_30deg_rotated()
