#!/usr/bin/env python3
"""
Test with blacklisting bad physical qubits to force transpiler to use better ones.
"""

import numpy as np
import json
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle_six, extract_correlations_from_counts_six, calc_leggett_for_angle

def test_with_blacklist(phi_deg=15, num_shots=5000, blacklist_qubits=None):
    """
    Test with blacklisted physical qubits.

    Args:
        blacklist_qubits: List of physical qubit indices to avoid (e.g., [23, 16])
    """
    phi_rad = np.radians(phi_deg)
    qc = create_leggett_circuit_for_angle_six(phi_rad)

    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)

    if blacklist_qubits:
        print(f"\n=== BLACKLISTING QUBITS: {blacklist_qubits} ===")

        # Get backend coupling map and remove blacklisted qubits
        original_coupling = backend.coupling_map
        original_edges = original_coupling.get_edges()

        # Filter out edges involving blacklisted qubits
        filtered_edges = [
            edge for edge in original_edges
            if edge[0] not in blacklist_qubits and edge[1] not in blacklist_qubits
        ]

        # Create new coupling map without blacklisted qubits
        coupling_map = CouplingMap(filtered_edges)

        print(f"Removed {len(original_edges) - len(filtered_edges)} edges involving blacklisted qubits")
    else:
        print(f"\n=== NO BLACKLIST (default) ===")
        coupling_map = backend.coupling_map

    # Transpile with the coupling map constraint
    qc_transpiled = transpile(
        qc,
        backend=backend,
        optimization_level=2,
        coupling_map=coupling_map
    )

    # Print actual layout
    layout = qc_transpiled.layout
    print(f"\nPhysical qubit mapping:")

    corr_names = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]
    qubit_pairs = []

    for logical_idx in range(qc.num_qubits):
        physical_qubit = layout.initial_layout._v2p[qc.qubits[logical_idx]]
        pair_idx = logical_idx // 2
        role = 'Alice' if logical_idx % 2 == 0 else 'Bob'

        if logical_idx % 2 == 0:
            qubit_pairs.append([physical_qubit])
        else:
            qubit_pairs[pair_idx].append(physical_qubit)

        print(f"  Logical {logical_idx} (pair {pair_idx}, {role:5s}, {corr_names[pair_idx]:12s}) -> Physical qubit {physical_qubit}")

    # Check if any blacklisted qubits were used
    if blacklist_qubits:
        used_blacklisted = []
        for pair_idx, (q1, q2) in enumerate(qubit_pairs):
            if q1 in blacklist_qubits or q2 in blacklist_qubits:
                used_blacklisted.append((corr_names[pair_idx], q1, q2))

        if used_blacklisted:
            print(f"\n  WARNING: Blacklisted qubits still used:")
            for name, q1, q2 in used_blacklisted:
                print(f"    {name}: qubits {q1}, {q2}")
        else:
            print(f"\n  ✓ Successfully avoided all blacklisted qubits!")

    # Run simulation
    print(f"\nRunning simulation with noise model...")
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc_transpiled, shots=num_shots).result()
    counts = result.get_counts()

    # Extract correlations
    correlations = extract_correlations_from_counts_six(counts, num_shots)
    result_data = calc_leggett_for_angle(correlations, phi_rad)

    print(f"\nResults for φ = {phi_deg}°:")
    for i, (name, corr_exp, corr_th) in enumerate(zip(corr_names, result_data['correlations'], result_data['correlations_theory'])):
        diff = corr_exp - corr_th
        qubits_str = f"(qubits {qubit_pairs[i][0]},{qubit_pairs[i][1]})"
        print(f"  {name:12s} {qubits_str:16s}: exp={corr_exp:7.4f}, theory={corr_th:7.4f}, diff={diff:+7.4f}")

    print(f"\n  L₃ (exp):    {result_data['L3']:.4f}")
    print(f"  L₃ (theory): {result_data['L3_theory']:.4f}")

    # Format result for saving
    result_entry = {
        'phi_deg': phi_deg,
        'phi_rad': phi_rad,
        **result_data,
        'qubit_pairs': qubit_pairs,
    }

    return result_entry

if __name__ == "__main__":
    print("="*70)
    print("TESTING WITH BLACKLISTED BAD QUBITS")
    print("="*70)

    # Test 1: No blacklist (baseline)
    print("\n" + "="*70)
    print("TEST 1: NO BLACKLIST (baseline)")
    print("="*70)
    result_baseline = test_with_blacklist(phi_deg=15, num_shots=5000, blacklist_qubits=None)

    # Test 2: Blacklist confirmed bad pairs: (23,16) from swap test + (0,1) + (9,8) consistently bad
    print("\n" + "="*70)
    print("TEST 2: BLACKLIST CONFIRMED BAD QUBITS: 0, 1, 9, 8, 23, 16")
    print("="*70)
    result_blacklist = test_with_blacklist(phi_deg=15, num_shots=5000, blacklist_qubits=[0, 1, 9, 8, 23, 16])

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    corr_names = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]
    print(f"\n{'Correlation':12s} {'Baseline err':>14s} {'Blacklist err':>15s} {'Improvement':>12s}")
    print("-"*70)

    for i, name in enumerate(corr_names):
        err_baseline = abs(result_baseline['correlations'][i] - result_baseline['correlations_theory'][i])
        err_blacklist = abs(result_blacklist['correlations'][i] - result_blacklist['correlations_theory'][i])
        improvement = err_baseline - err_blacklist

        marker = " ✓ BETTER" if improvement > 0.01 else ""
        marker += " ✗ WORSE" if improvement < -0.01 else ""

        print(f"{name:12s} {err_baseline:+14.4f} {err_blacklist:+15.4f} {improvement:+12.4f}{marker}")

    # Save and plot
    print("\n" + "="*70)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("="*70)

    import os
    os.chdir('/Users/adc/qsim/relative_phase_variant/geometry/simulator_gnuradio/production')

    with open('test_blacklist_baseline.json', 'w') as f:
        json.dump([result_baseline], f, indent=2, default=str)
    print("Saved: test_blacklist_baseline.json")

    with open('test_blacklist_filtered.json', 'w') as f:
        json.dump([result_blacklist], f, indent=2, default=str)
    print("Saved: test_blacklist_filtered.json")

    import subprocess
    print("\nGenerating plots...")
    subprocess.run(['python3', 'plot_correlation_radar_individual.py', 'test_blacklist_baseline.json', 'test_blacklist_baseline'])
    subprocess.run(['python3', 'plot_correlation_radar_individual.py', 'test_blacklist_filtered.json', 'test_blacklist_filtered'])

    print("\n" + "="*70)
    print("DONE! Compare:")
    print("  - test_blacklist_baseline_phi_15.png (with bad qubits)")
    print("  - test_blacklist_filtered_phi_15.png (bad qubits avoided)")
    print("="*70)
