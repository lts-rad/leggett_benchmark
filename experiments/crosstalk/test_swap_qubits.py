#!/usr/bin/env python3
"""
Test if C(a2,b2) error is specific to physical qubits or logical measurement.
By swapping which physical qubits are used for pair 2, we can isolate the issue.
"""

import numpy as np
import json
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle_six, extract_correlations_from_counts_six, calc_leggett_for_angle

def test_with_custom_layout(phi_deg=15, num_shots=1000, swap_pair2=False):
    """
    Test with custom qubit layout.

    If swap_pair2=True, use different physical qubits for pair 2 (C(a2,b2))
    """
    phi_rad = np.radians(phi_deg)
    qc = create_leggett_circuit_for_angle_six(phi_rad)

    service = QiskitRuntimeService()
    backend = service.backend('ibm_pittsburgh')
    noise_model = NoiseModel.from_backend(backend)

    if swap_pair2:
        # Custom layout that avoids qubits 23,16 for pair 2
        # Original: pair 2 gets 23,16 which shows error
        # New: swap pair 2 with pair 5 to use qubits 85,84 instead
        initial_layout = [
            2,   # Logical 0 (pair 0, Alice)
            3,   # Logical 1 (pair 0, Bob)
            5,   # Logical 2 (pair 1, Alice)
            4,   # Logical 3 (pair 1, Bob)
            85,  # Logical 4 (pair 2, Alice) - SWAPPED to avoid 23
            84,  # Logical 5 (pair 2, Bob) - SWAPPED to avoid 16
            0,   # Logical 6 (pair 3, Alice)
            1,   # Logical 7 (pair 3, Bob)
            21,  # Logical 8 (pair 4, Alice)
            22,  # Logical 9 (pair 4, Bob)
            23,  # Logical 10 (pair 5, Alice) - SWAPPED to use 23
            16,  # Logical 11 (pair 5, Bob) - SWAPPED to use 16
        ]
        print(f"\n=== SWAPPED LAYOUT ===")
        print(f"Pair 2 (C(a2,b2)) now uses physical qubits 85,84 (was 23,16)")
        print(f"Pair 5 (C(a3,b3')) now uses physical qubits 23,16 (was 85,84)")
    else:
        # Default layout (let transpiler choose)
        initial_layout = None
        print(f"\n=== DEFAULT LAYOUT ===")

    qc_transpiled = transpile(qc, backend=backend, optimization_level=2, initial_layout=initial_layout)

    # Print actual layout
    layout = qc_transpiled.layout
    print(f"\nPhysical qubit mapping:")
    for logical_idx in range(qc.num_qubits):
        physical_qubit = layout.initial_layout._v2p[qc.qubits[logical_idx]]
        pair_idx = logical_idx // 2
        role = 'Alice' if logical_idx % 2 == 0 else 'Bob'

        # Identify which correlation this is
        corr_names = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]
        print(f"  Logical {logical_idx} (pair {pair_idx}, {role}, {corr_names[pair_idx]}) -> Physical qubit {physical_qubit}")

    # Run simulation
    print(f"\nRunning simulation with noise model...")
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc_transpiled, shots=num_shots).result()
    counts = result.get_counts()

    # Extract correlations
    correlations = extract_correlations_from_counts_six(counts, num_shots)
    result_data = calc_leggett_for_angle(correlations, phi_rad)

    print(f"\nResults for φ = {phi_deg}°:")
    corr_names = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]
    for i, (name, corr_exp, corr_th) in enumerate(zip(corr_names, result_data['correlations'], result_data['correlations_theory'])):
        diff = corr_exp - corr_th
        print(f"  {name}: exp={corr_exp:7.4f}, theory={corr_th:7.4f}, diff={diff:+7.4f}")

    print(f"\n  L₃ (exp):    {result_data['L3']:.4f}")
    print(f"  L₃ (theory): {result_data['L3_theory']:.4f}")

    # Format result for saving
    result_entry = {
        'phi_deg': phi_deg,
        'phi_rad': phi_rad,
        **result_data,
    }

    return result_entry

if __name__ == "__main__":
    print("="*70)
    print("TESTING IF C(a2,b2) ERROR IS HARDWARE-SPECIFIC")
    print("="*70)

    # Test 1: Default layout (transpiler chooses)
    print("\n" + "="*70)
    print("TEST 1: DEFAULT LAYOUT (transpiler auto-selects)")
    print("="*70)
    result_default = test_with_custom_layout(phi_deg=15, num_shots=5000, swap_pair2=False)

    # Test 2: Swapped layout (avoid physical qubits 23,16 for pair 2)
    print("\n" + "="*70)
    print("TEST 2: SWAPPED LAYOUT (manually avoid qubits 23,16 for C(a2,b2))")
    print("="*70)
    result_swapped = test_with_custom_layout(phi_deg=15, num_shots=5000, swap_pair2=True)

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    corr_names = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]
    for i, name in enumerate(corr_names):
        diff_default = result_default['correlations'][i] - result_default['correlations_theory'][i]
        diff_swapped = result_swapped['correlations'][i] - result_swapped['correlations_theory'][i]

        marker = " ← ERROR MOVED!" if i == 2 and abs(diff_swapped) < abs(diff_default) else ""
        marker2 = " ← ERROR HERE NOW!" if i == 5 and abs(diff_swapped) > abs(diff_default) else ""

        print(f"{name:12s}: Default err={diff_default:+7.4f}, Swapped err={diff_swapped:+7.4f}{marker}{marker2}")

    print("\nIf error STAYS with C(a2,b2) → Circuit/measurement bug")
    print("If error MOVES to C(a3,b3') → Hardware issue with bad physical qubits")

    # Save results and generate plots
    print("\n" + "="*70)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("="*70)

    # Save JSON files
    import os
    os.chdir('/Users/adc/qsim/relative_phase_variant/geometry/simulator_gnuradio/production')

    with open('test_swap_default.json', 'w') as f:
        json.dump([result_default], f, indent=2, default=str)
    print("Saved: test_swap_default.json")

    with open('test_swap_swapped.json', 'w') as f:
        json.dump([result_swapped], f, indent=2, default=str)
    print("Saved: test_swap_swapped.json")

    # Generate radar plots
    import subprocess

    print("\nGenerating radar plot for DEFAULT layout...")
    subprocess.run(['python3', 'plot_correlation_radar_individual.py', 'test_swap_default.json', 'test_swap_default'])

    print("\nGenerating radar plot for SWAPPED layout...")
    subprocess.run(['python3', 'plot_correlation_radar_individual.py', 'test_swap_swapped.json', 'test_swap_swapped'])

    print("\n" + "="*70)
    print("DONE! Compare these plots:")
    print("  - test_swap_default_phi_15.png")
    print("  - test_swap_swapped_phi_15.png")
    print("="*70)
