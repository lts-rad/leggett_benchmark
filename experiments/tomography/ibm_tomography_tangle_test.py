import os
#!/usr/bin/env python3
"""
Two-Qubit State Tomography Test using IBM Quantum

Performs 3x3 Pauli basis measurements (XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ)
to reconstruct the density matrix and calculate tangle/concurrence.

Based on:
- James et al., Phys. Rev. A 64, 052312 (2001) for tangle calculation
- Branciard arXiv:1305.4671v2 for visibility thresholds

This version runs all 9 tomography measurements in a single circuit using
9 independent singlet pairs = 18 qubits total.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from tomography import (
    create_tomography_circuit_9pairs,
    extract_expectation_values,
    extract_single_qubit_expectations,
    analyze_tomography_results,
    get_theoretical_singlet_expectations,
    reconstruct_density_matrix,
    calculate_tangle,
    calculate_purity,
    calculate_fidelity_singlet
)


def run_tomography(num_shots, use_ibm, backend_name, dry_run=False,
                   enable_error_mitigation=True, use_noise_model=False):
    """
    Run state tomography for the singlet state.

    Args:
        num_shots: Number of measurements
        use_ibm: If True, run on IBM Quantum
        backend_name: Name of IBM backend (e.g., 'ibm_pittsburgh')
        dry_run: If True, only transpile without submitting
        enable_error_mitigation: If True, enable error mitigation
        use_noise_model: If True, simulate with IBM backend's noise model

    Returns:
        Dictionary with results including density matrix, tangle, concurrence, etc.
    """
    print(f"\n{'='*70}")
    print(f"Running State Tomography ({num_shots} shots, 18 qubits)")
    print(f"{'='*70}")

    # Create tomography circuit with 9 singlet pairs
    qc = create_tomography_circuit_9pairs()

    print(f"  Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    # Run on IBM or simulator
    if use_ibm:
        # Connect to IBM Quantum
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

        print(f"  Transpiling for {backend_name}...")
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

        if dry_run:
            print(f"  DRY RUN: Would submit job here. Exiting.")
            return None

        # Configure error mitigation
        options = SamplerOptions()
        if enable_error_mitigation:
            options.execution.init_qubits = True
            options.dynamical_decoupling.enable = True
            options.dynamical_decoupling.sequence_type = "XX"
            options.twirling.enable_gates = True
            options.twirling.enable_measure = True
            print(f"  Error mitigation: twirling + dynamical decoupling enabled")
        else:
            print(f"  Error mitigation: disabled")

        # Run with Sampler
        print(f"  Submitting job...")
        sampler = Sampler(backend, options=options)
        job = sampler.run([qc_transpiled], shots=num_shots)
        job_id = job.job_id()
        print(f"  Job ID: {job_id}")
        print(f"  Waiting for results...")
        print(f"  Check status: https://quantum.ibm.com/jobs/{job_id}")

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
        pub_result = result[0]
        counts_array = pub_result.data.meas.get_counts()

        counts = {}
        for bitstring, count in counts_array.items():
            counts[bitstring] = count

    else:
        if use_noise_model:
            print(f"  Running on local simulator with {backend_name} noise model...")
            service = QiskitRuntimeService()
            backend = service.backend(backend_name)

            from qiskit_aer.noise import NoiseModel
            noise_model = NoiseModel.from_backend(backend)

            print(f"  Transpiling for noise model...")
            qc_transpiled = transpile(qc, backend=backend, optimization_level=3)

            simulator = AerSimulator(noise_model=noise_model)
            result = simulator.run(qc_transpiled, shots=num_shots).result()
            counts = result.get_counts()
            job_id = f"noise_model_{backend_name}"
        else:
            print(f"  Running on local noiseless simulator...")
            simulator = AerSimulator(method='matrix_product_state')
            result = simulator.run(qc, shots=num_shots).result()
            counts = result.get_counts()
            job_id = "local_simulator_noiseless"

    # Extract expectation values
    two_qubit_exp = extract_expectation_values(counts, num_shots)
    single_exp = extract_single_qubit_expectations(counts, num_shots)

    # Analyze results
    analysis = analyze_tomography_results(two_qubit_exp, single_exp, verbose=True)

    # Build result dictionary
    result_dict = {
        'job_id': job_id,
        'num_shots': num_shots,
        'unique_bitstrings': len(counts),
        'timestamp': datetime.now().isoformat(),
        'tangle': float(analysis['tangle']),
        'concurrence': float(analysis['concurrence']),
        'purity': float(analysis['purity']),
        'fidelity': float(analysis['fidelity']),
        'visibility_estimate': float(analysis['visibility_estimate']),
        'expectations': {k: float(v) for k, v in two_qubit_exp.items()},
        'single_expectations': {k: float(v) for k, v in single_exp.items()},
        'density_matrix_real': analysis['density_matrix'].real.tolist(),
        'density_matrix_imag': analysis['density_matrix'].imag.tolist(),
        'counts': counts
    }

    return result_dict


def main():
    import sys

    # Parse command line arguments
    use_ibm = '--ibm' in sys.argv
    dry_run = '--dry-run' in sys.argv
    use_noise_model = '--noise-model' in sys.argv
    backend_name = "ibm_pittsburgh"
    num_shots = 4096
    output_file = None
    enable_error_mitigation = '--no-error-mitigation' not in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    # Set default output file
    if output_file is None:
        if use_ibm:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'tomography_results_ibm_{backend_name}.json')
        elif use_noise_model:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'tomography_results_ibm_{backend_name}_NOISE_MODEL.json')
        else:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'tomography_results_SIM_noiseless.json')

    print("=" * 70)
    print("TWO-QUBIT STATE TOMOGRAPHY: Tangle Measurement (IBM Quantum)")
    print("=" * 70)
    print(f"\nConfiguration:")
    if use_ibm:
        print(f"  Backend: IBM Quantum - {backend_name} (REAL HARDWARE)")
    elif use_noise_model:
        print(f"  Backend: Local simulator with {backend_name} noise model")
    else:
        print(f"  Backend: Local noiseless simulator")
    print(f"  Shots: {num_shots}")
    print(f"  Circuit size: 18 qubits (9 singlet pairs for 3x3 tomography)")
    print(f"  Error mitigation: {'enabled' if enable_error_mitigation else 'disabled'}")
    print(f"  Dry run: {dry_run}")
    print(f"  Output file: {output_file}")

    print("\n" + "-" * 70)
    print("THEORETICAL BACKGROUND")
    print("-" * 70)
    print("""
    Tangle (entanglement measure) calculation from state tomography:

    1. Measure all 9 Pauli basis combinations: XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
    2. Reconstruct density matrix: rho = (1/4) sum_ij r_ij (sigma_i x sigma_j)
    3. Calculate concurrence via spin-flip operation:
       R = rho * (Y x Y) * rho^T * (Y x Y)
       C = max(0, sqrt(r1) - sqrt(r2) - sqrt(r3) - sqrt(r4))
    4. Tangle T = C^2

    For singlet state |Psi^-⟩ = (|01⟩ - |10⟩)/sqrt(2):
      - <XX> = <YY> = <ZZ> = -1 (perfect anticorrelation)
      - All other correlations = 0
      - Tangle = Concurrence = 1 (maximally entangled)

    Key visibility thresholds (Branciard arXiv:1305.4671v2):
      - V > 1/3 (0.333): Entangled
      - V > 1/sqrt(2) (0.707): Violates CHSH Bell inequality
      - V <= (1+1/sqrt(2))/2 (0.854): Leggett model CAN reproduce
      - V > sqrt(3)/2 (0.866): Solidly violates Leggett inequality
      - 0.854 < V <= 0.866: Unknown region
    """)

    if dry_run and use_ibm:
        print("\n*** DRY RUN MODE: Will transpile circuit and exit ***")
        run_tomography(num_shots, use_ibm, backend_name, dry_run=True,
                      enable_error_mitigation=enable_error_mitigation)
        return

    # Run tomography
    result = run_tomography(num_shots, use_ibm, backend_name,
                           enable_error_mitigation=enable_error_mitigation,
                           use_noise_model=use_noise_model)

    if result is None:
        print("Tomography failed!")
        return

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Concurrence: {result['concurrence']:.6f}")
    print(f"  Tangle:      {result['tangle']:.6f}")
    print(f"  Purity:      {result['purity']:.6f}")
    print(f"  Fidelity:    {result['fidelity']:.6f}")

    V = result['visibility_estimate']
    print(f"\n  Estimated visibility: {V:.4f}")

    # Status checks
    V_ENTANGLED = 1/3
    V_CHSH = 1/np.sqrt(2)
    V_LEGGETT_COMPAT = (1 + 1/np.sqrt(2)) / 2  # ~0.854: Leggett model can still reproduce
    V_LEGGETT_VIOLATE = np.sqrt(3) / 2         # ~0.866: Solidly violates Leggett

    print(f"\n  Status:")
    print(f"    Entangled (V > {V_ENTANGLED:.3f}):              {'YES' if V > V_ENTANGLED else 'NO'}")
    print(f"    CHSH violation (V > {V_CHSH:.3f}):         {'YES' if V > V_CHSH else 'NO'}")
    print(f"    Leggett compatible (V <= {V_LEGGETT_COMPAT:.3f}): {'YES' if V <= V_LEGGETT_COMPAT else 'NO'}")
    print(f"    Leggett violation (V > {V_LEGGETT_VIOLATE:.3f}):  {'YES' if V > V_LEGGETT_VIOLATE else 'NO'}")
    if V_LEGGETT_COMPAT < V <= V_LEGGETT_VIOLATE:
        print(f"    ** In unknown region (0.854 < V <= 0.866) **")

    # Save results
    print(f"\nSaving results to {output_file}...")

    # Remove counts from saved file if too large
    result_to_save = result.copy()
    if len(result['counts']) > 1000:
        result_to_save['counts'] = f"<{len(result['counts'])} unique bitstrings>"

    with open(output_file, 'w') as f:
        json.dump(result_to_save, f, indent=2, default=str)

    print("Done!")

    return result


if __name__ == "__main__":
    main()
