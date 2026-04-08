import os
#!/usr/bin/env python3
"""
Leggett Inequality Test using IBM Quantum - Sequential Version with 24 Qubits
Using MAPOMATIC for automatic qubit selection based on current calibration.

Based on arXiv:0801.2241v2

This version uses mapomatic to find the best qubit layout at runtime,
using fresh calibration data from the backend.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions

try:
    import mapomatic as mm
    MAPOMATIC_AVAILABLE = True
except ImportError:
    MAPOMATIC_AVAILABLE = False
    print("WARNING: mapomatic not installed. Install with: pip install mapomatic")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle, extract_correlations_from_counts, calc_leggett_for_angle


def get_top_pairs_from_visibility(backend, top_n=50):
    """
    Get top N qubit pairs based on visibility prediction.
    Uses the same logic as predict_visibility.py.
    """
    from qiskit.transpiler import CouplingMap

    target = backend.target
    num_qubits = backend.num_qubits

    # Get coupling map
    cm = backend.coupling_map
    edges = list(cm.get_edges())

    # Get qubit errors
    qubit_errors = {}
    for q in range(num_qubits):
        qubit_errors[q] = {'readout_error': 0.01, 'sx_error': 0.001}
        try:
            if 'measure' in target.operation_names:
                meas_props = target['measure'].get((q,))
                if meas_props and meas_props.error is not None:
                    qubit_errors[q]['readout_error'] = meas_props.error
            if 'sx' in target.operation_names:
                sx_props = target['sx'].get((q,))
                if sx_props and sx_props.error is not None:
                    qubit_errors[q]['sx_error'] = sx_props.error
        except:
            pass

    # Get 2Q gate errors (check for cz, ecr, or cx)
    if 'cz' in target.operation_names:
        two_q_gate = 'cz'
    elif 'ecr' in target.operation_names:
        two_q_gate = 'ecr'
    else:
        two_q_gate = 'cx'
    edge_errors = {}
    try:
        gate_ops = target[two_q_gate]
        for qargs in gate_ops:
            if qargs and len(qargs) == 2:
                props = gate_ops[qargs]
                error = props.error if props and props.error is not None else 0.01
                edge_errors[qargs] = error
    except:
        pass

    # Calculate visibility for each edge
    # Using worst-case gate counts from Leggett circuit analysis (a1_b1p basis)
    n_1q, n_2q = 12, 1  # Worst case: 12 1Q gates, 1 2Q gate

    predictions = []
    edges_set = set(tuple(sorted(e)) for e in edges)

    for edge in edges_set:
        q1, q2 = edge
        ro1 = qubit_errors.get(q1, {}).get('readout_error', 0.01)
        ro2 = qubit_errors.get(q2, {}).get('readout_error', 0.01)
        sx1 = qubit_errors.get(q1, {}).get('sx_error', 0.001)
        sx2 = qubit_errors.get(q2, {}).get('sx_error', 0.001)

        twoq_err = edge_errors.get((q1, q2)) or edge_errors.get((q2, q1)) or 0.01

        readout_fid = (1 - ro1) * (1 - ro2)
        gate_fid = ((1 - (sx1 + sx2)/2) ** n_1q) * ((1 - twoq_err) ** n_2q)
        visibility = readout_fid * gate_fid

        predictions.append({
            'edge': (q1, q2),
            'visibility': visibility,
            'twoq_error': twoq_err
        })

    # Sort by visibility and return top N
    predictions.sort(key=lambda x: x['visibility'], reverse=True)
    top_edges = [p['edge'] for p in predictions[:top_n]]

    return top_edges, predictions[:top_n]


def deflate_circuit_safe(qc):
    """
    Safely deflate a circuit to only active qubits.
    Works around mapomatic's u3 gate compatibility issue with newer Qiskit.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    # Find which qubits are actually used
    used_qubits = set()
    for instruction in qc.data:
        for qubit in instruction.qubits:
            used_qubits.add(qc.find_bit(qubit).index)

    if len(used_qubits) == qc.num_qubits:
        # All qubits used, no deflation needed
        return qc

    # Create mapping from old indices to new indices
    used_sorted = sorted(used_qubits)
    qubit_map = {old: new for new, old in enumerate(used_sorted)}

    # Create new circuit with only used qubits
    new_qr = QuantumRegister(len(used_qubits), 'q')
    new_cr = ClassicalRegister(len(used_qubits), 'c') if qc.num_clbits > 0 else None

    if new_cr:
        new_qc = QuantumCircuit(new_qr, new_cr)
    else:
        new_qc = QuantumCircuit(new_qr)

    # Copy instructions with remapped qubits
    for instruction in qc.data:
        old_qubits = [qc.find_bit(q).index for q in instruction.qubits]
        new_qubits = [new_qr[qubit_map[q]] for q in old_qubits]

        if instruction.clbits:
            old_clbits = [qc.find_bit(c).index for c in instruction.clbits]
            new_clbits = [new_cr[qubit_map.get(c, c)] for c in old_clbits] if new_cr else []
            new_qc.append(instruction.operation, new_qubits, new_clbits)
        else:
            new_qc.append(instruction.operation, new_qubits)

    return new_qc


def find_best_layout_mapomatic(qc, backend, verbose=True, top_n_pairs=50, call_limit=10000):
    """
    Use mapomatic to find the best qubit layout based on current calibration.

    Uses a RESTRICTED coupling map with only the top N pairs by visibility,
    making the search much faster and more targeted.

    Args:
        qc: QuantumCircuit to map
        backend: IBM backend
        verbose: Print details
        top_n_pairs: Number of top pairs to consider (default 50)
        call_limit: Max VF2 mapper calls (default 10000)

    Returns:
        best_layout: List of physical qubits
        score: Error score (lower is better)
    """
    if not MAPOMATIC_AVAILABLE:
        raise ImportError("mapomatic is required. Install with: pip install mapomatic")

    from qiskit.transpiler import CouplingMap

    if verbose:
        print(f"  Finding best layout with mapomatic (top {top_n_pairs} pairs, call_limit={call_limit})...")

    # Get top pairs by visibility
    top_edges, top_predictions = get_top_pairs_from_visibility(backend, top_n_pairs)

    if verbose:
        print(f"  Top pair visibility range: {top_predictions[0]['visibility']:.4f} - {top_predictions[-1]['visibility']:.4f}")

    # Create restricted coupling map with only top pairs (bidirectional)
    restricted_edges = []
    for q1, q2 in top_edges:
        restricted_edges.append([q1, q2])
        restricted_edges.append([q2, q1])

    restricted_cmap = CouplingMap(restricted_edges)

    if verbose:
        print(f"  Restricted coupling map: {len(top_edges)} pairs ({len(restricted_edges)} directed edges)")

    # Transpile to the restricted coupling map first
    trans_qc = transpile(qc, coupling_map=restricted_cmap, optimization_level=3)

    if verbose:
        print(f"  Transpiled to restricted map: {trans_qc.num_qubits} qubits, depth {trans_qc.depth()}")

    # Deflate to only active qubits (using safe method to avoid u2/u3 compatibility issue)
    try:
        small_qc = mm.deflate_circuit(trans_qc)
    except AttributeError as e:
        # mapomatic uses legacy gate names (u2, u3) that don't exist in newer Qiskit
        if verbose:
            print(f"  Using safe deflation (mapomatic compatibility workaround: {e})")
        small_qc = deflate_circuit_safe(trans_qc)

    if verbose:
        print(f"  Deflated circuit: {small_qc.num_qubits} active qubits")

    # Find matching layouts on the restricted map with call limit
    layouts = mm.matching_layouts(small_qc, restricted_cmap, call_limit=call_limit)

    if verbose:
        print(f"  Found {len(layouts)} matching layouts")

    if not layouts:
        print("  WARNING: No layouts found with restricted map, falling back to full backend")
        trans_qc = transpile(qc, backend, optimization_level=3)
        try:
            small_qc = mm.deflate_circuit(trans_qc)
        except AttributeError as e:
            if verbose:
                print(f"  Using safe deflation for fallback: {e}")
            small_qc = deflate_circuit_safe(trans_qc)
        layouts = mm.matching_layouts(small_qc, backend, call_limit=call_limit)
        if not layouts:
            raise ValueError("No matching layouts found!")

    # Evaluate layouts based on error rates
    try:
        scores = mm.evaluate_layouts(small_qc, layouts, backend)
    except Exception as e:
        if verbose:
            print(f"  WARNING: mapomatic evaluate_layouts failed: {e}")
            print(f"  Falling back to manual visibility-based scoring...")
        scores = []

    if not scores:
        # Mapomatic evaluation failed, use our own visibility-based scoring
        if verbose:
            print(f"  Using visibility-based layout scoring (mapomatic evaluation unavailable)")

        # Score each layout based on visibility of the pairs used
        visibility_map = {p['edge']: p['visibility'] for p in top_predictions}
        scored_layouts = []
        for layout in layouts[:1000]:  # Limit to first 1000 for speed
            layout_list = list(layout)
            # Calculate average visibility for pairs in this layout
            total_vis = 0
            pair_count = 0
            for i in range(0, len(layout_list), 2):
                if i + 1 < len(layout_list):
                    q1, q2 = layout_list[i], layout_list[i+1]
                    edge = tuple(sorted([q1, q2]))
                    vis = visibility_map.get(edge, 0.95)
                    total_vis += vis
                    pair_count += 1
            avg_vis = total_vis / pair_count if pair_count > 0 else 0
            # Lower score is better for compatibility, so use 1 - visibility
            scored_layouts.append((layout, 1 - avg_vis))

        # Sort by score (lower is better)
        scored_layouts.sort(key=lambda x: x[1])
        scores = scored_layouts

    # scores is list of (layout, score) tuples, sorted by score (best first)
    best_layout, best_score = scores[0]

    if verbose:
        print(f"  Best layout score: {best_score:.6f}")
        print(f"  Best layout: {list(best_layout)}")

        # Show top 5
        print(f"\n  Top 5 layouts:")
        for i, (layout, score) in enumerate(scores[:5]):
            print(f"    {i+1}. Score={score:.6f} Layout={list(layout)[:6]}...")

    return list(best_layout), best_score, scores


def run_single_angle(phi_deg, num_shots, use_ibm, backend_name, dry_run=False,
                     enable_error_mitigation=True, use_noise_model=False,
                     use_mapomatic=True, cached_layout=None):
    """
    Run Leggett test for a single angle φ.

    Args:
        phi_deg: Angle in degrees
        num_shots: Number of measurements
        use_ibm: If True, run on IBM Quantum
        backend_name: Name of IBM backend
        dry_run: If True, only transpile without submitting
        enable_error_mitigation: If True, enable error mitigation
        use_noise_model: If True, simulate with noise model
        use_mapomatic: If True, use mapomatic for layout selection
        cached_layout: Pre-computed layout to reuse (optional)

    Returns:
        Dictionary with results
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running complementary angles φ = ±{abs(phi_deg):.1f}° ({num_shots} shots, 24 qubits)")
    print(f"{'='*70}")

    # Create circuit
    qc = create_leggett_circuit_for_angle(phi_rad)

    if use_ibm or use_noise_model:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

        # Find best layout with mapomatic (or use cached)
        if use_mapomatic and MAPOMATIC_AVAILABLE:
            if cached_layout is not None:
                best_layout = cached_layout
                print(f"  Using cached mapomatic layout: {best_layout[:6]}...")
            else:
                best_layout, score, all_scores = find_best_layout_mapomatic(qc, backend)
        else:
            # Fallback to default transpilation
            best_layout = None
            print(f"  Using default transpiler layout (mapomatic disabled)")

        print(f"  Transpiling for {backend_name}...")
        print(f"  Original circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

        if best_layout:
            qc_transpiled = transpile(qc, backend=backend, optimization_level=3,
                                     initial_layout=best_layout)
        else:
            qc_transpiled = transpile(qc, backend=backend, optimization_level=3)

        print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")

        # Log physical qubit mapping
        layout = qc_transpiled.layout
        print(f"  Physical qubit mapping:")
        pairs_used = []
        for logical_idx in range(qc.num_qubits):
            physical_qubit = layout.initial_layout._v2p[qc.qubits[logical_idx]]
            pair_idx = logical_idx // 2
            role = 'Alice' if logical_idx % 2 == 0 else 'Bob'
            if logical_idx % 2 == 1:
                alice_q = layout.initial_layout._v2p[qc.qubits[logical_idx-1]]
                pairs_used.append((alice_q, physical_qubit))
            if logical_idx < 6:  # Only show first 3 pairs
                print(f"    Logical {logical_idx} (pair {pair_idx}, {role}) -> Physical qubit {physical_qubit}")
        print(f"    ...")
        print(f"  Pairs used: {pairs_used}")

    if use_ibm:
        if dry_run:
            print(f"  DRY RUN: Would submit job here. Exiting.")
            return {'layout': best_layout}

        # Configure error mitigation
        options = SamplerOptions()
        if enable_error_mitigation:
            options.execution.init_qubits = True
            options.dynamical_decoupling.enable = True
            options.dynamical_decoupling.sequence_type = "XY4"
            options.twirling.enable_gates = True
            options.twirling.enable_measure = True
            print(f"  Error mitigation: twirling + dynamical decoupling (XY4) enabled")
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
            print(f"  Job {status}! Skipping this angle.")
            return None

        result = job.result()
        pub_result = result[0]
        counts_array = pub_result.data.meas.get_counts()

        counts = {}
        for bitstring, count in counts_array.items():
            counts[bitstring] = count

    elif use_noise_model:
        from qiskit_aer.noise import NoiseModel
        noise_model = NoiseModel.from_backend(backend)

        print(f"  Running on AerSimulator with noise model...")
        simulator = AerSimulator(noise_model=noise_model)
        result = simulator.run(qc_transpiled, shots=num_shots).result()
        counts = result.get_counts()
        job_id = f"noise_model_{backend_name}_mapomatic"

    else:
        print(f"  Running on local noiseless simulator...")
        simulator = AerSimulator(method='matrix_product_state')
        result = simulator.run(qc, shots=num_shots).result()
        counts = result.get_counts()
        job_id = "local_simulator_noiseless"
        best_layout = None

    # Extract correlations
    correlations_pos, correlations_neg = extract_correlations_from_counts(counts, num_shots)

    # Calculate results
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
    result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

    print(f"\n  Results for φ = +{abs(phi_deg):.1f}°:")
    print(f"    L₃ (exp):      {result_pos['L3']:.4f}")
    print(f"    L₃ (theory):   {result_pos['L3_theory']:.4f}")
    print(f"    L₃ bound:      {result_pos['bound']:.4f}")
    print(f"    Violated:      {result_pos['violated']}")

    print(f"\n  Results for φ = -{abs(phi_deg):.1f}°:")
    print(f"    L₃ (exp):      {result_neg['L3']:.4f}")
    print(f"    L₃ (theory):   {result_neg['L3_theory']:.4f}")
    print(f"    Violated:      {result_neg['violated']}")

    return {
        'positive': {
            'phi_deg': abs(phi_deg),
            'phi_rad': phi_rad,
            **result_pos,
            'job_id': job_id,
            'num_shots': num_shots,
            'layout': best_layout,
            'timestamp': datetime.now().isoformat()
        },
        'negative': {
            'phi_deg': -abs(phi_deg),
            'phi_rad': -phi_rad,
            **result_neg,
            'job_id': job_id,
            'num_shots': num_shots,
            'layout': best_layout,
            'timestamp': datetime.now().isoformat()
        },
        'counts': counts,
        'layout': best_layout
    }


def main():
    import sys

    use_ibm = '--ibm' in sys.argv
    dry_run = '--dry-run' in sys.argv
    use_noise_model = '--noise-model' in sys.argv
    no_mapomatic = '--no-mapomatic' in sys.argv
    backend_name = "ibm_pittsburgh"
    num_shots = 10000
    output_file = None
    enable_error_mitigation = '--no-error-mitigation' not in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    use_mapomatic = MAPOMATIC_AVAILABLE and not no_mapomatic

    if output_file is None:
        suffix = "_MAPOMATIC" if use_mapomatic else ""
        if use_ibm:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_{backend_name}_24qb{suffix}.json')
        elif use_noise_model:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_ibm_{backend_name}_NOISE{suffix}.json')
        else:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_SIM_noiseless{suffix}.json')

    print("=" * 70)
    print("LEGGETT INEQUALITY TEST: 24 Qubits with MAPOMATIC Layout Selection")
    print("=" * 70)
    print(f"\nConfiguration:")
    if use_ibm:
        print(f"  Backend: IBM Quantum - {backend_name} (REAL HARDWARE)")
    elif use_noise_model:
        print(f"  Backend: Local simulator with {backend_name} noise model")
    else:
        print(f"  Backend: Local noiseless simulator")
    print(f"  Shots per angle: {num_shots}")
    print(f"  Mapomatic: {'ENABLED' if use_mapomatic else 'DISABLED'}")
    print(f"  Error mitigation: {'enabled' if enable_error_mitigation else 'disabled'}")
    print(f"  Output file: {output_file}")

    if not MAPOMATIC_AVAILABLE and not no_mapomatic:
        print("\n  WARNING: mapomatic not installed!")
        print("  Install with: pip install mapomatic")
        print("  Falling back to default transpiler layout.\n")

    test_angles = [15, 25, 30, 45, 60]

    print(f"\nTest angles: {test_angles}")

    if dry_run:
        print("\n*** DRY RUN MODE ***")
        result = run_single_angle(test_angles[0], num_shots, use_ibm, backend_name,
                                 dry_run=True, use_mapomatic=use_mapomatic)
        if result and 'layout' in result:
            print(f"\nMapomatic selected layout: {result['layout']}")
        return

    # For efficiency, compute layout once and reuse for all angles
    # (calibration doesn't change within a few minutes)
    cached_layout = None

    results = []
    for i, angle in enumerate(test_angles):
        result = run_single_angle(
            angle, num_shots, use_ibm, backend_name,
            enable_error_mitigation=enable_error_mitigation,
            use_noise_model=use_noise_model,
            use_mapomatic=use_mapomatic,
            cached_layout=cached_layout
        )

        if result:
            results.append(result['positive'])
            results.append(result['negative'])

            # Cache the layout for subsequent angles
            if cached_layout is None and 'layout' in result:
                cached_layout = result['layout']
                print(f"\n  Caching layout for remaining angles: {cached_layout[:6]}...")

        if use_ibm and i < len(test_angles) - 1:
            print(f"\nWaiting 2 seconds...")
            time.sleep(2)

    # Summary
    results.sort(key=lambda x: x['phi_deg'])

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if cached_layout:
        print(f"\nLayout used (mapomatic): {cached_layout}")

    violations = sum(1 for r in results if r['violated'])
    print(f"\nTotal angles tested: {len(results)}")
    print(f"Violations: {violations}/{len(results)}")

    print(f"\n{'Angle':>8} {'L₃(exp)':>10} {'L₃(th)':>10} {'Bound':>10} {'Status':>12}")
    print("-" * 60)

    for r in results:
        status = "VIOLATION" if r['violated'] else "No violation"
        print(f"{r['phi_deg']:>+8.1f}° {r['L3']:>10.4f} {r['L3_theory']:>10.4f} {r['bound']:>10.4f} {status:>12}")

    # Save
    print(f"\nSaving results to {output_file}...")
    save_data = {
        'layout': cached_layout,
        'results': results,
        'config': {
            'backend': backend_name,
            'num_shots': num_shots,
            'mapomatic': use_mapomatic,
        }
    }
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print("Done!")


if __name__ == "__main__":
    main()
