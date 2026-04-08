import os
#!/usr/bin/env python3
"""
Leggett Inequality Test using IBM Quantum - SABRE Layout Version

This version explicitly uses SabreLayout to see what qubits it selects,
comparing default vs noise-aware layout strategies.
"""

import numpy as np
import json
import time
from datetime import datetime

from qiskit import transpile
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap, SetLayout, ApplyLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, SamplerOptions
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from leggett import create_leggett_circuit_for_angle, extract_correlations_from_counts, calc_leggett_for_angle


def get_selected_pairs(qc_transpiled):
    """Extract the physical qubit pairs used after transpilation."""
    layout = qc_transpiled.layout
    if layout is None:
        return []

    final_layout = layout.final_index_layout()

    # The circuit uses pairs: (0,1), (2,3), ..., (22,23) for 12 singlet pairs
    pairs = []
    for i in range(0, 24, 2):
        if i < len(final_layout) and i+1 < len(final_layout):
            p1, p2 = final_layout[i], final_layout[i+1]
            pairs.append((min(p1, p2), max(p1, p2)))

    return sorted(pairs)


def transpile_with_strategy(qc, backend, strategy='default'):
    """
    Transpile circuit with different layout strategies.

    Args:
        qc: QuantumCircuit to transpile
        backend: IBM backend
        strategy: 'default', 'sabre', 'noise_aware', or 'vf2'

    Returns:
        Transpiled circuit
    """
    if strategy == 'default':
        # Standard optimization_level=3 (what ibm_leggett_test_sequential.py uses)
        qc_t = transpile(qc, backend=backend, optimization_level=3)

    elif strategy == 'sabre':
        # Explicit SABRE layout without noise awareness
        qc_t = transpile(qc, backend=backend, optimization_level=3,
                        layout_method='sabre', routing_method='sabre')

    elif strategy == 'noise_aware':
        # Use VF2PostLayout for noise-aware optimization
        # This considers error rates when choosing layout
        qc_t = transpile(qc, backend=backend, optimization_level=3,
                        layout_method='sabre', routing_method='sabre')
        # Then apply VF2PostLayout pass for noise-aware refinement
        from qiskit.transpiler.passes import VF2PostLayout
        pm = PassManager([VF2PostLayout(backend.target, strict_direction=False)])
        qc_t = pm.run(qc_t)

    elif strategy == 'trivial':
        # Trivial layout - just maps virtual qubit i to physical qubit i
        qc_t = transpile(qc, backend=backend, optimization_level=3,
                        layout_method='trivial')

    elif strategy == 'dense':
        # Dense layout - finds dense subgraph of coupling map
        qc_t = transpile(qc, backend=backend, optimization_level=3,
                        layout_method='dense')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return qc_t


def compare_layouts(backend_name='ibm_pittsburgh', sabre_seeds=1):
    """Compare different layout strategies.

    Args:
        backend_name: IBM backend name
        sabre_seeds: Number of random seeds to try for SABRE (default 1, use 100 for thorough comparison)
    """

    print("=" * 80)
    print("LAYOUT STRATEGY COMPARISON FOR LEGGETT CIRCUIT")
    print("=" * 80)

    # Connect to IBM
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    print(f"\nBackend: {backend_name} ({backend.num_qubits} qubits)")

    # Create test circuit
    phi_rad = np.radians(30)
    qc = create_leggett_circuit_for_angle(phi_rad)
    print(f"Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    # Load visibility predictions for comparison
    try:
        predictions_file = f'visibility_predictions_{backend_name}.json'
        with open(predictions_file) as f:
            vis_data = json.load(f)
        pair_to_pred = {}
        for p in vis_data['predictions']:
            key = (min(p['q1'], p['q2']), max(p['q1'], p['q2']))
            pair_to_pred[key] = p
        has_vis_data = True
        print(f"Loaded predictions from {predictions_file}")
    except FileNotFoundError:
        print(f"Warning: {predictions_file} not found - run predict_visibility.py --backend {backend_name} first")
        has_vis_data = False
        pair_to_pred = {}

    def calc_adjusted_visibility(pairs, ops, pair_to_pred):
        """Calculate visibility adjusted for actual gate count."""
        # Count actual 2Q gates from transpiled circuit
        n_2q = ops.get('cz', 0) + ops.get('cx', 0) + ops.get('ecr', 0)
        n_2q_per_pair = n_2q / len(pairs) if pairs else 1

        # Baseline assumes 1 CZ per pair
        baseline_2q = 1
        extra_2q = n_2q_per_pair - baseline_2q

        visibilities = []
        for pair in pairs:
            pred = pair_to_pred.get(pair)
            if pred:
                # Adjust for extra gates: V_adj = V_base * (F_2q)^extra_gates
                # Use average 2Q error from predictions
                twoq_fid = 1 - pred.get('twoq_error', 0.01)
                v_base = pred['visibility']
                v_adj = v_base * (twoq_fid ** extra_2q)
                visibilities.append(v_adj)

        return visibilities

    strategies = ['default', 'sabre', 'trivial', 'dense']
    results = {}

    for strategy in strategies:
        print(f"\n{'-'*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'-'*60}")

        try:
            qc_t = transpile_with_strategy(qc, backend, strategy)
            pairs = get_selected_pairs(qc_t)
            ops = dict(qc_t.count_ops())
            n_2q = ops.get('cz', 0) + ops.get('cx', 0) + ops.get('ecr', 0)

            print(f"  Transpiled depth: {qc_t.depth()}")
            print(f"  2Q gates: {n_2q} (baseline: 12)")
            print(f"  Ops: {ops}")
            print(f"\n  Physical qubit pairs selected:")

            for pair in pairs:
                pred = pair_to_pred.get(pair)
                if pred:
                    print(f"    {pair}: V_base = {pred['visibility']:.4f}")
                else:
                    print(f"    {pair}: V = ???")

            visibilities = calc_adjusted_visibility(pairs, ops, pair_to_pred)

            if visibilities:
                mean_vis = np.mean(visibilities)
                min_vis = min(visibilities)
                print(f"\n  Adjusted for {n_2q} 2Q gates:")
                print(f"  Mean visibility: {mean_vis:.4f}")
                print(f"  Min visibility:  {min_vis:.4f}")
                results[strategy] = {
                    'pairs': pairs,
                    'mean_vis': mean_vis,
                    'min_vis': min_vis,
                    'depth': qc_t.depth(),
                    'n_2q': n_2q
                }
            else:
                results[strategy] = {
                    'pairs': pairs,
                    'depth': qc_t.depth(),
                    'n_2q': n_2q
                }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[strategy] = {'error': str(e)}

    # SABRE with multiple seeds
    if sabre_seeds > 1:
        print(f"\n{'='*80}")
        print(f"SABRE WITH {sabre_seeds} RANDOM SEEDS")
        print(f"{'='*80}")

        sabre_results = []
        for seed in range(sabre_seeds):
            try:
                qc_t = transpile(qc, backend=backend, optimization_level=3,
                                layout_method='sabre', routing_method='sabre',
                                seed_transpiler=seed)
                pairs = get_selected_pairs(qc_t)
                ops = dict(qc_t.count_ops())
                n_2q = ops.get('cz', 0) + ops.get('cx', 0) + ops.get('ecr', 0)

                visibilities = calc_adjusted_visibility(pairs, ops, pair_to_pred)
                if visibilities:
                    sabre_results.append({
                        'seed': seed,
                        'pairs': pairs,
                        'mean_vis': np.mean(visibilities),
                        'min_vis': min(visibilities),
                        'depth': qc_t.depth(),
                        'n_2q': n_2q
                    })
            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")

        # Sort by mean visibility
        sabre_results.sort(key=lambda x: x['mean_vis'], reverse=True)

        print(f"\n{'Seed':>6} {'Mean V':>10} {'Min V':>10} {'2Q Gates':>10} {'Depth':>8}")
        print("-" * 50)

        # Show top 10 and bottom 5
        for r in sabre_results[:10]:
            print(f"{r['seed']:>6} {r['mean_vis']:>10.4f} {r['min_vis']:>10.4f} {r['n_2q']:>10} {r['depth']:>8}")

        if len(sabre_results) > 15:
            print("  ...")
            for r in sabre_results[-5:]:
                print(f"{r['seed']:>6} {r['mean_vis']:>10.4f} {r['min_vis']:>10.4f} {r['n_2q']:>10} {r['depth']:>8}")

        print(f"\nSABRE statistics over {len(sabre_results)} seeds:")
        mean_vals = [r['mean_vis'] for r in sabre_results]
        print(f"  Best mean V:  {max(mean_vals):.4f} (seed {sabre_results[0]['seed']})")
        print(f"  Worst mean V: {min(mean_vals):.4f} (seed {sabre_results[-1]['seed']})")
        print(f"  Avg mean V:   {np.mean(mean_vals):.4f} ± {np.std(mean_vals):.4f}")

        # Add best SABRE to results
        results['sabre_best'] = sabre_results[0]
        results['sabre_worst'] = sabre_results[-1]

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY (visibility adjusted for actual gate count)")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<15} {'Mean V':>10} {'Min V':>10} {'2Q Gates':>10} {'Depth':>8}")
    print("-" * 55)

    for strategy in strategies:
        r = results.get(strategy, {})
        if 'error' in r:
            print(f"{strategy:<15} {'ERROR':>10}")
        elif 'mean_vis' in r:
            print(f"{strategy:<15} {r['mean_vis']:>10.4f} {r['min_vis']:>10.4f} {r['n_2q']:>10} {r['depth']:>8}")
        else:
            print(f"{strategy:<15} {'N/A':>10}")

    if 'sabre_best' in results:
        r = results['sabre_best']
        print(f"{'sabre_best':<15} {r['mean_vis']:>10.4f} {r['min_vis']:>10.4f} {r['n_2q']:>10} {r['depth']:>8}")

    # Compare with optimal (non-overlapping pairs only)
    if has_vis_data:
        # Greedy selection of best non-overlapping pairs
        used_qubits = set()
        optimal_pairs = []
        for p in vis_data['predictions']:
            q1, q2 = p['q1'], p['q2']
            if q1 not in used_qubits and q2 not in used_qubits:
                optimal_pairs.append(p)
                used_qubits.add(q1)
                used_qubits.add(q2)
                if len(optimal_pairs) == 12:
                    break

        top_vis = [p['visibility'] for p in optimal_pairs]
        top_mean = np.mean(top_vis)
        top_min = min(top_vis)
        print(f"\n{'OPTIMAL (12 disjoint)':<20} {top_mean:>10.4f} {top_min:>10.4f} {'12':>10} {'-':>8}")

    return results


def run_single_angle(phi_deg, num_shots, backend_name, strategy='default',
                     dry_run=False, enable_error_mitigation=True):
    """
    Run Leggett test for a single angle with specified layout strategy.
    """
    phi_rad = np.radians(phi_deg)

    print(f"\n{'='*70}")
    print(f"Running φ = ±{abs(phi_deg):.1f}° ({num_shots} shots, strategy={strategy})")
    print(f"{'='*70}")

    # Create circuit
    qc = create_leggett_circuit_for_angle(phi_rad)

    # Connect to IBM Quantum
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    print(f"  Transpiling with {strategy} strategy...")
    print(f"  Original circuit: {qc.num_qubits} qubits, depth {qc.depth()}")

    qc_transpiled = transpile_with_strategy(qc, backend, strategy)
    pairs = get_selected_pairs(qc_transpiled)

    print(f"  Transpiled: depth {qc_transpiled.depth()}, ops {qc_transpiled.count_ops()}")
    print(f"  Physical pairs: {pairs}")

    if dry_run:
        print(f"  DRY RUN: Would submit job here. Exiting.")
        return {'pairs': pairs, 'strategy': strategy}

    # Configure error mitigation
    options = SamplerOptions()
    if enable_error_mitigation:
        options.execution.init_qubits = True
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = "XX"
        options.twirling.enable_gates = True
        options.twirling.enable_measure = True
        print(f"  Error mitigation: enabled")

    # Run with Sampler
    print(f"  Submitting job...")
    sampler = Sampler(backend, options=options)
    job = sampler.run([qc_transpiled], shots=num_shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Waiting for results...")

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
    counts = dict(counts_array)

    # Extract correlations
    correlations_pos, correlations_neg = extract_correlations_from_counts(counts, num_shots)
    result_pos = calc_leggett_for_angle(correlations_pos, phi_rad)
    result_neg = calc_leggett_for_angle(correlations_neg, -phi_rad)

    print(f"\n  Results for φ = +{abs(phi_deg):.1f}°:")
    print(f"    L₃ (exp):    {result_pos['L3']:.4f}")
    print(f"    L₃ (theory): {result_pos['L3_theory']:.4f}")
    print(f"    Violated:    {result_pos['violated']}")

    print(f"\n  Results for φ = -{abs(phi_deg):.1f}°:")
    print(f"    L₃ (exp):    {result_neg['L3']:.4f}")
    print(f"    L₃ (theory): {result_neg['L3_theory']:.4f}")
    print(f"    Violated:    {result_neg['violated']}")

    return {
        'positive': {
            'phi_deg': abs(phi_deg),
            **result_pos,
            'job_id': job_id,
            'strategy': strategy,
            'pairs': pairs,
            'timestamp': datetime.now().isoformat()
        },
        'negative': {
            'phi_deg': -abs(phi_deg),
            **result_neg,
            'job_id': job_id,
            'strategy': strategy,
            'pairs': pairs,
            'timestamp': datetime.now().isoformat()
        },
        'counts': counts
    }


def main():
    import sys

    # Parse arguments
    compare_only = '--compare' in sys.argv
    dry_run = '--dry-run' in sys.argv
    use_ibm = '--ibm' in sys.argv
    backend_name = "ibm_pittsburgh"
    strategy = 'default'
    num_shots = 1000

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--strategy' and i + 1 < len(sys.argv):
            strategy = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            num_shots = int(sys.argv[i + 1])

    # Parse --seeds argument
    sabre_seeds = 1
    for i, arg in enumerate(sys.argv):
        if arg == '--seeds' and i + 1 < len(sys.argv):
            sabre_seeds = int(sys.argv[i + 1])

    if compare_only:
        # Just compare layout strategies without running jobs (no hardware execution)
        compare_layouts(backend_name, sabre_seeds=sabre_seeds)
        return

    if not use_ibm:
        print("=" * 70)
        print("DRY RUN MODE (no --ibm flag)")
        print("=" * 70)
        print(f"\nThis will show what pairs would be selected with {strategy} strategy.")
        print("Add --ibm flag to actually run on IBM Quantum hardware.\n")

        # Just show what pairs would be selected
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        phi_rad = np.radians(30)
        qc = create_leggett_circuit_for_angle(phi_rad)
        qc_t = transpile_with_strategy(qc, backend, strategy)
        pairs = get_selected_pairs(qc_t)

        print(f"Strategy: {strategy}")
        print(f"Physical pairs that would be used: {pairs}")
        print(f"Transpiled depth: {qc_t.depth()}")
        return

    print("=" * 70)
    print(f"LEGGETT TEST WITH {strategy.upper()} LAYOUT STRATEGY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Backend: {backend_name} (REAL HARDWARE)")
    print(f"  Strategy: {strategy}")
    print(f"  Shots: {num_shots}")
    print(f"  Dry run: {dry_run}")

    if dry_run:
        # Just show what pairs would be selected
        result = run_single_angle(30, num_shots, backend_name, strategy, dry_run=True)
        print(f"\nPairs that would be used: {result['pairs']}")
        return

    # Run full test
    test_angles = [15, 25, 30, 45, 60]
    results = []

    for angle in test_angles:
        result = run_single_angle(angle, num_shots, backend_name, strategy)
        if result:
            results.append(result['positive'])
            results.append(result['negative'])
        time.sleep(2)

    # Save results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'leggett_results_{backend_name}_{strategy}_24qb.json')
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Done!")


if __name__ == "__main__":
    main()
