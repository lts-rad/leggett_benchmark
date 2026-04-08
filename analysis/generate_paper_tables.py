#!/usr/bin/env python3
"""
Generate table values for paper.tex

Tables:
1. Leggett violation requirements at different phi angles
2. Predicted visibility from all hardware platforms
"""

import json
import numpy as np


def calculate_leggett_requirements():
    """Calculate L3 bounds and visibility requirements for violation."""
    print("=" * 70)
    print("Table 1: Leggett Violation Requirements")
    print("=" * 70)
    print(f"{'φ (deg)':<10} {'L3_Leggett':<12} {'L3_QM':<12} {'V_required':<12}")
    print("-" * 46)

    for phi_deg in [15, 30, 45, 60]:
        phi_rad = np.radians(phi_deg)
        L3_leggett = 2 - (2/3) * np.sin(phi_rad / 2)
        L3_qm = 2 * np.cos(phi_rad / 2)
        V_required = L3_leggett / L3_qm
        print(f"{phi_deg:<10} {L3_leggett:<12.3f} {L3_qm:<12.3f} {V_required:<12.3f}")
    print()


def calculate_ibm_predictions(json_path, n_pairs):
    """Calculate IBM visibility predictions from saved data.

    Args:
        json_path: Path to predictions JSON file
        n_pairs: Number of top pairs to use, or None to use all pairs
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    predictions = data['predictions'] if n_pairs is None else data['predictions'][:n_pairs]

    vis = [p['visibility'] for p in predictions]
    ro = [p['readout_fidelity'] for p in predictions]
    gate = [p['gate_fidelity'] for p in predictions]
    t2 = [p['t2_fidelity'] for p in predictions]

    return {
        'F_readout': (np.mean(ro), np.std(ro)),
        'F_gate': (np.mean(gate), np.std(gate)),
        'F_T2': (np.mean(t2), np.std(t2)),
        'V_predicted': (np.mean(vis), np.std(vis)),
    }


def calculate_ibm_auto_predictions(json_path):
    """Calculate IBM visibility for default transpiler-selected pairs.

    These are the pairs selected by transpile() with optimization_level=3
    for a 24-qubit Leggett circuit on ibm_pittsburgh.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Pairs selected by default transpiler (from actual transpilation)
    auto_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (16, 23),
                  (17, 27), (21, 22), (24, 25), (37, 45), (46, 47), (86, 87)]

    # Create lookup dict
    pair_to_pred = {}
    for p in data['predictions']:
        key = (min(p['q1'], p['q2']), max(p['q1'], p['q2']))
        pair_to_pred[key] = p

    # Get predictions for auto-selected pairs
    predictions = [pair_to_pred[pair] for pair in auto_pairs if pair in pair_to_pred]

    vis = [p['visibility'] for p in predictions]
    ro = [p['readout_fidelity'] for p in predictions]
    gate = [p['gate_fidelity'] for p in predictions]
    t2 = [p['t2_fidelity'] for p in predictions]

    return {
        'F_readout': (np.mean(ro), np.std(ro)),
        'F_gate': (np.mean(gate), np.std(gate)),
        'F_T2': (np.mean(t2), np.std(t2)),
        'V_predicted': (np.mean(vis), np.std(vis)),
    }


def calculate_ionq_predictions():
    """Calculate IonQ visibility predictions from published specs."""
    # IonQ Forte-1 characterization data (Dec 2025)
    e_1q = 0.00020      # 0.020% median 1Q gate error
    e_2q = 0.00540      # 0.540% median 2Q gate error
    e_spam = 0.00400    # 0.400% SPAM error per qubit
    T2 = 1.0            # seconds
    t_1q = 130          # µs
    t_2q = 970          # µs

    n_1q = 12
    n_2q = 1

    t_circuit_s = (n_1q * t_1q + n_2q * t_2q) * 1e-6

    F_readout = (1 - e_spam) ** 2
    F_gate = ((1 - e_1q) ** n_1q) * ((1 - e_2q) ** n_2q)
    F_T2 = np.exp(-t_circuit_s / T2)
    V_predicted = F_readout * F_gate * F_T2

    # No std dev for IonQ (uses published medians)
    return {
        'F_readout': (F_readout, None),
        'F_gate': (F_gate, None),
        'F_T2': (F_T2, None),
        'V_predicted': (V_predicted, None),
    }


def calculate_braket_predictions(json_path, top_n=12):
    """Calculate visibility predictions from Braket device JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract single-qubit properties
    qubit_props = {}
    for qubit_id, props in data['oneQubitProperties'].items():
        qid = int(qubit_id)
        t1_val = props.get('T1', {}).get('value')
        t2_val = props.get('T2', {}).get('value')
        qubit_props[qid] = {
            't1': t1_val if t1_val and t1_val > 0 else 50e-6,
            't2': t2_val if t2_val and t2_val > 0 else 20e-6,
            'readout': 0.95,
            'oneq_fid': 0.999,
        }
        for fid in props.get('oneQubitFidelity', []):
            fid_type = fid['fidelityType']['name']
            if fid_type == 'READOUT':
                qubit_props[qid]['readout'] = fid['fidelity']
            elif fid_type in ['RANDOMIZED_BENCHMARKING', 'SIMULTANEOUS_RANDOMIZED_BENCHMARKING']:
                qubit_props[qid]['oneq_fid'] = fid['fidelity']

    # Calculate per-pair visibility
    pair_predictions = []
    n_1q = 12
    n_2q = 1
    circuit_time = 0.6e-6

    if 'twoQubitProperties' in data:
        for edge_id, props in data['twoQubitProperties'].items():
            parts = edge_id.replace('-', '_').split('_')
            if len(parts) != 2:
                continue
            try:
                q1, q2 = int(parts[0]), int(parts[1])
            except ValueError:
                continue

            if q1 not in qubit_props or q2 not in qubit_props:
                continue

            twoq_fid = 0.98
            for fid in props.get('twoQubitGateFidelity', []):
                twoq_fid = fid['fidelity']
                break

            qp1, qp2 = qubit_props[q1], qubit_props[q2]

            if qp1['t2'] < 1e-9 or qp2['t2'] < 1e-9:
                continue

            F_readout = qp1['readout'] * qp2['readout']
            avg_1q_error = (1 - qp1['oneq_fid'] + 1 - qp2['oneq_fid']) / 2
            F_gate = ((1 - avg_1q_error) ** n_1q) * (twoq_fid ** n_2q)
            F_T2 = np.exp(-circuit_time / qp1['t2']) * np.exp(-circuit_time / qp2['t2'])

            pair_predictions.append({
                'readout_fidelity': F_readout,
                'gate_fidelity': F_gate,
                't2_fidelity': F_T2,
                'visibility': F_readout * F_gate * F_T2,
            })

    # Sort and take top N
    pair_predictions.sort(key=lambda x: x['visibility'], reverse=True)
    top = pair_predictions[:top_n]

    ro = [p['readout_fidelity'] for p in top]
    gate = [p['gate_fidelity'] for p in top]
    t2 = [p['t2_fidelity'] for p in top]
    vis = [p['visibility'] for p in top]

    return {
        'F_readout': (np.mean(ro), np.std(ro)),
        'F_gate': (np.mean(gate), np.std(gate)),
        'F_T2': (np.mean(t2), np.std(t2)),
        'V_predicted': (np.mean(vis), np.std(vis)),
    }


def fmt(val, std, decimals=3):
    """Format value with optional std dev."""
    if std is None or std == 0:
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def main():
    print()
    calculate_leggett_requirements()

    print("=" * 70)
    print("Table 2: Predicted Visibility from Hardware Characterization")
    print("=" * 70)
    print()

    # Collect all predictions
    results = {}

    # IBM predictions
    try:
        results['IBM Pittsburgh (24)'] = calculate_ibm_predictions('visibility_predictions_ibm_pittsburgh.json', 24)
        # Also calculate "auto" - actual transpiler-selected pairs
        results['IBM Pittsburgh (auto)'] = calculate_ibm_auto_predictions('visibility_predictions_ibm_pittsburgh.json')
        print("IBM Pittsburgh loaded (24 best + auto-selected)")
    except FileNotFoundError:
        print("Warning: IBM predictions not found")

    # IonQ predictions
    results['IonQ Forte-1'] = calculate_ionq_predictions()
    print("IonQ calculated from published specs")

    # Rigetti predictions
    try:
        results['Rigetti Ankaa-3'] = calculate_braket_predictions('braket_ankaa_rigetti.json', 12)
        print("Rigetti Ankaa-3 loaded (top 12)")
    except FileNotFoundError:
        print("Warning: Rigetti predictions not found")

    # IQM predictions
    try:
        results['IQM Emerald'] = calculate_braket_predictions('braket_emerald_iqm.json', 12)
        print("IQM Emerald loaded (top 12)")
    except FileNotFoundError:
        print("Warning: IQM predictions not found")

    # Print summary
    print()
    print("-" * 70)
    platforms = ['IBM Pittsburgh (24)', 'IBM Pittsburgh (auto)', 'IonQ Forte-1', 'Rigetti Ankaa-3', 'IQM Emerald']
    for platform in platforms:
        if platform not in results:
            continue
        r = results[platform]
        print(f"\n{platform}:")
        for key in ['F_readout', 'F_gate', 'F_T2', 'V_predicted']:
            val, std = r[key]
            if std:
                print(f"  {key:<12} {val:.4f} ± {std:.4f}")
            else:
                print(f"  {key:<12} {val:.4f}")

    # Generate LaTeX
    print()
    print("=" * 70)
    print("LaTeX Table Code")
    print("=" * 70)

    def build_table_rows(cols, first_col_label='\\textbf{Factor}'):
        rows = []
        for metric, label in [('F_readout', '$F_{\\text{readout}}$'),
                              ('F_gate', '$F_{\\text{gate}}$'),
                              ('F_T2', '$F_{T_2}$')]:
            vals = []
            for p in cols:
                val, std = results[p][metric]
                vals.append(fmt(val, std))
            rows.append(f"{label} & " + " & ".join(vals) + " \\\\")

        # V_predicted row
        v_vals = []
        for p in cols:
            val, std = results[p]['V_predicted']
            v_vals.append(fmt(val, std))
        rows.append("\\hline")
        rows.append(f"$V_{{\\text{{predicted}}}}$ & " + " & ".join(v_vals) + " \\\\")
        return rows

    # Table (a): IBM (24), IBM (auto), and IonQ
    cols_a = [p for p in ['IBM Pittsburgh (24)', 'IBM Pittsburgh (auto)', 'IonQ Forte-1'] if p in results]
    header_a = " & ".join([f"\\textbf{{{c}}}" for c in cols_a])
    rows_a = build_table_rows(cols_a)

    # Table (b): Rigetti and IQM
    cols_b = [p for p in ['Rigetti Ankaa-3', 'IQM Emerald'] if p in results]
    header_b = " & ".join([f"\\textbf{{{c}}}" for c in cols_b])
    rows_b = build_table_rows(cols_b)

    print(f"""
\\begin{{table}}[H]
\\centering
\\caption{{Predicted Visibility from Hardware Characterization}}
\\label{{tab:hardware_visibility}}
\\small
\\begin{{subtable}}{{0.58\\textwidth}}
\\centering
\\begin{{tabular}}{{|l|{'c|' * len(cols_a)}}}
\\hline
\\textbf{{Factor}} & {header_a} \\\\
\\hline
{chr(10).join(rows_a)}
\\hline
\\end{{tabular}}
\\end{{subtable}}
\\hfill
\\begin{{subtable}}{{0.40\\textwidth}}
\\centering
\\begin{{tabular}}{{|l|{'c|' * len(cols_b)}}}
\\hline
\\textbf{{}} & {header_b} \\\\
\\hline
{chr(10).join(rows_b)}
\\hline
\\end{{tabular}}
\\end{{subtable}}
\\vspace{{0.5em}}
\\\\
\\footnotesize
\\textit{{IBM (24) uses top 24 pairs; IBM (auto) uses default transpiler layout; Rigetti and IQM use top 12 pairs. IonQ uses published median error rates.}}
\\end{{table}}
""")


if __name__ == "__main__":
    main()
