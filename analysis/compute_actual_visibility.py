#!/usr/bin/env python3
"""
Compute actual visibility from L3 measurements and generate LaTeX table.

V = L3_measured / L3_theory

Where L3_theory = 2 * cos(phi/2)

Columns: V_predicted (from hardware characterization model),
         V_noise_sim (noise model simulation),
         V_experimental (actual hardware)
"""

import json
import numpy as np
from pathlib import Path


def compute_visibility_from_file(json_path):
    """Compute visibility from L3 results file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    visibilities = []
    violations = 0
    total = 0

    for entry in data:
        L3_theory = entry.get('L3_theory')
        if not L3_theory:
            continue

        # Handle different JSON structures
        if 'L3' in entry:
            L3_meas = entry['L3']
        elif 'L3_pos' in entry:
            L3_meas = (entry['L3_pos'] + entry['L3_neg']) / 2
        else:
            continue

        V = L3_meas / L3_theory
        visibilities.append(V)

        # Count violations - handle different file formats
        total += 1
        if 'violated' in entry:
            # IBM format: single violated field
            violated = entry.get('violated', False)
            if violated == 'True' or violated is True:
                violations += 1
        elif 'violated_pos' in entry:
            # IonQ format: count as violated if average L3 > bound
            L3_avg = (entry['L3_pos'] + entry['L3_neg']) / 2
            if L3_avg > entry['bound']:
                violations += 1

    mean = np.mean(visibilities) if visibilities else 0
    std = np.std(visibilities) if visibilities else 0
    return mean, std, violations, total


def get_predicted_visibility(platform):
    """Get V_predicted from hardware characterization model."""
    # These come from visibility_predictions_ibm_pittsburgh*.json
    # Using date-matched calibration data for each experiment
    predictions = {
        'IBM_newbits': 0.990,  # newbits layout (12-14 calibration, 12-15 experiment)
        'IBM_10k': 0.981,      # 10k layout (12-08 experiment)
        'IBM_default': 0.943,  # Default transpile layout
        'IonQ': 0.979,         # Published specs (all-to-all connectivity)
        'Rigetti': 0.898,      # Top 12 pairs
        'IQM': 0.877,          # Top 12 pairs
    }
    return predictions.get(platform, None)


def fmt(val, std=None, decimals=3):
    """Format value with optional std dev for LaTeX."""
    if val is None:
        return "---"
    if std is None or std == 0 or std < 0.0005:
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def main():
    base = Path('/Users/adc/qsim/relative_phase_variant/geometry/simulator_gnuradio')

    # Define data sources for each platform
    # IBM layouts: newbits (12-15 experiment) and 10k (12-08 experiment)
    platforms = {
        'IBM_newbits': {
            'noise_sim': base / 'paper/ibm/leggett_results_ibm_ibm_pittsburgh_NOISE_MODEL_24qb_BEST_LAYOUT_newbits.json',
            'experimental': base / 'paper/ibm/leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT_newbits.json',
        },
        'IBM_10k': {
            'noise_sim': None,  # No noise sim for this layout
            'experimental': base / 'paper/ibm/leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT_10k.json',
        },
        'IonQ': {
            'noise_sim': base / 'leggett_results_ionq_forte_NOISE_MODEL_24qb.json',
            'experimental': base / 'paper/data/leggett_results_ionq_forte_sequential_24qb.json',
        },
    }

    results = {}

    print("=" * 70)
    print("Computing Visibility from L3 Measurements")
    print("=" * 70)

    for platform, paths in platforms.items():
        results[platform] = {
            'V_predicted': (get_predicted_visibility(platform), None),
        }

        # Noise simulation
        if paths['noise_sim'] and paths['noise_sim'].exists():
            mean, std, viol, total = compute_visibility_from_file(paths['noise_sim'])
            results[platform]['V_noise_sim'] = (mean, std)
            results[platform]['violations_sim'] = (viol, total)
            print(f"{platform} Noise Sim: V = {mean:.4f} ± {std:.4f}, violations = {viol}/{total}")
        else:
            results[platform]['V_noise_sim'] = (None, None)
            results[platform]['violations_sim'] = (None, None)
            if paths['noise_sim']:
                print(f"{platform} Noise Sim: not found")
            else:
                print(f"{platform} Noise Sim: N/A")

        # Experimental hardware
        if paths['experimental'].exists():
            mean, std, viol, total = compute_visibility_from_file(paths['experimental'])
            results[platform]['V_experimental'] = (mean, std)
            results[platform]['violations_exp'] = (viol, total)
            print(f"{platform} Hardware:  V = {mean:.4f} ± {std:.4f}, violations = {viol}/{total}")
        else:
            results[platform]['V_experimental'] = (None, None)
            results[platform]['violations_exp'] = (None, None)
            print(f"{platform} Hardware: not found")

    # Print summary table
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Platform':<20} {'V_predicted':<12} {'V_noise_sim':<14} {'V_experimental':<14} {'Violations':<12}")
    print("-" * 80)

    # IBM newbits layout
    r = results['IBM_newbits']
    v_pred = r['V_predicted'][0]
    v_sim = r['V_noise_sim'][0]
    v_exp = r['V_experimental'][0]
    viol_exp = r['violations_exp']
    viol_str = f"{viol_exp[0]}/{viol_exp[1]}" if viol_exp[0] is not None else "---"
    print(f"{'IBM (newbits)':<20} {fmt(v_pred):<12} {fmt(v_sim, r['V_noise_sim'][1]):<14} "
          f"{fmt(v_exp, r['V_experimental'][1]):<14} {viol_str:<12}")

    # IBM 10k layout
    r = results['IBM_10k']
    v_pred = r['V_predicted'][0]
    v_sim = r['V_noise_sim'][0]
    v_exp = r['V_experimental'][0]
    viol_exp = r['violations_exp']
    viol_str = f"{viol_exp[0]}/{viol_exp[1]}" if viol_exp[0] is not None else "---"
    v_sim_str = fmt(v_sim, r['V_noise_sim'][1]) if v_sim else "---"
    print(f"{'IBM (10k)':<20} {fmt(v_pred):<12} {v_sim_str:<14} "
          f"{fmt(v_exp, r['V_experimental'][1]):<14} {viol_str:<12}")

    # IBM default (no data)
    v_pred_default = get_predicted_visibility('IBM_default')
    print(f"{'IBM (default)':<20} {fmt(v_pred_default):<12} {'---':<14} {'---':<14} {'---':<12}")

    # IonQ
    r = results['IonQ']
    v_pred = r['V_predicted'][0]
    v_sim = r['V_noise_sim'][0]
    v_exp = r['V_experimental'][0]
    viol_exp = r['violations_exp']
    viol_str = f"{viol_exp[0]}/{viol_exp[1]}" if viol_exp[0] is not None else "---"
    print(f"{'IonQ':<20} {fmt(v_pred):<12} {fmt(v_sim, r['V_noise_sim'][1]):<14} "
          f"{fmt(v_exp, r['V_experimental'][1]):<14} {viol_str:<12}")

    # Generate LaTeX table
    print()
    print("=" * 70)
    print("LaTeX Table Code")
    print("=" * 70)

    rows = []

    # IBM newbits layout
    r = results['IBM_newbits']
    v_pred_val = r['V_predicted'][0]
    v_sim_val, v_sim_std = r['V_noise_sim']
    v_exp_val, v_exp_std = r['V_experimental']
    viol = r['violations_exp']
    viol_str = f"{viol[0]}/{viol[1]} ({100*viol[0]/viol[1]:.0f}\\%)" if viol[0] is not None else "---"
    rows.append(f"IBM (newbits) & {fmt(v_pred_val)} & {fmt(v_sim_val, v_sim_std)} & {fmt(v_exp_val, v_exp_std)} & {viol_str} \\\\")

    # IBM 10k layout
    r = results['IBM_10k']
    v_pred_val = r['V_predicted'][0]
    v_sim_val, v_sim_std = r['V_noise_sim']
    v_exp_val, v_exp_std = r['V_experimental']
    viol = r['violations_exp']
    viol_str = f"{viol[0]}/{viol[1]} ({100*viol[0]/viol[1]:.0f}\\%)" if viol[0] is not None else "---"
    v_sim_str = fmt(v_sim_val, v_sim_std) if v_sim_val else "---"
    rows.append(f"IBM (10k) & {fmt(v_pred_val)} & {v_sim_str} & {fmt(v_exp_val, v_exp_std)} & {viol_str} \\\\")

    # IBM default (no data)
    v_pred_default = get_predicted_visibility('IBM_default')
    rows.append(f"IBM (default) & {fmt(v_pred_default)} & --- & --- & --- \\\\")

    # IonQ
    r = results['IonQ']
    v_pred_val = r['V_predicted'][0]
    v_sim_val, v_sim_std = r['V_noise_sim']
    v_exp_val, v_exp_std = r['V_experimental']
    viol = r['violations_exp']
    viol_str = f"{viol[0]}/{viol[1]} ({100*viol[0]/viol[1]:.0f}\\%)" if viol[0] is not None else "---"
    rows.append(f"IonQ & {fmt(v_pred_val)} & {fmt(v_sim_val, v_sim_std)} & {fmt(v_exp_val, v_exp_std)} & {viol_str} \\\\")

    print(f"""
\\begin{{table}}[H]
\\centering
\\caption{{Predicted vs Measured Visibility}}
\\label{{tab:visibility_comparison}}
\\small
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\textbf{{Platform}} & $V_{{\\text{{predicted}}}}$ & $V_{{\\text{{noise sim}}}}$ & $V_{{\\text{{experimental}}}}$ & \\textbf{{Violations}} \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\vspace{{0.5em}}
\\footnotesize
\\textit{{$V_{{\\text{{predicted}}}}$ from hardware characterization model. $V_{{\\text{{noise sim}}}}$ from backend noise model simulation. $V_{{\\text{{experimental}}}}$ from hardware execution.}}
\\end{{table}}
""")

    # Also show the gap analysis
    print()
    print("=" * 70)
    print("Gap Analysis (for paper discussion)")
    print("=" * 70)

    for platform in ['IBM_newbits', 'IBM_10k', 'IonQ']:
        r = results[platform]
        v_pred = r['V_predicted'][0]
        v_sim = r['V_noise_sim'][0]
        v_exp = r['V_experimental'][0]

        print(f"\n{platform}:")
        if v_pred and v_exp:
            gap_pred_exp = v_pred - v_exp
            print(f"  V_predicted - V_experimental = {gap_pred_exp:.4f} ({gap_pred_exp*100:.1f}%)")
        if v_sim and v_exp:
            gap_sim_exp = v_sim - v_exp
            print(f"  V_noise_sim - V_experimental = {gap_sim_exp:.4f} ({gap_sim_exp*100:.1f}%)")
        if v_pred and v_sim:
            gap_pred_sim = v_pred - v_sim
            print(f"  V_predicted - V_noise_sim    = {gap_pred_sim:.4f} ({gap_pred_sim*100:.1f}%)")

    # IBM layout comparison
    print()
    print("=" * 70)
    print("IBM Layout Comparison")
    print("=" * 70)
    v_ibm_newbits = get_predicted_visibility('IBM_newbits')
    v_ibm_10k = get_predicted_visibility('IBM_10k')
    v_ibm_default = get_predicted_visibility('IBM_default')
    print(f"\n  V_predicted (newbits layout): {v_ibm_newbits:.3f}")
    print(f"  V_predicted (10k layout):     {v_ibm_10k:.3f}")
    print(f"  V_predicted (default layout): {v_ibm_default:.3f}")
    print(f"  Improvement newbits vs default: +{(v_ibm_newbits - v_ibm_default)*100:.1f}%")
    print(f"  Improvement 10k vs default:     +{(v_ibm_10k - v_ibm_default)*100:.1f}%")


if __name__ == "__main__":
    main()
