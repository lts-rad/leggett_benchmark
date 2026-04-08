#!/usr/bin/env python3
"""
Plot Leggett inequality test results comparing 4 quantum hardware vendors:
A) IonQ Forte-1
B) IBM Pittsburgh
C) IQM Emerald
D) Rigetti Ankaa-3

Shows: ideal (where available), noisy simulation (where available), and hardware results.
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_L3_error(correlations, num_shots=1000):
    """
    Calculate the standard error for L3 from correlations.

    σ_i = sqrt((1 - C_i²) / N) for each correlation
    σ_t1 = sqrt(σ_0² + σ_1²)  for first term
    σ_t2 = sqrt(σ_2² + σ_3²)  for second term
    σ_t3 = sqrt(σ_4² + σ_5²)  for third term
    σ_L3 = (1/3) * sqrt(σ_t1² + σ_t2² + σ_t3²)
    """
    if correlations is None or len(correlations) < 6:
        return 0.0

    C = correlations
    N = num_shots

    # Calculate σ for each correlation
    sigmas = [math.sqrt((1 - c**2) / N) for c in C]

    # Calculate σ for each term (sum of two correlations)
    sigma_t1 = math.sqrt(sigmas[0]**2 + sigmas[1]**2)
    sigma_t2 = math.sqrt(sigmas[2]**2 + sigmas[3]**2)
    sigma_t3 = math.sqrt(sigmas[4]**2 + sigmas[5]**2)

    # Calculate σ_L3
    sigma_L3 = (1/3) * math.sqrt(sigma_t1**2 + sigma_t2**2 + sigma_t3**2)

    return sigma_L3


def expand_old_format(results):
    """
    Convert old format results (with L3_pos/L3_neg) to new format.
    For positive angles: use L3_pos
    For negative angles: use L3_neg
    New format results (with L3 or L3_exp) are returned as-is.
    Also preserves correlations and num_shots for error calculation.
    """
    expanded = []
    for r in results:
        if 'L3_pos' in r and 'L3_neg' in r:
            # Old format: choose L3_pos or L3_neg based on angle sign
            phi = r['phi_deg']
            # Get correlations from pos/neg format
            if phi >= 0:
                corrs = r.get('correlations_pos') or r.get('correlations')
            else:
                corrs = r.get('correlations_neg') or r.get('correlations')
            r_new = {
                'phi_deg': phi,
                'L3': r['L3_pos'] if phi >= 0 else r['L3_neg'],
                'violated': r.get('violated_pos' if phi >= 0 else 'violated_neg', False),
                'correlations': corrs,
                'num_shots': r.get('num_shots', 1000)
            }
            expanded.append(r_new)
        else:
            # New format: already has single angle
            r_new = r.copy()
            if 'L3_exp' in r:
                r_new['L3'] = r['L3_exp']
            if 'violated' not in r_new:
                r_new['violated'] = False
            if 'num_shots' not in r_new:
                r_new['num_shots'] = 1000
            expanded.append(r_new)

    return expanded


def extract_with_errors(results):
    """Extract angles, L3 values, and error bars from results."""
    angles = [r['phi_deg'] for r in results]
    L3_values = [r['L3'] for r in results]
    errors = []
    for r in results:
        corrs = r.get('correlations')
        n_shots = r.get('num_shots', 1000)
        errors.append(calculate_L3_error(corrs, n_shots))
    return angles, L3_values, errors


def plot_leggett_4vendor_comparison():
    """Create comparison plot of 4 quantum hardware vendors."""

    # Load and expand results
    # IonQ Forte-1: ideal, noisy, hardware
    ionq_hw_results = expand_old_format(load_results('leggett_results_ionq_forte_sequential_24qb.json'))
    ionq_sim_results = expand_old_format(load_results('leggett_results_ionq_forte_IDEAL_24qb.json'))
    ionq_noise_results = expand_old_format(load_results('leggett_results_ionq_forte_NOISE_MODEL_24qb.json'))

    # IBM Pittsburgh: ideal, noisy, hardware
    ibm_hw_results = expand_old_format(load_results('leggett_results_ibm_ibm_pittsburgh_sequential_24qb.json'))
    ibm_sim_results = expand_old_format(load_results('leggett_results_ibm_sequential_SIM_noiseless_24qb.json'))
    ibm_noise_results = expand_old_format(load_results('leggett_results_ibm_ibm_pittsburgh_NOISE_MODEL_24qb.json'))

    # IQM Emerald: hardware only
    iqm_hw_results = expand_old_format(load_results('leggett_results_iqm_emerald_sequential.json'))

    # Rigetti Ankaa-3: hardware only
    rigetti_hw_results = expand_old_format(load_results('leggett_results_rigetti_ankaa3_sequential.json'))

    # Extract data with errors - IonQ
    ionq_hw_angles, ionq_hw_L3, ionq_hw_err = extract_with_errors(ionq_hw_results)
    ionq_sim_angles, ionq_sim_L3, ionq_sim_err = extract_with_errors(ionq_sim_results)
    ionq_noise_angles, ionq_noise_L3, ionq_noise_err = extract_with_errors(ionq_noise_results)

    # Extract data with errors - IBM
    ibm_hw_angles, ibm_hw_L3, ibm_hw_err = extract_with_errors(ibm_hw_results)
    ibm_sim_angles, ibm_sim_L3, ibm_sim_err = extract_with_errors(ibm_sim_results)
    ibm_noise_angles, ibm_noise_L3, ibm_noise_err = extract_with_errors(ibm_noise_results)

    # Extract data with errors - IQM
    iqm_hw_angles, iqm_hw_L3, iqm_hw_err = extract_with_errors(iqm_hw_results)

    # Extract data with errors - Rigetti
    rigetti_hw_angles, rigetti_hw_L3, rigetti_hw_err = extract_with_errors(rigetti_hw_results)

    # Create smooth curves for theory
    phi_smooth = np.linspace(-90, 90, 200)
    phi_rad_smooth = np.radians(phi_smooth)

    # Leggett bound (upper bound for Leggett model)
    L_bound_smooth = 2 - (2/3) * np.abs(np.sin(phi_rad_smooth/2))

    # Quantum mechanical prediction (pure singlet)
    L_QM_smooth = 2 * np.abs(np.cos(phi_rad_smooth/2))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot theory curves
    ax.plot(phi_smooth, L_bound_smooth, 'k-', linewidth=2.5, label='Leggett Bound', zorder=1)
    ax.plot(phi_smooth, L_QM_smooth, '--', color='gray', linewidth=2, label='QM Prediction', zorder=2)

    # Define colors for each vendor
    ionq_color = 'green'
    ibm_color = 'blue'
    iqm_color = 'orange'
    rigetti_color = 'red'

    # Vendor names
    ionq_name = "IonQ Forte-1"
    ibm_name = "IBM Pittsburgh"
    iqm_name = "IQM Emerald"
    rigetti_name = "Rigetti Ankaa-3"

    # Plot experimental data
    # Shapes: ideal sim = triangle (^), noisy sim = circle (o), real hardware = diamond (D)
    # Simulators are hollow (facecolors='none')
    # Hardware data gets 2σ (95% CI) error bars

    # Scale errors to 2-sigma for 95% confidence interval
    ionq_hw_err_2s = [2*e for e in ionq_hw_err]
    ibm_hw_err_2s = [2*e for e in ibm_hw_err]
    iqm_hw_err_2s = [2*e for e in iqm_hw_err]
    rigetti_hw_err_2s = [2*e for e in rigetti_hw_err]

    # Plot simulations first (lower zorder)
    ax.scatter(ionq_noise_angles, ionq_noise_L3, s=100, marker='o', facecolors='none',
               edgecolors=ionq_color, linewidth=2, label=f'{ionq_name} Noise Model', zorder=5)
    ax.scatter(ionq_sim_angles, ionq_sim_L3, s=100, marker='^', facecolors='none',
               edgecolors=ionq_color, linewidth=2, label=f'{ionq_name} Ideal', zorder=4)
    ax.scatter(ibm_noise_angles, ibm_noise_L3, s=100, marker='o', facecolors='none',
               edgecolors=ibm_color, linewidth=2, label=f'{ibm_name} Noise Model', zorder=5)
    ax.scatter(ibm_sim_angles, ibm_sim_L3, s=100, marker='^', facecolors='none',
               edgecolors=ibm_color, linewidth=2, label=f'{ibm_name} Ideal', zorder=4)

    # Plot hardware with error bars on top (higher zorder)
    # IonQ (green)
    ax.errorbar(ionq_hw_angles, ionq_hw_L3, yerr=ionq_hw_err_2s, fmt='none',
                ecolor='darkgreen', elinewidth=2, capsize=4, capthick=2, zorder=11)
    ax.scatter(ionq_hw_angles, ionq_hw_L3, s=120, marker='D', color=ionq_color,
               edgecolors='darkgreen', linewidth=2, label=f'{ionq_name} Hardware', zorder=12)

    # IBM (blue)
    ax.errorbar(ibm_hw_angles, ibm_hw_L3, yerr=ibm_hw_err_2s, fmt='none',
                ecolor='darkblue', elinewidth=2, capsize=4, capthick=2, zorder=9)
    ax.scatter(ibm_hw_angles, ibm_hw_L3, s=120, marker='D', color=ibm_color,
               edgecolors='darkblue', linewidth=2, label=f'{ibm_name} Hardware', zorder=10)

    # IQM (orange)
    ax.errorbar(iqm_hw_angles, iqm_hw_L3, yerr=iqm_hw_err_2s, fmt='none',
                ecolor='darkorange', elinewidth=2, capsize=4, capthick=2, zorder=7)
    ax.scatter(iqm_hw_angles, iqm_hw_L3, s=120, marker='D', color=iqm_color,
               edgecolors='darkorange', linewidth=2, label=f'{iqm_name} Hardware', zorder=8)

    # Rigetti (red)
    ax.errorbar(rigetti_hw_angles, rigetti_hw_L3, yerr=rigetti_hw_err_2s, fmt='none',
                ecolor='darkred', elinewidth=2, capsize=4, capthick=2, zorder=5)
    ax.scatter(rigetti_hw_angles, rigetti_hw_L3, s=120, marker='D', color=rigetti_color,
               edgecolors='darkred', linewidth=2, label=f'{rigetti_name} Hardware', zorder=6)

    # Styling
    ax.set_xlabel('Angle $\\varphi$ (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('$L_3$ Parameter', fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95, ncol=2)

    # Set axis limits
    ax.set_xlim(-90, 90)
    ax.set_ylim(0.6, 2.05)

    plt.tight_layout()

    # Save figure
    output_file = 'leggett_4vendor_comparison_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    plt.show()


if __name__ == "__main__":
    plot_leggett_4vendor_comparison()
