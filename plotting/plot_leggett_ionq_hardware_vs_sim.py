#!/usr/bin/env python3
"""
Plot Leggett inequality test results comparing IonQ Forte-1 hardware vs simulator (24-qubit).
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def expand_old_format(results):
    """
    Convert old format results (with L3_pos/L3_neg) to new format.
    For positive angles: use L3_pos
    For negative angles: use L3_neg
    New format results (with L3 or L3_exp) are returned as-is.
    """
    expanded = []
    for r in results:
        if 'L3_pos' in r and 'L3_neg' in r:
            # Old format: choose L3_pos or L3_neg based on angle sign
            phi = r['phi_deg']
            r_new = {
                'phi_deg': phi,
                'L3': r['L3_pos'] if phi >= 0 else r['L3_neg'],
                'violated': r.get('violated_pos' if phi >= 0 else 'violated_neg', False)
            }
            expanded.append(r_new)
        else:
            # New format: already has single angle
            r_new = r.copy()
            if 'L3_exp' in r:
                r_new['L3'] = r['L3_exp']
            if 'violated' not in r_new:
                r_new['violated'] = False
            expanded.append(r_new)

    return expanded


def plot_leggett_ionq_hardware_vs_sim():
    """Create comparison plot of IonQ hardware vs simulator results."""

    # Load and expand results (expands old L3_pos/L3_neg format into separate points)
    ionq_hw_results = expand_old_format(load_results('leggett_results_ionq_forte_sequential_24qb.json'))
    ionq_sim_results = expand_old_format(load_results('leggett_results_ionq_forte_IDEAL_24qb.json'))
    ionq_noise_results = expand_old_format(load_results('leggett_results_ionq_forte_NOISE_MODEL_24qb.json'))
    ibm_hw_results = expand_old_format(load_results('leggett_results_ibm_ibm_pittsburgh_sequential_24qb.json'))
    ibm_sim_results = expand_old_format(load_results('leggett_results_ibm_sequential_SIM_noiseless_24qb.json'))
    ibm_noise_results = expand_old_format(load_results('leggett_results_ibm_ibm_pittsburgh_NOISE_MODEL_24qb.json'))
    ibm_2q_results = expand_old_format(load_results('leggett_results_ibm_midcircuit_ibm_pittsburgh_20251117_062142.json'))
    ibm_2q_noise_results = expand_old_format(load_results('leggett_results_ibm_2qb_noise_ibm_pittsburgh_20251116_204211.json'))

    # Extract data
    ionq_hw_angles = [r['phi_deg'] for r in ionq_hw_results]
    ionq_hw_L3 = [r['L3'] for r in ionq_hw_results]

    ionq_sim_angles = [r['phi_deg'] for r in ionq_sim_results]
    ionq_sim_L3 = [r['L3'] for r in ionq_sim_results]

    ionq_noise_angles = [r['phi_deg'] for r in ionq_noise_results]
    ionq_noise_L3 = [r['L3'] for r in ionq_noise_results]

    ibm_hw_angles = [r['phi_deg'] for r in ibm_hw_results]
    ibm_hw_L3 = [r['L3'] for r in ibm_hw_results]

    ibm_sim_angles = [r['phi_deg'] for r in ibm_sim_results]
    ibm_sim_L3 = [r['L3'] for r in ibm_sim_results]

    ibm_noise_angles = [r['phi_deg'] for r in ibm_noise_results]
    ibm_noise_L3 = [r['L3'] for r in ibm_noise_results]

    ibm_2q_angles = [r['phi_deg'] for r in ibm_2q_results]
    ibm_2q_L3 = [r['L3'] for r in ibm_2q_results]

    ibm_2q_noise_angles = [r['phi_deg'] for r in ibm_2q_noise_results]
    ibm_2q_noise_L3 = [r['L3'] for r in ibm_2q_noise_results]

    # Create smooth curves for theory
    phi_smooth = np.linspace(-90, 90, 200)
    phi_rad_smooth = np.radians(phi_smooth)

    # Leggett bound (upper bound for Leggett model)
    L_bound_smooth = 2 - (2/3) * np.abs(np.sin(phi_rad_smooth/2))

    # Quantum mechanical prediction (pure singlet)
    L_QM_smooth = 2 * np.abs(np.cos(phi_rad_smooth/2))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot theory curves
    ax.plot(phi_smooth, L_bound_smooth, 'k-', linewidth=2, label='Leggett Bound', zorder=1)
    ax.plot(phi_smooth, L_QM_smooth, '--', color='gray', linewidth=2, label='QM Prediction', zorder=2)

    ionq = "IonQ Forte-1"
    #ionq = "Vendor 1"
    ibm = "IBM Pittsburgh"
    #ibm = "Vendor 2"
    # Plot experimental data - IonQ = green, IBM = blue, 2Q sequence = purple
    # Shapes: ideal sim = triangle, noisy sim = circle, real hardware = diamond, 2Q sequence = square
    # Simulators are hollow (facecolors='none')
    ax.scatter(ionq_hw_angles, ionq_hw_L3, s=120, marker='D', color='green',
               edgecolors='darkgreen', linewidth=2, label=ionq+' Hardware', zorder=8)
    ax.scatter(ionq_noise_angles, ionq_noise_L3, s=120, marker='o', facecolors='none',
               edgecolors='green', linewidth=2, linestyle='--', label=ionq+' Noise Model', zorder=7)
    ax.scatter(ionq_sim_angles, ionq_sim_L3, s=120, marker='^', facecolors='none',
               edgecolors='green', linewidth=2, linestyle='--', label='IONQ Ideal Simulator', zorder=6)
    ax.scatter(ibm_hw_angles, ibm_hw_L3, s=120, marker='D', color='blue',
               edgecolors='darkblue', linewidth=2, label=ibm+' Hardware', zorder=5)
    ax.scatter(ibm_noise_angles, ibm_noise_L3, s=120, marker='o', facecolors='none',
               edgecolors='blue', linewidth=2, linestyle='--', label=ibm + ' Noise Model', zorder=4)
    ax.scatter(ibm_sim_angles, ibm_sim_L3, s=120, marker='^', facecolors='none',
               edgecolors='blue', linewidth=2, linestyle='--', label='IBM Ideal Simulator', zorder=3)
    ax.scatter(ibm_2q_noise_angles, ibm_2q_noise_L3, s=120, marker='o', facecolors='none',
               edgecolors='purple', linewidth=2, linestyle='--', label=ibm+' 2Q Sequence (Noisy Sim)', zorder=9)
    ax.scatter(ibm_2q_angles, ibm_2q_L3, s=120, marker='s', color='purple',
               edgecolors='darkviolet', linewidth=2, label=ibm+' 2Q Sequence Hardware', zorder=10)

    # Styling
    ax.set_xlabel('Angle φ (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('L₃ Parameter', fontsize=14, fontweight='bold')
    ax.set_title('Leggett Inequality Violation',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)

    # Set axis limits
    ax.set_xlim(-90, 90)
    ax.set_ylim(1.5, 2.05)

    # Add violation counts (handle both boolean and string "True"/"False")
    ionq_hw_violations = sum(1 for r in ionq_hw_results if (r['violated'] if isinstance(r['violated'], bool) else r['violated'] == 'True'))
    ionq_noise_violations = sum(1 for r in ionq_noise_results if (r['violated'] if isinstance(r['violated'], bool) else r['violated'] == 'True'))
    ionq_sim_violations = sum(1 for r in ionq_sim_results if (r['violated'] if isinstance(r['violated'], bool) else r['violated'] == 'True'))
    ibm_hw_violations = sum(1 for r in ibm_hw_results if (r['violated'] if isinstance(r['violated'], bool) else r['violated'] == 'True'))
    ibm_noise_violations = sum(1 for r in ibm_noise_results if (r['violated'] if isinstance(r['violated'], bool) else r['violated'] == 'True'))
    ibm_sim_violations = sum(1 for r in ibm_sim_results if (r['violated'] if isinstance(r['violated'], bool) else r['violated'] == 'True'))
    ibm_2q_violations = sum(1 for r in ibm_2q_results if (r['violated'] if isinstance(r['violated'], bool) else r['violated'] == 'True'))
    ibm_2q_noise_violations = sum(1 for r in ibm_2q_noise_results if (r['violated'] if isinstance(r['violated'], bool) else r['violated'] == 'True'))

    """
    # Add violation counts in cleaner format
    info_text = f'Violations / Total Tests\n'
    info_text += f'Vendor 2 Hardware: {ionq_hw_violations}/{len(ionq_hw_results)}\n'
    info_text += f'Vendor 2 Noise: {ionq_noise_violations}/{len(ionq_noise_results)}\n'
    info_text += f'Vendor 2 Ideal: {ionq_sim_violations}/{len(ionq_sim_results)}\n'
    info_text += f'Vendor 2 Hardware: {ibm_hw_violations}/{len(ibm_hw_results)}\n'
    info_text += f'Vendor 2 Noise: {ibm_noise_violations}/{len(ibm_noise_results)}\n'
    info_text += f'Vendor 2 Ideal: {ibm_sim_violations}/{len(ibm_sim_results)}\n'
    info_text += f'Vendor 2 2Q Seq: {ibm_2q_violations}/{len(ibm_2q_results)}\n'
    info_text += f'Vendor 2 2Q Noise: {ibm_2q_noise_violations}/{len(ibm_2q_noise_results)}'

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1))
    """
    plt.tight_layout()

    # Save figure
    output_file = 'leggett_ionq_hardware_vs_sim_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    plt.show()


if __name__ == "__main__":
    plot_leggett_ionq_hardware_vs_sim()
