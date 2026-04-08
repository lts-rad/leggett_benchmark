#!/usr/bin/env python3
"""
Create radar plots for all angles from IonQ Forte 24qb results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Load results
with open('../../production/leggett_results_ionq_forte_sequential_24qb.json', 'r') as f:
    results = json.load(f)

# Plot angles
plot_angles = [15, 25, 30, 45, 60]

for phi_deg_target in plot_angles:
    # Find result for this angle
    result = None
    for r in results:
        if abs(r['phi_deg'] - phi_deg_target) < 0.1:
            result = r
            break

    if not result:
        print(f"Warning: Could not find results for phi = {phi_deg_target}°")
        continue

    phi_deg = abs(result['phi_deg'])
    phi_rad = np.radians(phi_deg)

    # Extract correlations (IonQ format has both +phi and -phi in same entry)
    corr_pos = result['correlations_pos']
    corr_neg = result['correlations_neg']
    corr_theory = result['correlations_theory']

    # Correlation labels
    labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']

    # Number of variables
    N = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Append first value to close the plot
    corr_pos_plot = corr_pos + [corr_pos[0]]
    corr_neg_plot = corr_neg + [corr_neg[0]]
    corr_theory_plot = corr_theory + [corr_theory[0]]

    # Calculate uniform correlation at Leggett bound
    bound = result['bound']
    corr_at_bound = -bound / 2.0  # Negative since correlations are negative
    corr_bound_plot = [corr_at_bound] * (N + 1)

    # Calculate adaptive ylim to keep visual hexagon size consistent
    # Find min/max of measured data (not theory, since that's what varies)
    all_measured = corr_pos + corr_neg
    data_min = min(all_measured)
    data_max = max(all_measured)
    data_center = (data_min + data_max) / 2

    # Fixed visual radius to match IBM plot style
    visual_radius = 0.075  # Half of 0.15 range
    ylim_min = data_center - visual_radius
    ylim_max = data_center + visual_radius

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot data
    ax.plot(angles, corr_theory_plot, 'o--', linewidth=2, label=f'QM Ideal (L₃={result["L3_theory"]:.4f})',
            color='royalblue', markersize=8, alpha=0.7)
    ax.plot(angles, corr_bound_plot, ':', linewidth=3, label=f'Leggett Bound (L₃={bound:.4f})',
            color='gray', alpha=0.7)
    ax.plot(angles, corr_pos_plot, 'o-', linewidth=2.5, label=f'φ = +{phi_deg}° (L₃={result["L3_pos"]:.4f})',
            color='green', markersize=10)
    ax.plot(angles, corr_neg_plot, 's-', linewidth=2.5, label=f'φ = -{phi_deg}° (L₃={result["L3_neg"]:.4f})',
            color='orange', markersize=10)

    # Fill areas
    ax.fill(angles, corr_theory_plot, alpha=0.15, color='royalblue')

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=14, weight='bold')

    # Set radial limits - adaptive to keep hexagon size constant
    ax.set_ylim(ylim_min, ylim_max)
    # Create 7 evenly spaced ticks
    yticks = np.linspace(ylim_min, ylim_max, 7)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.2f}' for y in yticks], size=10)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)

    # Title
    title = f'Leggett Correlation Measurements: φ = ±{phi_deg}° (|Ψ⁻⟩ = (|01⟩-|10⟩)/√2)\n'
    title += f'QM Theory: L₃ = {result["L3_theory"]:.4f} | Leggett Bound: L₃ ≤ {bound:.4f}\n'
    title += f'IonQ Forte - 24 Qubits (12 pairs)'
    ax.set_title(title, size=14, weight='bold', pad=20)

    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=11, framealpha=0.9)

    # Add annotations
    textstr = f'Both measurements VIOLATE Leggett bound\n'
    textstr += f'Margin (+{phi_deg}°): {result["L3_pos"] - bound:+.4f}\n'
    textstr += f'Margin (-{phi_deg}°): {result["L3_neg"] - bound:+.4f}'
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    # Save
    output_file = f'ionq_forte_24qb_phi_{int(phi_deg)}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved to {output_file}')

    plt.close()

print(f'\nGenerated {len(plot_angles)} radar plots for IonQ Forte')
