import os
#!/usr/bin/env python3
"""
Radar plots for Quantinuum H2-1 Emulator results.
Shows 6 correlations per angle with +phi and -phi data.
Based on plot_radar_rotated_vs_new.py style.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Set up high-quality rendering
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica Neue', 'Helvetica', 'Arial']
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.labelweight'] = 'light'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Labels for 6 correlation types
labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Qubit pairs for Quantinuum H2 (logical pairs, no physical mapping needed for trapped ion)
pairs_pos = ['0-1', '2-3', '4-5', '6-7', '8-9', '10-11']
pairs_neg = ['12-13', '14-15', '16-17', '18-19', '20-21', '22-23']

# Load data
with open('leggett_results_azure_sim_h2-1_sequential_24qb.json', 'r') as f:
    data = json.load(f)

# Color palette
color_qm = '#3498DB'           # Bright blue for QM theory
color_leggett = '#7F8C8D'      # Gray for Leggett bound
color_pos = '#E74C3C'          # Red for +phi
color_neg = '#9B59B6'          # Purple for -phi
color_text = '#2C3E50'         # Dark blue-gray for text

# Common angles
common_angles = [15, 25, 30, 45, 60]


def get_data_by_phi(phi_deg, positive=True):
    """Get data for a specific angle and sign."""
    for entry in data:
        if positive and entry['phi_deg'] == phi_deg:
            return entry
        elif not positive and entry['phi_deg'] == -phi_deg:
            return entry
    return None


def normalize_data(values, center, outer, power=1.0):
    """Normalization with custom range per subplot."""
    result = []
    for v in values:
        linear = (v - center) / (outer - center)
        normalized = linear ** power
        result.append(normalized)
    return result


def create_radar_subplot(ax, phi_deg):
    """Create radar subplot for a specific angle."""

    entry_pos = get_data_by_phi(phi_deg, positive=True)
    entry_neg = get_data_by_phi(phi_deg, positive=False)

    if entry_pos is None:
        print(f"Missing data for phi={phi_deg}")
        return

    # Get correlations
    corr_pos = entry_pos['correlations']
    corr_neg = entry_neg['correlations'] if entry_neg else corr_pos
    corr_theory = entry_pos['correlations_theory']

    L3_pos = entry_pos['L3']
    L3_neg = entry_neg['L3'] if entry_neg else L3_pos
    L3_theory = entry_pos['L3_theory']
    bound = entry_pos['bound']
    corr_bound = -bound / 2.0

    # Close the plots (add first value at end)
    corr_pos_plot = corr_pos + [corr_pos[0]]
    corr_neg_plot = corr_neg + [corr_neg[0]]
    theory_plot = corr_theory + [corr_theory[0]]
    bound_plot = [corr_bound] * (N + 1)

    # Compute range for normalization
    all_data = corr_pos + corr_neg
    data_max = max(all_data)

    center = -1.00
    outer = data_max + 0.02

    # Use cubic scaling for larger angles
    if phi_deg in [45, 60]:
        power = 3.0
    else:
        power = 1.0

    # Normalize all data
    norm_theory = normalize_data(theory_plot, center, outer, power)
    norm_bound = normalize_data(bound_plot, center, outer, power)
    norm_pos = normalize_data(corr_pos_plot, center, outer, power)
    norm_neg = normalize_data(corr_neg_plot, center, outer, power)

    text_outline = [pe.withStroke(linewidth=2, foreground='white')]

    # QM Theory - dashed blue line
    qm_line, = ax.plot(angles, norm_theory, '--', linewidth=2.5,
                       dashes=(8, 4), color=color_qm, alpha=0.95, zorder=5)

    # Leggett bound - dashed gray line
    leggett_line, = ax.plot(angles, norm_bound, '--', linewidth=2.5,
                            dashes=(4, 2), color=color_leggett, alpha=0.9, zorder=4)

    # +phi data (red)
    for i in range(3):
        ax.plot(angles, norm_pos, '-', linewidth=6-i*1.5,
                color=color_pos, alpha=0.06, zorder=6)
    pos_line, = ax.plot(angles, norm_pos, '-', linewidth=2.5,
                        color=color_pos, alpha=0.95, zorder=8)
    ax.plot(angles, norm_pos, 'o', markersize=8,
            markerfacecolor=color_pos, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # -phi data (purple)
    for i in range(3):
        ax.plot(angles, norm_neg, '-', linewidth=6-i*1.5,
                color=color_neg, alpha=0.06, zorder=6)
    neg_line, = ax.plot(angles, norm_neg, '-', linewidth=2.5,
                        color=color_neg, alpha=0.95, zorder=8)
    ax.plot(angles, norm_neg, 's', markersize=7,
            markerfacecolor=color_neg, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # Add qubit pair labels
    for i in range(6):
        # +phi labels
        ax.annotate(pairs_pos[i], (angles[i], norm_pos[i]),
                    fontsize=6, fontweight='bold', color=color_pos, ha='left', va='bottom',
                    xytext=(4, 8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)
        # -phi labels
        ax.annotate(pairs_neg[i], (angles[i], norm_neg[i]),
                    fontsize=6, fontweight='bold', color=color_neg, ha='left', va='top',
                    xytext=(4, -8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, weight='semibold', color=color_text)

    # Grid setup - find actual max of normalized data and set ylim accordingly
    all_norm = norm_pos + norm_neg + norm_theory + norm_bound
    max_norm = max(all_norm)
    ylim_max = max_norm * 1.05

    ax.set_ylim(0, ylim_max)

    # Calculate tick positions and labels
    tick_positions = [0, ylim_max * 0.25, ylim_max * 0.5, ylim_max * 0.75, ylim_max]
    tick_labels = []
    for t in tick_positions:
        linear_t = t ** (1.0 / power) if power != 1.0 else t
        corr_val = center + linear_t * (outer - center)
        tick_labels.append(f'{corr_val:.2f}')
    ax.set_rticks(tick_positions)
    ax.set_yticklabels(tick_labels, size=9, color='#444444', fontweight='medium')

    # Grid styling
    ax.grid(True, linestyle='-', alpha=0.35, linewidth=1.0, color='#BDC3C7')
    ax.spines['polar'].set_color('#D5DBDB')
    ax.spines['polar'].set_linewidth(1.5)

    # Title
    ax.set_title(f'ϕ = ±{phi_deg}°', size=14, weight='bold', pad=15, color=color_text)

    # Legend
    legend = ax.legend(
        [qm_line, leggett_line, pos_line, neg_line],
        [f'QM Theory (L₃={L3_theory:.3f})',
         f'Leggett Bound ({bound:.3f})',
         f'+{phi_deg}° (L₃={L3_pos:.3f})',
         f'−{phi_deg}° (L₃={L3_neg:.3f})'],
        loc='upper left', bbox_to_anchor=(1.02, 1.0),
        fontsize=10, frameon=True, fancybox=False,
        shadow=False, framealpha=0.95,
        edgecolor='#E8E8E8', facecolor='white',
        borderpad=0.8, labelspacing=0.5,
        handlelength=2.0
    )
    legend.get_frame().set_linewidth(0.8)


# Create figure with 3x2 grid
fig = plt.figure(figsize=(20, 24), facecolor='white')

for idx, phi_deg in enumerate(common_angles):
    ax = plt.subplot(3, 2, idx + 1, projection='polar', facecolor='#FAFBFC')
    create_radar_subplot(ax, phi_deg)

# Main title
fig.suptitle('Quantinuum H2-1 Emulator (Sequential 24qb)\n1000 Shots',
             size=18, weight='bold', color=color_text, y=0.98)

plt.tight_layout()

output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_quantinuum_h2_sequential_24qb.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved to {output_file}')

plt.close()
