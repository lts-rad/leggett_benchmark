import os
#!/usr/bin/env python3
"""
Radar plots for IBM Pittsburgh ALTERNATE layout (top 12 pairs from 64-pair analysis).
Shows results at all tested angles.
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

# ALTERNATE LAYOUT physical qubit pairs (top 12 from 64-pair analysis)
# Pairs 0-5 are +phi, pairs 6-11 are -phi
# From ibm_leggett_test_sequential_24qb_alternate_10k.py:
# [72,73, 86,87, 20,21, 98,111, 12,13, 122,123, 24,25, 144,145, 97,107, 68,69, 19,35, 38,49]
pairs_pos = ['72-73', '86-87', '20-21', '98-111', '12-13', '122-123']
pairs_neg = ['24-25', '144-145', '97-107', '68-69', '19-35', '38-49']

# Load data
with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_ALTERNATE_10k.json', 'r') as f:
    data = json.load(f)

# Color palette
color_qm = '#3498DB'           # Bright blue for QM theory
color_leggett = '#7F8C8D'      # Gray for Leggett bound
color_pos = '#E74C3C'          # Red for +phi
color_neg = '#9B59B6'          # Purple for -phi
color_text = '#2C3E50'         # Dark blue-gray for text


def get_data_for_angle(phi_deg):
    """Get data entry for specific angle."""
    for entry in data:
        if entry['phi_deg'] == phi_deg:
            return entry
    return None


def normalize_data(values, center, outer, power=1.0):
    """Normalization with custom range."""
    result = []
    for v in values:
        linear = (v - center) / (outer - center)
        normalized = linear ** power
        result.append(normalized)
    return result


def create_radar_subplot(ax, phi_deg):
    """Create radar subplot for a specific angle."""

    entry_pos = get_data_for_angle(phi_deg)
    entry_neg = get_data_for_angle(-phi_deg)

    if entry_pos is None:
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

    # Close the plots
    corr_pos_plot = corr_pos + [corr_pos[0]]
    corr_neg_plot = corr_neg + [corr_neg[0]]
    theory_plot = corr_theory + [corr_theory[0]]
    bound_plot = [corr_bound] * 7

    # Compute range for normalization
    all_data = corr_pos + corr_neg
    data_max = max(all_data)

    center = -1.00
    outer = data_max + 0.02

    # Use cubic scaling for larger angles
    power = 3.0 if phi_deg in [45, 60] else 1.0

    # Normalize
    norm_theory = normalize_data(theory_plot, center, outer, power)
    norm_bound = normalize_data(bound_plot, center, outer, power)
    norm_pos = normalize_data(corr_pos_plot, center, outer, power)
    norm_neg = normalize_data(corr_neg_plot, center, outer, power)

    text_outline = [pe.withStroke(linewidth=2, foreground='white')]

    # QM Theory
    qm_line, = ax.plot(angles, norm_theory, '--', linewidth=2.5,
                       dashes=(8, 4), color=color_qm, alpha=0.95, zorder=5)

    # Leggett bound
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

    # Add physical qubit pair labels
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

    # Grid setup
    all_norm = norm_pos + norm_neg
    max_norm = max(all_norm)
    ylim_max = max_norm * 1.05

    ax.set_ylim(0, ylim_max)

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

test_angles = [15, 25, 30, 45, 60]

for idx, phi_deg in enumerate(test_angles):
    ax = plt.subplot(3, 2, idx + 1, projection='polar', facecolor='#FAFBFC')
    create_radar_subplot(ax, phi_deg)

# Main title
fig.suptitle('IBM Pittsburgh ALTERNATE Layout (Top 12 Pairs from 64-Pair Analysis)\n24 Qubits, 10k Shots',
             size=18, weight='bold', color=color_text, y=0.98)

plt.tight_layout()

output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_alternate_layout_24qb.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved to {output_file}')

plt.close()
