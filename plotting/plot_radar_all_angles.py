import os
#!/usr/bin/env python3
"""
Side-by-side radar plots for IBM Pittsburgh and IonQ Forte at ALL angles.
Publication-quality visualization.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Set up high-quality rendering
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.labelweight'] = 'light'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Labels
labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ========== Load Data ==========

# IBM Pittsburgh BEST_LAYOUT (multi-angle)
with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT.json', 'r') as f:
    data_ibm = json.load(f)

# IonQ Forte (multi-angle)
with open(os.path.join(_DATA_DIR, 'leggett_results_ionq_forte_sequential_24qb.json') , 'r') as f:
    data_ionq = json.load(f)

# Color palette
color_qm = '#3498DB'           # Bright blue for QM
color_leggett = '#7F8C8D'      # Sophisticated gray for classical bound
color_pos = '#F39C12'          # Deep amber/gold
color_neg = '#F7DC6F'          # Light yellow
color_text = '#2C3E50'         # Dark blue-gray for text

# Common angles (matching between IBM and IonQ)
common_angles = [15, 30, 45, 60]

def get_ibm_data(phi_deg):
    """Get IBM data for a specific angle"""
    for entry in data_ibm:
        if entry['phi_deg'] == phi_deg:
            return entry
    return None

def get_ionq_data(phi_deg):
    """Get IonQ data for a specific angle"""
    for entry in data_ionq:
        if entry['phi_deg'] == phi_deg:
            return entry
    return None

def normalize_data(values, qm_val, leggett_val):
    """Normalize data with -1 at center, QM at fixed 0.4, Leggett at fixed 0.7"""
    # Piecewise linear transform to fix all three points:
    # -1 → 0 (center)
    # QM → 0.4 (fixed inner hexagon)
    # Leggett → 0.7 (fixed outer hexagon)
    target_qm = 0.4
    target_leggett = 0.7

    result = []
    for v in values:
        if v <= qm_val:
            # Between -1 and QM: linear from 0 to target_qm
            # -1 → 0, qm_val → target_qm
            t = (v - (-1)) / (qm_val - (-1))  # 0 to 1
            result.append(t * target_qm)
        elif v <= leggett_val:
            # Between QM and Leggett: linear from target_qm to target_leggett
            t = (v - qm_val) / (leggett_val - qm_val)  # 0 to 1
            result.append(target_qm + t * (target_leggett - target_qm))
        else:
            # Beyond Leggett: continue linearly
            t = (v - leggett_val) / (leggett_val - qm_val)
            result.append(target_leggett + t * (target_leggett - target_qm))
    return result

def style_radar_plot(ax, corr_pos_plot, corr_neg_plot, corr_theory_plot, corr_bound_plot,
                     L3_pos, L3_neg, L3_theory, bound, phi_deg, title_name):
    """Apply styling to a radar plot"""

    # Get raw values for normalization
    qm_val = corr_theory_plot[0]
    leggett_val = corr_bound_plot[0]

    # Normalize all data to fixed visual scale
    norm_theory = normalize_data(corr_theory_plot, qm_val, leggett_val)
    norm_bound = normalize_data(corr_bound_plot, qm_val, leggett_val)
    norm_pos = normalize_data(corr_pos_plot, qm_val, leggett_val)
    norm_neg = normalize_data(corr_neg_plot, qm_val, leggett_val)

    # QM prediction - clean dashed blue line
    qm_line, = ax.plot(angles, norm_theory, '--', linewidth=2.5,
             dashes=(8, 4), color=color_qm, alpha=0.95, zorder=5)

    # Leggett bound - clean dashed line
    leggett_line, = ax.plot(angles, norm_bound, '--', linewidth=2.5,
             dashes=(4, 2), color=color_leggett, alpha=0.9, zorder=4)

    # Measurement lines with glow effect
    # Positive phi
    for i in range(4):
        ax.plot(angles, norm_pos, '-', linewidth=8-i*1.5,
                color=color_pos, alpha=0.08, zorder=6)
    pos_line, = ax.plot(angles, norm_pos, '-', linewidth=3,
             color=color_pos, alpha=0.95, zorder=8)
    ax.plot(angles, norm_pos, 'o', markersize=10,
             markerfacecolor=color_pos, markeredgecolor='white',
             markeredgewidth=2, alpha=0.95, zorder=9)

    # Negative phi
    for i in range(4):
        ax.plot(angles, norm_neg, '-', linewidth=8-i*1.5,
                color=color_neg, alpha=0.08, zorder=6)
    neg_line, = ax.plot(angles, norm_neg, '-', linewidth=3,
             color=color_neg, alpha=0.95, zorder=8)
    ax.plot(angles, norm_neg, 's', markersize=9,
             markerfacecolor=color_neg, markeredgecolor='white',
             markeredgewidth=2, alpha=0.95, zorder=9)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, weight='semibold', color=color_text)

    # Fixed positions: -1 at 0, QM at 0.4, Leggett at 0.7
    # Many more grid circles for better resolution
    ax.set_ylim(0, 1.3)  # Tighter to show outer label

    # Calculate what correlation value the outer circle (1.2) represents
    # Reverse the piecewise transform for norm > 0.7:
    # t = (norm - 0.7) / 0.3, v = leggett + t * (leggett - qm)
    outer_norm = 1.2
    t_outer = (outer_norm - 0.7) / 0.3
    outer_corr = leggett_val + t_outer * (leggett_val - qm_val)

    # Grid circles every 0.1 from 0 to 1.2
    ax.set_rticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    # Labels at key positions: -1 (0), QM (0.4), Leggett (0.7), outer (1.2)
    ax.set_yticklabels(['-1', '', '', '', f'{abs(qm_val):.2f}', '', '', f'{abs(leggett_val):.2f}', '', '', '', '', f'{abs(outer_corr):.2f}'], size=10, color='#666666')

    # Grid
    ax.grid(True, linestyle='-', alpha=0.35, linewidth=1.0, color='#BDC3C7')
    ax.spines['polar'].set_color('#D5DBDB')
    ax.spines['polar'].set_linewidth(1.5)

    # Title
    ax.set_title(title_name, size=14, weight='bold', pad=15, color=color_text)

    # Legend
    legend = ax.legend(
        [qm_line, leggett_line, pos_line, neg_line],
        [f'QM (L₃={L3_theory:.3f})',
         f'Leggett (L₃={bound:.3f})',
         f'φ=+{phi_deg}° (L₃={L3_pos:.3f})',
         f'φ=−{phi_deg}° (L₃={L3_neg:.3f})'],
        loc='upper left', bbox_to_anchor=(1.02, 1.0),
        fontsize=8, frameon=True, fancybox=False,
        shadow=False, framealpha=0.95,
        edgecolor='#E8E8E8', facecolor='white',
        borderpad=0.8, labelspacing=0.6,
        handlelength=2.0
    )
    legend.get_frame().set_linewidth(0.8)

# Create figure: 4 angles x 2 platforms = 8 subplots
# Increased size for better resolution
fig = plt.figure(figsize=(32, 28), facecolor='white')

for idx, phi_deg in enumerate(common_angles):
    # Get data
    ibm_entry = get_ibm_data(phi_deg)
    ibm_entry_neg = get_ibm_data(-phi_deg)
    ionq_entry = get_ionq_data(phi_deg)

    if ibm_entry is None or ionq_entry is None:
        print(f"Missing data for phi={phi_deg}")
        continue

    # IBM data
    ibm_corr_pos = ibm_entry['correlations']
    ibm_corr_neg = ibm_entry_neg['correlations'] if ibm_entry_neg else ibm_corr_pos
    ibm_corr_theory = ibm_entry['correlations_theory']
    ibm_bound = ibm_entry['bound']
    ibm_L3_pos = ibm_entry['L3']
    ibm_L3_neg = ibm_entry_neg['L3'] if ibm_entry_neg else ibm_L3_pos
    ibm_L3_theory = ibm_entry['L3_theory']

    # IonQ data
    ionq_corr_pos = ionq_entry['correlations_pos']
    ionq_corr_neg = ionq_entry['correlations_neg']
    ionq_corr_theory = ionq_entry['correlations_theory']
    ionq_bound = ionq_entry['bound']
    ionq_L3_pos = ionq_entry['L3_pos']
    ionq_L3_neg = ionq_entry['L3_neg']
    ionq_L3_theory = ionq_entry['L3_theory']

    # Close plots
    ibm_corr_pos_plot = ibm_corr_pos + [ibm_corr_pos[0]]
    ibm_corr_neg_plot = ibm_corr_neg + [ibm_corr_neg[0]]
    ibm_corr_theory_plot = ibm_corr_theory + [ibm_corr_theory[0]]
    ibm_corr_bound = -ibm_bound / 2.0
    ibm_corr_bound_plot = [ibm_corr_bound] * (N + 1)

    ionq_corr_pos_plot = ionq_corr_pos + [ionq_corr_pos[0]]
    ionq_corr_neg_plot = ionq_corr_neg + [ionq_corr_neg[0]]
    ionq_corr_theory_plot = ionq_corr_theory + [ionq_corr_theory[0]]
    ionq_corr_bound = -ionq_bound / 2.0
    ionq_corr_bound_plot = [ionq_corr_bound] * (N + 1)

    # IBM subplot (left column)
    ax1 = plt.subplot(4, 2, idx*2 + 1, projection='polar', facecolor='#FAFBFC')
    style_radar_plot(ax1, ibm_corr_pos_plot, ibm_corr_neg_plot, ibm_corr_theory_plot,
                     ibm_corr_bound_plot, ibm_L3_pos, ibm_L3_neg, ibm_L3_theory,
                     ibm_bound, phi_deg, f'IBM Pittsburgh BEST (φ=±{phi_deg}°)')

    # IonQ subplot (right column)
    ax2 = plt.subplot(4, 2, idx*2 + 2, projection='polar', facecolor='#FAFBFC')
    style_radar_plot(ax2, ionq_corr_pos_plot, ionq_corr_neg_plot, ionq_corr_theory_plot,
                     ionq_corr_bound_plot, ionq_L3_pos, ionq_L3_neg, ionq_L3_theory,
                     ionq_bound, phi_deg, f'IonQ Forte (φ=±{phi_deg}°)')

# Main title
fig.suptitle('Leggett Inequality Violation: All Angles Comparison',
             fontsize=24, weight='bold', y=0.98, color=color_text)

# Column headers
fig.text(0.28, 0.94, 'IBM Pittsburgh BEST_LAYOUT (Superconducting)', ha='center', fontsize=16,
         color='#7F8C8D', weight='medium')
fig.text(0.72, 0.94, 'IonQ Forte (Trapped-Ion)', ha='center', fontsize=16,
         color='#7F8C8D', weight='medium')

plt.tight_layout(rect=[0, 0.02, 1, 0.92])

# Save
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_all_angles_comparison.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved all-angles comparison to {output_file}')

plt.close()
