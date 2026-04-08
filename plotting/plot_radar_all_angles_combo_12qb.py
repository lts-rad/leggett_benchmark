import os
#!/usr/bin/env python3
"""
Combined radar plots: IBM 12qb BEST_LAYOUT and IonQ Forte 24qb on same plot for each angle.
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

# IBM Pittsburgh 12qb BEST_LAYOUT
with open('leggett_results_ibm_ibm_pittsburgh_sequential_12qb_BEST_LAYOUT.json', 'r') as f:
    data_ibm = json.load(f)

# IonQ Forte 24qb
with open(os.path.join(_DATA_DIR, 'leggett_results_ionq_forte_sequential_24qb.json') , 'r') as f:
    data_ionq = json.load(f)

# Color palette
color_qm = '#3498DB'           # Bright blue for QM theory
color_leggett = '#7F8C8D'      # Gray for Leggett bound

# IonQ colors (yellow/orange)
color_ionq_pos = '#F39C12'     # Deep amber/gold for +phi
color_ionq_neg = '#F7DC6F'     # Light yellow for -phi

# IBM colors (purple shades)
color_ibm_pos = '#6C3483'      # Dark purple for +phi
color_ibm_neg = '#A569BD'      # Light purple for -phi

color_text = '#2C3E50'         # Dark blue-gray for text

# Common angles (all 5)
common_angles = [15, 25, 30, 45, 60]


def get_ibm_data(phi_deg, positive=True):
    """Get IBM 12qb data for a specific angle.
    The 12qb file uses phi_rad to distinguish +/- angles.
    """
    for entry in data_ibm:
        if entry['phi_deg'] == phi_deg:
            if positive and entry['phi_rad'] > 0:
                return entry
            elif not positive and entry['phi_rad'] < 0:
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
    target_qm = 0.4
    target_leggett = 0.7

    result = []
    for v in values:
        if v <= qm_val:
            t = (v - (-1)) / (qm_val - (-1))
            result.append(t * target_qm)
        elif v <= leggett_val:
            t = (v - qm_val) / (leggett_val - qm_val)
            result.append(target_qm + t * (target_leggett - target_qm))
        else:
            t = (v - leggett_val) / (leggett_val - qm_val)
            result.append(target_leggett + t * (target_leggett - target_qm))
    return result


def style_combined_radar(ax, ibm_pos, ibm_neg, ionq_pos, ionq_neg,
                         theory, bound_plot, phi_deg,
                         ibm_L3_pos, ibm_L3_neg, ionq_L3_pos, ionq_L3_neg,
                         L3_theory, bound):
    """Create combined radar plot with IBM and IonQ data"""

    qm_val = theory[0]
    leggett_val = bound_plot[0]

    # Normalize all data
    norm_theory = normalize_data(theory, qm_val, leggett_val)
    norm_bound = normalize_data(bound_plot, qm_val, leggett_val)
    norm_ibm_pos = normalize_data(ibm_pos, qm_val, leggett_val)
    norm_ibm_neg = normalize_data(ibm_neg, qm_val, leggett_val)
    norm_ionq_pos = normalize_data(ionq_pos, qm_val, leggett_val)
    norm_ionq_neg = normalize_data(ionq_neg, qm_val, leggett_val)

    # QM prediction - dashed blue line
    qm_line, = ax.plot(angles, norm_theory, '--', linewidth=2.5,
                       dashes=(8, 4), color=color_qm, alpha=0.95, zorder=5)

    # Leggett bound - dashed gray line
    leggett_line, = ax.plot(angles, norm_bound, '--', linewidth=2.5,
                            dashes=(4, 2), color=color_leggett, alpha=0.9, zorder=4)

    # IBM +phi (dark purple)
    for i in range(3):
        ax.plot(angles, norm_ibm_pos, '-', linewidth=6-i*1.5,
                color=color_ibm_pos, alpha=0.06, zorder=6)
    ibm_pos_line, = ax.plot(angles, norm_ibm_pos, '-', linewidth=2.5,
                            color=color_ibm_pos, alpha=0.95, zorder=8)
    ax.plot(angles, norm_ibm_pos, 'o', markersize=8,
            markerfacecolor=color_ibm_pos, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # IBM -phi (light purple)
    for i in range(3):
        ax.plot(angles, norm_ibm_neg, '-', linewidth=6-i*1.5,
                color=color_ibm_neg, alpha=0.06, zorder=6)
    ibm_neg_line, = ax.plot(angles, norm_ibm_neg, '-', linewidth=2.5,
                            color=color_ibm_neg, alpha=0.95, zorder=8)
    ax.plot(angles, norm_ibm_neg, 's', markersize=7,
            markerfacecolor=color_ibm_neg, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # IonQ +phi (amber/gold)
    for i in range(3):
        ax.plot(angles, norm_ionq_pos, '-', linewidth=6-i*1.5,
                color=color_ionq_pos, alpha=0.06, zorder=6)
    ionq_pos_line, = ax.plot(angles, norm_ionq_pos, '-', linewidth=2.5,
                             color=color_ionq_pos, alpha=0.95, zorder=8)
    ax.plot(angles, norm_ionq_pos, '^', markersize=8,
            markerfacecolor=color_ionq_pos, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # IonQ -phi (light yellow)
    for i in range(3):
        ax.plot(angles, norm_ionq_neg, '-', linewidth=6-i*1.5,
                color=color_ionq_neg, alpha=0.06, zorder=6)
    ionq_neg_line, = ax.plot(angles, norm_ionq_neg, '-', linewidth=2.5,
                             color=color_ionq_neg, alpha=0.95, zorder=8)
    ax.plot(angles, norm_ionq_neg, 'v', markersize=7,
            markerfacecolor=color_ionq_neg, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, weight='semibold', color=color_text)

    # Grid setup
    ax.set_ylim(0, 1.2)

    # Use fewer, evenly-spaced tick marks with clear labels
    # Label every 4th circle: 0, 0.4, 0.8, 1.2 (normalized positions)
    # These correspond to: -1, QM theory, Leggett bound region, outer
    ax.set_rticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_yticklabels(['-1.00', '', f'{qm_val:.2f}', '', f'{leggett_val:.2f}', '', '-0.70'],
                       size=10, color='#444444', fontweight='medium')

    # Grid styling
    ax.grid(True, linestyle='-', alpha=0.35, linewidth=1.0, color='#BDC3C7')
    ax.spines['polar'].set_color('#D5DBDB')
    ax.spines['polar'].set_linewidth(1.5)

    # Title (just the angle)
    ax.set_title(f'φ = ±{phi_deg}°', size=14, weight='bold', pad=15, color=color_text)

    # Legend with system names and L3 values only
    legend = ax.legend(
        [qm_line, leggett_line, ibm_pos_line, ibm_neg_line, ionq_pos_line, ionq_neg_line],
        [f'QM Theory (L₃={L3_theory:.3f})',
         f'Leggett Bound ({bound:.3f})',
         f'IBM +{phi_deg}° (L₃={ibm_L3_pos:.3f})',
         f'IBM −{phi_deg}° (L₃={ibm_L3_neg:.3f})',
         f'IonQ +{phi_deg}° (L₃={ionq_L3_pos:.3f})',
         f'IonQ −{phi_deg}° (L₃={ionq_L3_neg:.3f})'],
        loc='upper left', bbox_to_anchor=(1.02, 1.0),
        fontsize=12, frameon=True, fancybox=False,
        shadow=False, framealpha=0.95,
        edgecolor='#E8E8E8', facecolor='white',
        borderpad=0.8, labelspacing=0.5,
        handlelength=2.0
    )
    legend.get_frame().set_linewidth(0.8)


# Create figure: 3x2 grid for 5 angles (6th spot empty)
fig = plt.figure(figsize=(20, 24), facecolor='white')

for idx, phi_deg in enumerate(common_angles):
    # Get data - IBM 12qb uses phi_rad to distinguish +/-
    ibm_entry_pos = get_ibm_data(phi_deg, positive=True)
    ibm_entry_neg = get_ibm_data(phi_deg, positive=False)
    ionq_entry = get_ionq_data(phi_deg)

    if ibm_entry_pos is None or ibm_entry_neg is None or ionq_entry is None:
        print(f"Missing data for phi={phi_deg}")
        continue

    # IBM data
    ibm_corr_pos = ibm_entry_pos['correlations']
    ibm_corr_neg = ibm_entry_neg['correlations']
    ibm_L3_pos = ibm_entry_pos['L3']
    ibm_L3_neg = ibm_entry_neg['L3']

    # IonQ data
    ionq_corr_pos = ionq_entry['correlations_pos']
    ionq_corr_neg = ionq_entry['correlations_neg']
    ionq_L3_pos = ionq_entry['L3_pos']
    ionq_L3_neg = ionq_entry['L3_neg']

    # Theory and bounds (same for both)
    corr_theory = ibm_entry_pos['correlations_theory']
    L3_theory = ibm_entry_pos['L3_theory']
    bound = ibm_entry_pos['bound']
    corr_bound = -bound / 2.0

    # Close the plots (add first point at end)
    ibm_pos_plot = ibm_corr_pos + [ibm_corr_pos[0]]
    ibm_neg_plot = ibm_corr_neg + [ibm_corr_neg[0]]
    ionq_pos_plot = ionq_corr_pos + [ionq_corr_pos[0]]
    ionq_neg_plot = ionq_corr_neg + [ionq_corr_neg[0]]
    theory_plot = corr_theory + [corr_theory[0]]
    bound_plot = [corr_bound] * (N + 1)

    # Create subplot (3x2 grid)
    ax = plt.subplot(3, 2, idx + 1, projection='polar', facecolor='#FAFBFC')
    style_combined_radar(ax, ibm_pos_plot, ibm_neg_plot, ionq_pos_plot, ionq_neg_plot,
                         theory_plot, bound_plot, phi_deg,
                         ibm_L3_pos, ibm_L3_neg, ionq_L3_pos, ionq_L3_neg,
                         L3_theory, bound)

plt.tight_layout()

# Save
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_all_angles_combo_12qb.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved combined comparison to {output_file}')

plt.close()
