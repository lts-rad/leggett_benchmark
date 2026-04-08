import os
#!/usr/bin/env python3
"""
Combined radar plots: IBM BEST_LAYOUT and IonQ Forte on same plot for each angle.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Set up high-quality rendering
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica Neue', 'Helvetica', 'Arial']
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.labelweight'] = 'light'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Labels
labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Physical qubit pairs for each dataset
# IBM BEST_LAYOUT (newbits): [86,87, 113,114, 99,115, 60,61, 37,45, 46,47, 11,18, 147,148, 100,101, 40,41, 97,107, 84,85]
# Pairs 0-5 (+phi): (86,87), (113,114), (99,115), (60,61), (37,45), (46,47)
# Pairs 6-11 (-phi): (11,18), (147,148), (100,101), (40,41), (97,107), (84,85)
ibm_pairs_pos = ['86-87', '113-114', '99-115', '60-61', '37-45', '46-47']
ibm_pairs_neg = ['11-18', '147-148', '100-101', '40-41', '97-107', '84-85']

# IonQ Forte: virtual = physical, pairs 0-5 (+phi), pairs 6-11 (-phi)
ionq_pairs_pos = ['0-1', '2-3', '4-5', '6-7', '8-9', '10-11']
ionq_pairs_neg = ['12-13', '14-15', '16-17', '18-19', '20-21', '22-23']

# ========== Load Data ==========

# IBM Pittsburgh BEST_LAYOUT
with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT_newbits.json', 'r') as f:
    data_ibm = json.load(f)

# IonQ Forte
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

# Common angles (4 evenly spaced)
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


def normalize_data(values, center, outer, power=1.0):
    """Normalization with custom range per subplot.
    center: correlation value at center (e.g., -1.0)
    outer: correlation value at outer edge
    power: exponent for non-linear scaling (1.0 = linear, 3.0 = cubic)
    """
    result = []
    for v in values:
        linear = (v - center) / (outer - center)
        # Apply power transformation - expands near center, compresses near edge
        normalized = linear ** power
        result.append(normalized)
    return result


def style_combined_radar(ax, ibm_pos, ibm_neg, ionq_pos, ionq_neg,
                         theory, bound_plot, phi_deg,
                         ibm_L3_pos, ibm_L3_neg, ionq_L3_pos, ionq_L3_neg,
                         L3_theory, bound):
    """Create combined radar plot with IBM and IonQ data"""

    qm_val = theory[0]
    leggett_val = bound_plot[0]

    # Compute range for this subplot
    all_data = ibm_pos + ibm_neg + ionq_pos + ionq_neg
    data_max = max(all_data)  # least negative (closest to 0)

    # For ±15°, ±25°, ±30°: use -1.00 as center, linear scaling
    # For ±45°, ±60°: use -1.00 as center, cubic scaling to expand blue hexagon
    center = -1.00
    outer = data_max + 0.02

    # Use cubic (x³) scaling for larger angles to expand the blue hexagon region
    if phi_deg in [45, 60]:
        power = 3.0  # cubic
    else:
        power = 1.0  # linear

    # Normalize all data with per-subplot range
    norm_theory = normalize_data(theory, center, outer, power)
    norm_bound = normalize_data(bound_plot, center, outer, power)
    norm_ibm_pos = normalize_data(ibm_pos, center, outer, power)
    norm_ibm_neg = normalize_data(ibm_neg, center, outer, power)
    norm_ionq_pos = normalize_data(ionq_pos, center, outer, power)
    norm_ionq_neg = normalize_data(ionq_neg, center, outer, power)

    # QM prediction - dashed blue line
    qm_line, = ax.plot(angles, norm_theory, '--', linewidth=2.5,
                       dashes=(8, 4), color=color_qm, alpha=0.95, zorder=5)

    # Leggett bound - dashed gray line
    leggett_line, = ax.plot(angles, norm_bound, '--', linewidth=2.5,
                            dashes=(4, 2), color=color_leggett, alpha=0.9, zorder=4)

    # IBM +phi (dark red)
    for i in range(3):
        ax.plot(angles, norm_ibm_pos, '-', linewidth=6-i*1.5,
                color=color_ibm_pos, alpha=0.06, zorder=6)
    ibm_pos_line, = ax.plot(angles, norm_ibm_pos, '-', linewidth=2.5,
                            color=color_ibm_pos, alpha=0.95, zorder=8)
    ax.plot(angles, norm_ibm_pos, 'o', markersize=8,
            markerfacecolor=color_ibm_pos, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # IBM -phi (light red)
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

    # Add physical qubit pair labels at each data point
    # Use path_effects for white outline around text
    text_outline = [pe.withStroke(linewidth=2, foreground='white')]

    for i in range(6):
        # IBM +phi: offset up-right
        ax.annotate(ibm_pairs_pos[i], (angles[i], norm_ibm_pos[i]),
                    fontsize=6, fontweight='bold', color=color_ibm_pos, ha='left', va='bottom',
                    xytext=(4, 8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)
        # IBM -phi: offset down-right
        ax.annotate(ibm_pairs_neg[i], (angles[i], norm_ibm_neg[i]),
                    fontsize=6, fontweight='bold', color=color_ibm_neg, ha='left', va='top',
                    xytext=(4, -8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)
        # IonQ +phi: offset up-left
        ax.annotate(ionq_pairs_pos[i], (angles[i], norm_ionq_pos[i]),
                    fontsize=6, fontweight='bold', color=color_ionq_pos, ha='right', va='bottom',
                    xytext=(-4, 8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)
        # IonQ -phi: offset down-left
        ax.annotate(ionq_pairs_neg[i], (angles[i], norm_ionq_neg[i]),
                    fontsize=6, fontweight='bold', color=color_ionq_neg, ha='right', va='top',
                    xytext=(-4, -8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, weight='semibold', color=color_text)

    # Grid setup - find actual max of normalized data and set ylim accordingly
    all_norm = norm_ibm_pos + norm_ibm_neg + norm_ionq_pos + norm_ionq_neg
    max_norm = max(all_norm)
    ylim_max = max_norm * 1.05  # 5% padding

    ax.set_ylim(0, ylim_max)

    # Calculate tick positions that span from 0 to ylim_max
    # and calculate corresponding correlation values
    tick_positions = [0, ylim_max * 0.25, ylim_max * 0.5, ylim_max * 0.75, ylim_max]
    # Reverse the power transform to get actual correlation values
    tick_labels = []
    for t in tick_positions:
        # t = ((v - center) / (outer - center))^power
        # so (v - center) / (outer - center) = t^(1/power)
        linear_t = t ** (1.0 / power) if power != 1.0 else t
        corr_val = center + linear_t * (outer - center)
        tick_labels.append(f'{corr_val:.2f}')
    ax.set_rticks(tick_positions)
    ax.set_yticklabels(tick_labels, size=9, color='#444444', fontweight='medium')

    # Grid styling
    ax.grid(True, linestyle='-', alpha=0.35, linewidth=1.0, color='#BDC3C7')
    ax.spines['polar'].set_color('#D5DBDB')
    ax.spines['polar'].set_linewidth(1.5)

    # Title (just the angle) - use varphi (ϕ U+03D5)
    ax.set_title(f'ϕ = ±{phi_deg}°', size=14, weight='bold', pad=15, color=color_text)

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


# Create figure with 3x2 grid layout - all same size
fig = plt.figure(figsize=(20, 24), facecolor='white')

for phi_deg in common_angles:
    # Get data
    ibm_entry_pos = get_ibm_data(phi_deg)
    ibm_entry_neg = get_ibm_data(-phi_deg)
    ionq_entry = get_ionq_data(phi_deg)

    if ibm_entry_pos is None or ionq_entry is None:
        print(f"Missing data for phi={phi_deg}")
        continue

    # IBM data
    ibm_corr_pos = ibm_entry_pos['correlations']
    ibm_corr_neg = ibm_entry_neg['correlations'] if ibm_entry_neg else ibm_corr_pos
    ibm_L3_pos = ibm_entry_pos['L3']
    ibm_L3_neg = ibm_entry_neg['L3'] if ibm_entry_neg else ibm_L3_pos

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

    # Create subplot using standard grid
    idx = common_angles.index(phi_deg)
    ax = plt.subplot(3, 2, idx + 1, projection='polar', facecolor='#FAFBFC')
    style_combined_radar(ax, ibm_pos_plot, ibm_neg_plot, ionq_pos_plot, ionq_neg_plot,
                         theory_plot, bound_plot, phi_deg,
                         ibm_L3_pos, ibm_L3_neg, ionq_L3_pos, ionq_L3_neg,
                         L3_theory, bound)

# Layout and save
plt.tight_layout()
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_all_angles_combo_24qb.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved combined comparison to {output_file}')

plt.show()
plt.close()
