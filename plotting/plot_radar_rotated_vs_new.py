import os
#!/usr/bin/env python3
"""
Radar plot comparison: ROTATED (Dec 8) vs NEW (Dec 7) IBM Pittsburgh runs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Set up high-quality rendering
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica Neue', 'Helvetica', 'Arial']
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.labelweight'] = 'light'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Labels for correlation measurements
labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']

# Physical qubit pairs for each dataset
# Non-rotated (New): [67,68, 29,30, 54,55, 93,92, 109,110, 2,3, 13,14, 112,113, 153,152, 75,74, 94,95, 107,97]
# Rotated:           [109,110, 54,55, 13,14, 67,68, 2,3, 94,95, 93,92, 153,152, 29,30, 75,74, 107,97, 112,113]
new_pairs_pos = ['67-68', '29-30', '54-55', '93-92', '109-110', '2-3']
new_pairs_neg = ['13-14', '112-113', '153-152', '75-74', '94-95', '107-97']
rotated_pairs_pos = ['109-110', '54-55', '13-14', '67-68', '2-3', '94-95']
rotated_pairs_neg = ['93-92', '153-152', '29-30', '75-74', '107-97', '112-113']
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ========== Load Data ==========

# ROTATED (Dec 8) - 10k shots
with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT_rotated_10k.json', 'r') as f:
    data_rotated = json.load(f)

# NEW (Dec 7) - 10k shots
with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT_10k.json', 'r') as f:
    data_new = json.load(f)

# Color palette
color_qm = '#3498DB'           # Bright blue for QM theory
color_leggett = '#7F8C8D'      # Gray for Leggett bound

# Rotated colors (purple shades)
color_rotated_pos = '#6B3FA0'  # Dark purple for +phi
color_rotated_neg = '#9B59B6'  # Light purple for -phi

# Optimized 24QB colors (blue shades)
color_new_pos = '#1A5276'      # Dark blue for +phi
color_new_neg = '#2874A6'      # Medium-dark blue for -phi (darker than before)

color_text = '#2C3E50'         # Dark blue-gray for text

# Common angles (4 evenly spaced)
common_angles = [15, 30, 45, 60]


def get_data_by_phi(data, phi_deg, positive=True):
    """Get data for a specific angle and sign."""
    for entry in data:
        if entry['phi_deg'] == phi_deg and ((positive and entry['phi_rad'] > 0) or (not positive and entry['phi_rad'] < 0)):
            return entry
        elif entry['phi_deg'] == phi_deg and entry['phi_deg'] == abs(phi_deg):
            # Handle case where phi_deg might not have sign
            if positive and entry['phi_rad'] > 0:
                return entry
            elif not positive and entry['phi_rad'] < 0:
                return entry
    # Fallback: try matching just the magnitude
    for entry in data:
        if abs(entry['phi_deg']) == abs(phi_deg):
            if positive and entry['phi_rad'] > 0:
                return entry
            elif not positive and entry['phi_rad'] < 0:
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


def style_comparison_radar(ax, rotated_pos, rotated_neg, new_pos, new_neg,
                           theory, bound_plot, phi_deg,
                           rotated_L3_pos, rotated_L3_neg, new_L3_pos, new_L3_neg,
                           L3_theory, bound):
    """Create comparison radar plot."""

    qm_val = theory[0]
    leggett_val = bound_plot[0]

    # Compute range for this subplot
    all_data = rotated_pos + rotated_neg + new_pos + new_neg
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
    norm_rotated_pos = normalize_data(rotated_pos, center, outer, power)
    norm_rotated_neg = normalize_data(rotated_neg, center, outer, power)
    norm_new_pos = normalize_data(new_pos, center, outer, power)
    norm_new_neg = normalize_data(new_neg, center, outer, power)

    # QM prediction - dashed blue line
    qm_line, = ax.plot(angles, norm_theory, '--', linewidth=2.5,
                       dashes=(8, 4), color=color_qm, alpha=0.95, zorder=5)

    # Leggett bound - dashed gray line
    leggett_line, = ax.plot(angles, norm_bound, '--', linewidth=2.5,
                            dashes=(4, 2), color=color_leggett, alpha=0.9, zorder=4)

    # ROTATED +phi (dark purple)
    for i in range(3):
        ax.plot(angles, norm_rotated_pos, '-', linewidth=6-i*1.5,
                color=color_rotated_pos, alpha=0.06, zorder=6)
    rotated_pos_line, = ax.plot(angles, norm_rotated_pos, '-', linewidth=2.5,
                                color=color_rotated_pos, alpha=0.95, zorder=8)
    ax.plot(angles, norm_rotated_pos, 'o', markersize=8,
            markerfacecolor=color_rotated_pos, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # ROTATED -phi (light purple)
    for i in range(3):
        ax.plot(angles, norm_rotated_neg, '-', linewidth=6-i*1.5,
                color=color_rotated_neg, alpha=0.06, zorder=6)
    rotated_neg_line, = ax.plot(angles, norm_rotated_neg, '-', linewidth=2.5,
                                color=color_rotated_neg, alpha=0.95, zorder=8)
    ax.plot(angles, norm_rotated_neg, 's', markersize=7,
            markerfacecolor=color_rotated_neg, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # NEW +phi (dark blue)
    for i in range(3):
        ax.plot(angles, norm_new_pos, '-', linewidth=6-i*1.5,
                color=color_new_pos, alpha=0.06, zorder=6)
    new_pos_line, = ax.plot(angles, norm_new_pos, '-', linewidth=2.5,
                            color=color_new_pos, alpha=0.95, zorder=8)
    ax.plot(angles, norm_new_pos, '^', markersize=8,
            markerfacecolor=color_new_pos, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # NEW -phi (light blue)
    for i in range(3):
        ax.plot(angles, norm_new_neg, '-', linewidth=6-i*1.5,
                color=color_new_neg, alpha=0.06, zorder=6)
    new_neg_line, = ax.plot(angles, norm_new_neg, '-', linewidth=2.5,
                            color=color_new_neg, alpha=0.95, zorder=8)
    ax.plot(angles, norm_new_neg, 'v', markersize=7,
            markerfacecolor=color_new_neg, markeredgecolor='white',
            markeredgewidth=1.5, alpha=0.95, zorder=9)

    # Add qubit pair labels at each data point with unique offsets to avoid overlap
    # Each series gets a different direction offset
    # Use path_effects for black border around text
    import matplotlib.patheffects as pe
    text_outline = [pe.withStroke(linewidth=1, foreground='white')]

    for i in range(6):
        # Rotated +phi: offset up-right
        ax.annotate(rotated_pairs_pos[i], (angles[i], norm_rotated_pos[i]),
                    fontsize=6, fontweight='bold', color=color_rotated_pos, ha='left', va='bottom',
                    xytext=(4, 8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)
        # Rotated -phi: offset down-right
        ax.annotate(rotated_pairs_neg[i], (angles[i], norm_rotated_neg[i]),
                    fontsize=6, fontweight='bold', color=color_rotated_neg, ha='left', va='top',
                    xytext=(4, -8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)
        # New +phi: offset up-left
        ax.annotate(new_pairs_pos[i], (angles[i], norm_new_pos[i]),
                    fontsize=6, fontweight='bold', color=color_new_pos, ha='right', va='bottom',
                    xytext=(-4, 8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)
        # New -phi: offset down-left
        ax.annotate(new_pairs_neg[i], (angles[i], norm_new_neg[i]),
                    fontsize=6, fontweight='bold', color=color_new_neg, ha='right', va='top',
                    xytext=(-4, -8), textcoords='offset points', zorder=15,
                    path_effects=text_outline)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, weight='semibold', color=color_text)

    # Grid setup - find actual max of normalized data and set ylim accordingly
    all_norm = norm_rotated_pos + norm_rotated_neg + norm_new_pos + norm_new_neg
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

    # Title (use varphi ϕ U+03D5)
    ax.set_title(f'ϕ = ±{phi_deg}°', size=14, weight='bold', pad=15, color=color_text)

    # Legend
    legend = ax.legend(
        [qm_line, leggett_line, rotated_pos_line, rotated_neg_line, new_pos_line, new_neg_line],
        [f'QM Theory (L₃={L3_theory:.3f})',
         f'Leggett Bound ({bound:.3f})',
         f'Opt. 24QB Rotated +{phi_deg}° (L₃={rotated_L3_pos:.3f})',
         f'Opt. 24QB Rotated −{phi_deg}° (L₃={rotated_L3_neg:.3f})',
         f'Opt. 24QB +{phi_deg}° (L₃={new_L3_pos:.3f})',
         f'Opt. 24QB −{phi_deg}° (L₃={new_L3_neg:.3f})'],
        loc='upper left', bbox_to_anchor=(1.02, 1.0),
        fontsize=11, frameon=True, fancybox=False,
        shadow=False, framealpha=0.95,
        edgecolor='#E8E8E8', facecolor='white',
        borderpad=0.8, labelspacing=0.5,
        handlelength=2.0
    )
    legend.get_frame().set_linewidth(0.8)


# Create figure: 3x2 grid for 5 angles
fig = plt.figure(figsize=(20, 24), facecolor='white')

for idx, phi_deg in enumerate(common_angles):
    # Get data
    rotated_entry_pos = get_data_by_phi(data_rotated, phi_deg, positive=True)
    rotated_entry_neg = get_data_by_phi(data_rotated, phi_deg, positive=False)
    new_entry_pos = get_data_by_phi(data_new, phi_deg, positive=True)
    new_entry_neg = get_data_by_phi(data_new, phi_deg, positive=False)

    if rotated_entry_pos is None or new_entry_pos is None:
        print(f"Missing data for phi={phi_deg}")
        continue

    # Rotated data
    rotated_corr_pos = rotated_entry_pos['correlations']
    rotated_corr_neg = rotated_entry_neg['correlations'] if rotated_entry_neg else rotated_corr_pos
    rotated_L3_pos = rotated_entry_pos['L3']
    rotated_L3_neg = rotated_entry_neg['L3'] if rotated_entry_neg else rotated_L3_pos

    # New data
    new_corr_pos = new_entry_pos['correlations']
    new_corr_neg = new_entry_neg['correlations'] if new_entry_neg else new_corr_pos
    new_L3_pos = new_entry_pos['L3']
    new_L3_neg = new_entry_neg['L3'] if new_entry_neg else new_L3_pos

    # Theory and bounds
    corr_theory = rotated_entry_pos['correlations_theory']
    L3_theory = rotated_entry_pos['L3_theory']
    bound = rotated_entry_pos['bound']
    corr_bound = -bound / 2.0

    # Close the plots
    rotated_pos_plot = rotated_corr_pos + [rotated_corr_pos[0]]
    rotated_neg_plot = rotated_corr_neg + [rotated_corr_neg[0]]
    new_pos_plot = new_corr_pos + [new_corr_pos[0]]
    new_neg_plot = new_corr_neg + [new_corr_neg[0]]
    theory_plot = corr_theory + [corr_theory[0]]
    bound_plot = [corr_bound] * (N + 1)

    # Create subplot
    ax = plt.subplot(3, 2, idx + 1, projection='polar', facecolor='#FAFBFC')
    style_comparison_radar(ax, rotated_pos_plot, rotated_neg_plot, new_pos_plot, new_neg_plot,
                           theory_plot, bound_plot, phi_deg,
                           rotated_L3_pos, rotated_L3_neg, new_L3_pos, new_L3_neg,
                           L3_theory, bound)

# Title removed per user request
plt.tight_layout()

# Save
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_rotated_vs_new_comparison.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved comparison to {output_file}')

plt.close()
