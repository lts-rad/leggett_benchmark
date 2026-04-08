import os
#!/usr/bin/env python3
"""
Side-by-side radar plots for IBM Pittsburgh and IonQ Forte.
Publication-quality visualization with stunning visual effects.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
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
plt.rcParams['text.antialiased'] = True
plt.rcParams['lines.antialiased'] = True

# Labels with better typography
labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ========== Load Data ==========

# IBM Pittsburgh
with open('leggett_results_ibm_ibm_pittsburgh_OPTIMIZED_layout_20251202_114427.json', 'r') as f:
    data_ibm = json.load(f)

ibm_corr_pos = data_ibm['positive']['correlations']
ibm_corr_neg = data_ibm['negative']['correlations']
ibm_L3_pos = data_ibm['positive']['L3']
ibm_L3_neg = data_ibm['negative']['L3']
ibm_L3_theory = data_ibm['positive']['L3_theory']
bound = data_ibm['positive']['bound']

# IonQ Forte
with open(os.path.join(_DATA_DIR, 'leggett_results_ionq_forte_sequential_24qb.json') , 'r') as f:
    data_ionq = json.load(f)

ionq_result = None
for entry in data_ionq:
    if entry['phi_deg'] == 30.0:
        ionq_result = entry
        break

ionq_corr_pos = ionq_result['correlations_pos']
ionq_corr_neg = ionq_result['correlations_neg']
ionq_L3_pos = ionq_result['L3_pos']
ionq_L3_neg = ionq_result['L3_neg']
ionq_L3_theory = ionq_result['L3_theory']

# Theory values (same for both)
corr_theory = data_ibm['positive']['correlations_theory']
corr_at_bound = -bound / 2.0

# Close the plots
ibm_corr_pos_plot = ibm_corr_pos + [ibm_corr_pos[0]]
ibm_corr_neg_plot = ibm_corr_neg + [ibm_corr_neg[0]]
ionq_corr_pos_plot = ionq_corr_pos + [ionq_corr_pos[0]]
ionq_corr_neg_plot = ionq_corr_neg + [ionq_corr_neg[0]]
corr_theory_plot = corr_theory + [corr_theory[0]]
corr_bound_plot = [corr_at_bound] * (N + 1)

phi_deg = 30

# Stunning color palette
color_qm = '#3498DB'           # Bright blue for QM
color_qm_fill = '#3498DB'      # QM fill
color_leggett = '#7F8C8D'      # Sophisticated gray for classical bound
color_pos = '#F39C12'          # Deep amber/gold
color_neg = '#F7DC6F'          # Light yellow
color_text = '#2C3E50'         # Dark blue-gray for text
color_grid = '#BDC3C7'         # Light gray for grid

# Create figure with dark edges for contrast
fig = plt.figure(figsize=(22, 11), facecolor='white')

def style_radar_plot(ax, corr_pos_plot, corr_neg_plot, L3_pos, L3_neg, L3_theory, title_name):
    """Apply stunning styling to a radar plot"""

    # QM prediction - clean dashed blue line (no glow)
    qm_line, = ax.plot(angles, corr_theory_plot, '--', linewidth=2.5,
             dashes=(8, 4), color=color_qm, alpha=0.95, zorder=5)

    # Leggett bound - clean dashed line (no glow)
    leggett_line, = ax.plot(angles, corr_bound_plot, '--', linewidth=2.5,
             dashes=(4, 2), color=color_leggett, alpha=0.9, zorder=4)

    # Measurement lines with glow effect
    # Positive phi - with sophisticated glow
    for i in range(4):
        ax.plot(angles, corr_pos_plot, '-', linewidth=8-i*1.5,
                color=color_pos, alpha=0.08, zorder=6)
    pos_line, = ax.plot(angles, corr_pos_plot, '-', linewidth=3,
             color=color_pos, alpha=0.95, zorder=8)
    ax.plot(angles, corr_pos_plot, 'o', markersize=12,
             markerfacecolor=color_pos, markeredgecolor='white',
             markeredgewidth=2.5, alpha=0.95, zorder=9)

    # Negative phi - with sophisticated glow
    for i in range(4):
        ax.plot(angles, corr_neg_plot, '-', linewidth=8-i*1.5,
                color=color_neg, alpha=0.08, zorder=6)
    neg_line, = ax.plot(angles, corr_neg_plot, '-', linewidth=3,
             color=color_neg, alpha=0.95, zorder=8)
    ax.plot(angles, corr_neg_plot, 's', markersize=11,
             markerfacecolor=color_neg, markeredgecolor='white',
             markeredgewidth=2.5, alpha=0.95, zorder=9)

    # Axis labels with dots at each C() position
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12, weight='semibold', color=color_text)

    # Add dots at the outer edge for each C() label
    outer_radius = -0.88
    for angle in angles[:-1]:
        ax.plot(angle, outer_radius, 'o', markersize=5,
                color='#7F8C8D', alpha=0.6, zorder=10)

    ax.set_ylim(-1.0, -0.88)
    ax.set_rticks([-1.0, -0.98, -0.96, -0.94, -0.92, -0.90])
    ax.set_yticklabels(['−1', '', '', '', '', '−0.9'], size=10, color='#95A5A6')

    # Refined grid - more visible rings
    ax.grid(True, linestyle='-', alpha=0.35, linewidth=1.0, color='#BDC3C7')
    ax.spines['polar'].set_color('#D5DBDB')
    ax.spines['polar'].set_linewidth(1.5)

    # Title
    ax.set_title(title_name, size=18, weight='bold', pad=25, color=color_text,
                 fontfamily='sans-serif')

    # Elegant legend with L3 values included
    legend = ax.legend(
        [qm_line, leggett_line, pos_line, neg_line],
        [f'QM Prediction (L₃={L3_theory:.4f})',
         f'Leggett Bound (L₃={bound:.4f})',
         f'φ = +{phi_deg}° (L₃={L3_pos:.4f})',
         f'φ = −{phi_deg}° (L₃={L3_neg:.4f})'],
        loc='upper left', bbox_to_anchor=(1.05, 1.0),
        fontsize=11, frameon=True, fancybox=False,
        shadow=False, framealpha=0.95,
        edgecolor='#E8E8E8', facecolor='white',
        borderpad=1.0, labelspacing=0.8,
        handlelength=2.5
    )
    legend.get_frame().set_linewidth(0.8)

# ========== Left: IBM Pittsburgh ==========
ax1 = plt.subplot(121, projection='polar', facecolor='#FAFBFC')
style_radar_plot(ax1, ibm_corr_pos_plot, ibm_corr_neg_plot,
                 ibm_L3_pos, ibm_L3_neg, ibm_L3_theory, 'IBM Pittsburgh')

# ========== Right: IonQ Forte ==========
ax2 = plt.subplot(122, projection='polar', facecolor='#FAFBFC')
style_radar_plot(ax2, ionq_corr_pos_plot, ionq_corr_neg_plot,
                 ionq_L3_pos, ionq_L3_neg, ionq_L3_theory, 'IonQ Forte')

# Main title with refined typography
fig.suptitle('Leggett Inequality Violation',
             fontsize=26, weight='bold', y=0.96, color=color_text,
             fontfamily='sans-serif')

# Subtitle
fig.text(0.5, 0.91, 'Superconducting vs Trapped-Ion Quantum Processors  ·  24 Qubits  ·  φ = ±30°',
         ha='center', fontsize=14, color='#95A5A6', weight='light',
         fontfamily='sans-serif')

plt.tight_layout(rect=[0, 0.02, 1, 0.88])

# Save
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_comparison.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved comparison plot to {output_file}')

plt.close()
