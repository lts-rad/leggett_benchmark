import os
#!/usr/bin/env python3
"""
Create a polished, publication-quality radar plot for IonQ Forte hardware results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Set publication-quality defaults
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['grid.linewidth'] = 0.8
rcParams['grid.alpha'] = 0.3

# Load results - find φ = 30° data
with open(os.path.join(_DATA_DIR, 'leggett_results_ionq_forte_sequential_24qb.json') , 'r') as f:
    data = json.load(f)

# Find φ = 30° entry
result = None
for entry in data:
    if entry['phi_deg'] == 30.0:
        result = entry
        break

phi_deg = result['phi_deg']

# Extract correlations
corr_pos = result['correlations_pos']
corr_neg = result['correlations_neg']
corr_theory = result['correlations_theory']

# Calculate Leggett bound
bound = result['bound']
L3_pos = result['L3_pos']
L3_neg = result['L3_neg']
L3_theory = result['L3_theory']

# Correlation labels - clean and professional
labels = ['C(a₁,b₁)', 'C(a₁,b₁′)', 'C(a₂,b₂)', 'C(a₂,b₂′)', 'C(a₃,b₃)', 'C(a₃,b₃′)']

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
corr_at_bound = -bound / 2.0
corr_bound_plot = [corr_at_bound] * (N + 1)

# Create figure with high DPI
fig = plt.figure(figsize=(12, 10), dpi=150)
ax = fig.add_subplot(111, projection='polar')

# Sophisticated color palette - inspired by modern data viz
color_theory = '#5B8FA3'      # Muted steel blue
color_bound = '#B8B8B8'       # Light gray
color_pos = '#E8927C'         # Soft coral/salmon
color_neg = '#9DC183'         # Soft sage green

# Plot with sophisticated styling
# 1. Theory (subtle background reference)
ax.plot(angles, corr_theory_plot, 'o--', linewidth=2,
        color=color_theory, markersize=7, alpha=0.6,
        label=f'QM Theory',
        zorder=2)
ax.fill(angles, corr_theory_plot, alpha=0.08, color=color_theory, zorder=1)

# 2. Leggett bound (reference line)
ax.plot(angles, corr_bound_plot, ':', linewidth=2.5,
        color=color_bound, alpha=0.8,
        label=f'Leggett Bound',
        zorder=1)

# 3. Positive angle measurement
ax.plot(angles, corr_pos_plot, 'o-', linewidth=2.5,
        color=color_pos, markersize=9, markeredgewidth=1.5,
        markeredgecolor='white', alpha=0.9,
        label=f'φ = +30°',
        zorder=4)
ax.fill(angles, corr_pos_plot, alpha=0.15, color=color_pos, zorder=3)

# 4. Negative angle measurement
ax.plot(angles, corr_neg_plot, 's-', linewidth=2.5,
        color=color_neg, markersize=9, markeredgewidth=1.5,
        markeredgecolor='white', alpha=0.9,
        label=f'φ = −30°',
        zorder=4)
ax.fill(angles, corr_neg_plot, alpha=0.15, color=color_neg, zorder=3)

# Set labels with clean formatting
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=13, weight='normal', color='#333333')

# Set radial limits - tight range for clean appearance
ax.set_ylim(-0.975, -0.940)
ax.set_yticks([-0.97, -0.96, -0.95])
ax.set_yticklabels(['-0.97', '-0.96', '-0.95'], size=10, color='#666666')

# Clean grid
ax.grid(True, linestyle='-', alpha=0.25, linewidth=0.8, color='#CCCCCC')
ax.set_axisbelow(True)

# Clean, minimal title
title_main = f'Leggett Inequality: Correlation Measurements'
ax.set_title(title_main, size=16, weight='normal', pad=20, color='#2C3E50')

# Legend with clean styling
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.05),
                   fontsize=11, frameon=True, fancybox=False,
                   shadow=False, framealpha=0.98, edgecolor='#DDDDDD',
                   borderpad=0.9, labelspacing=0.7)
legend.get_frame().set_linewidth(1.0)

# Tight layout
plt.tight_layout()

# Save with high quality
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_ionq_forte_polished.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved polished radar plot to {output_file}')

plt.close()
