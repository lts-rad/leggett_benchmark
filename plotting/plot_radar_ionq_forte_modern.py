import os
#!/usr/bin/env python3
"""
Create a modern, visually striking radar plot for IonQ Forte with prominent Leggett bound.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Modern styling
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Load results
with open(os.path.join(_DATA_DIR, 'leggett_results_ionq_forte_sequential_24qb.json') , 'r') as f:
    data = json.load(f)

result = None
for entry in data:
    if entry['phi_deg'] == 30.0:
        result = entry
        break

# Extract data
corr_pos = result['correlations_pos']
corr_neg = result['correlations_neg']
corr_theory = result['correlations_theory']
bound = result['bound']

# Labels
labels = ['C(a₁,b₁)', 'C(a₁,b₁′)', 'C(a₂,b₂)', 'C(a₂,b₂′)', 'C(a₃,b₃)', 'C(a₃,b₃′)']
N = len(labels)

# Angles
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Close the plots
corr_pos_plot = corr_pos + [corr_pos[0]]
corr_neg_plot = corr_neg + [corr_neg[0]]
corr_theory_plot = corr_theory + [corr_theory[0]]

# Leggett bound line
corr_at_bound = -bound / 2.0
corr_bound_plot = [corr_at_bound] * (N + 1)

# Create figure
fig = plt.figure(figsize=(11, 11), facecolor='white')
ax = fig.add_subplot(111, projection='polar', facecolor='#FAFAFA')

# Modern color palette - high contrast, publication quality
color_violation = '#FF6B6B'    # Vibrant red for violations
color_allowed = '#4ECDC4'       # Teal for allowed region
color_theory = '#45B7D1'        # Sky blue for theory
color_bound = '#2C3E50'         # Dark blue-gray for bound

# 1. FILL THE VIOLATION REGION (between bound and edge) - this shows quantum violations
outer_limit = [-0.94] * (N + 1)
ax.fill_between(angles, corr_bound_plot, outer_limit,
                alpha=0.08, color=color_violation, zorder=1,
                label='_nolegend_')

# 2. PROMINENT LEGGETT BOUND - thick, dark line
ax.plot(angles, corr_bound_plot, '-', linewidth=4,
        color=color_bound, alpha=0.9,
        label='Leggett Bound', zorder=3)

# 3. QM Theory - elegant reference
ax.plot(angles, corr_theory_plot, '--', linewidth=2.5,
        color=color_theory, alpha=0.7,
        label='QM Theory', zorder=2)

# 4. HARDWARE MEASUREMENTS - bold and prominent
ax.plot(angles, corr_pos_plot, 'o-', linewidth=3.5,
        color=color_violation, markersize=11,
        markerfacecolor=color_violation,
        markeredgewidth=2, markeredgecolor='white',
        label='φ = +30°', zorder=5, alpha=0.95)

ax.plot(angles, corr_neg_plot, 's-', linewidth=3.5,
        color='#95E1D3', markersize=11,
        markerfacecolor='#95E1D3',
        markeredgewidth=2, markeredgecolor='white',
        label='φ = −30°', zorder=5, alpha=0.95)

# Clean axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=14, weight='500', color='#2C3E50')

# Radial limits
ax.set_ylim(-0.975, -0.940)
ax.set_yticks([-0.97, -0.96, -0.95])
ax.set_yticklabels(['-0.97', '-0.96', '-0.95'], size=11, color='#7F8C8D')

# Minimal grid
ax.grid(True, linestyle='-', alpha=0.15, linewidth=1.2, color='#BDC3C7')
ax.set_axisbelow(True)

# Remove circular spines for cleaner look
ax.spines['polar'].set_visible(False)

# Title
ax.set_title('Leggett Inequality: Correlation Measurements',
             size=17, weight='500', pad=25, color='#2C3E50')

# Legend - clean and positioned well
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08),
                   fontsize=12, frameon=True, fancybox=False,
                   shadow=False, framealpha=1.0,
                   edgecolor='#D5D8DC', facecolor='white',
                   borderpad=1.0, labelspacing=0.9)
legend.get_frame().set_linewidth(1.5)

plt.tight_layout()

# Save
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_ionq_forte_modern.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved modern radar plot to {output_file}')

plt.close()
