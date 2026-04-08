import os
#!/usr/bin/env python3
"""
Clean, professional radar plot for IBM Pittsburgh with obvious Leggett bound.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Clean styling
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Load results
with open('leggett_results_ibm_ibm_pittsburgh_OPTIMIZED_layout_20251202_114427.json', 'r') as f:
    data = json.load(f)

# Extract data - IBM format is different from IonQ
corr_pos = data['positive']['correlations']
corr_neg = data['negative']['correlations']
corr_theory = data['positive']['correlations_theory']
bound = data['positive']['bound']

# Labels
labels = ['C(a₁,b₁)', 'C(a₁,b₁′)', 'C(a₂,b₂)', 'C(a₂,b₂′)', 'C(a₃,b₃)', 'C(a₃,b₃′)']
N = len(labels)

# Angles
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Transform to absolute values: larger |C| = further out = stronger correlations
corr_pos_abs = [abs(c) for c in corr_pos]
corr_neg_abs = [abs(c) for c in corr_neg]
corr_theory_abs = [abs(c) for c in corr_theory]

# Log transform: -log(1 - |C|) expands region near 1
def log_transform(c):
    return -np.log(1 - min(c, 0.999))  # cap to avoid inf

corr_pos_log = [log_transform(c) for c in corr_pos_abs]
corr_neg_log = [log_transform(c) for c in corr_neg_abs]
corr_theory_log = [log_transform(c) for c in corr_theory_abs]

# Close the plots
corr_pos_plot = corr_pos_log + [corr_pos_log[0]]
corr_neg_plot = corr_neg_log + [corr_neg_log[0]]
corr_theory_plot = corr_theory_log + [corr_theory_log[0]]

# Leggett bound line (absolute value, log transformed)
corr_at_bound = bound / 2.0  # Now positive
corr_bound_log = log_transform(corr_at_bound)
corr_bound_plot = [corr_bound_log] * (N + 1)

# Create figure
fig = plt.figure(figsize=(10, 10), facecolor='white')
ax = fig.add_subplot(111, projection='polar', facecolor='white')

# Professional color palette - coordinated blue tones for bounds, clearly differentiable
color_measure1 = '#6C5CE7'    # Purple - hardware measurement 1
color_measure2 = '#00B894'    # Teal/green - hardware measurement 2
color_qm = '#2E86AB'          # Darker steel blue - QM prediction
color_leggett = '#A3D5FF'     # Light pastel blue - Leggett bound (clearly lighter)

# 1. QM Prediction - elegant dashed line (draw first)
ax.plot(angles, corr_theory_plot, '--', linewidth=2.5, dashes=(8, 4),
        color=color_qm, alpha=0.9,
        label='QM Prediction', zorder=2)

# 2. HARDWARE MEASUREMENTS (draw next)
ax.plot(angles, corr_pos_plot, 'o-', linewidth=2.5,
        color=color_measure1, markersize=10,
        markerfacecolor=color_measure1,
        markeredgewidth=2.5, markeredgecolor='white',
        label='φ = +30°', zorder=3, alpha=0.95)

ax.plot(angles, corr_neg_plot, 's-', linewidth=2.5,
        color=color_measure2, markersize=10,
        markerfacecolor=color_measure2,
        markeredgewidth=2.5, markeredgecolor='white',
        label='φ = −30°', zorder=3, alpha=0.95)

# 3. LEGGETT BOUND - dashed with different pattern, lighter blue
ax.plot(angles, corr_bound_plot, '--', linewidth=3.5, dashes=(4, 2),
        color=color_leggett, alpha=1.0,
        label='Leggett Bound', zorder=10)

# Clean axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=14, weight='400', color='#2C3E50')

# Log-scaled radial axis to expand the region near 1
# Set ylim to QM prediction so it becomes the outer boundary
qm_log = corr_theory_log[0]  # QM prediction (all same value)
ax.set_ylim(0, qm_log)
ax.set_yticks([-np.log(1-0.5), -np.log(1-0.9), corr_bound_log])
ax.set_yticklabels(['0.50', '0.90', '0.914'], size=11, color='#7F8C8D')
ax.spines['polar'].set_visible(False)  # Hide the outer black circle

# Minimal grid
ax.grid(True, linestyle='-', alpha=0.2, linewidth=1, color='#BDC3C7')
ax.set_axisbelow(True)

# Title - accurate description
ax.set_title('Leggett Inequality Violation (IBM Pittsburgh)',
             size=18, weight='500', pad=25, color='#2C3E50')

# Legend
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.08),
                   fontsize=12, frameon=True, fancybox=False,
                   shadow=False, framealpha=1.0,
                   edgecolor='#DFE6E9', facecolor='white',
                   borderpad=1.0, labelspacing=0.8)
legend.get_frame().set_linewidth(1.2)

plt.tight_layout()

# Save
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'radar_ibm_pittsburgh_clean.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved clean radar plot to {output_file}')

plt.close()
