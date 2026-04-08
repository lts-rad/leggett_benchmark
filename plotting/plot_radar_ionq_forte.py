import os
#!/usr/bin/env python3
"""
Create radar plot for Leggett correlation measurements from IonQ Forte 24qb.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Load results
with open('../../production/leggett_results_ionq_forte_sequential_24qb.json', 'r') as f:
    results = json.load(f)

# Find results for phi = 30
result = None

for r in results:
    if abs(r['phi_deg'] - 30) < 0.1:
        result = r
        break

if not result:
    print("Error: Could not find results for phi = 30°")
    exit(1)

phi_deg = abs(result['phi_deg'])
phi_rad = np.radians(phi_deg)

# Extract correlations (IonQ format has both +phi and -phi in same entry)
corr_pos = result['correlations_pos']
corr_neg = result['correlations_neg']
corr_theory = result['correlations_theory']

# Correlation labels
labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']

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
# L₃ = (1/3) * sum of |C_i|, so if all equal: |C| = L₃ / 2
bound = result['bound']
corr_at_bound = -bound / 2.0  # Negative since correlations are negative
corr_bound_plot = [corr_at_bound] * (N + 1)

# Create figure
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot data
ax.plot(angles, corr_theory_plot, 'o--', linewidth=2, label=f'QM Ideal (L₃={result["L3_theory"]:.4f})',
        color='royalblue', markersize=8, alpha=0.7)
ax.plot(angles, corr_bound_plot, ':', linewidth=3, label=f'Leggett Bound (L₃={bound:.4f})',
        color='gray', alpha=0.7)
ax.plot(angles, corr_pos_plot, 'o-', linewidth=2.5, label=f'φ = +{phi_deg}° (L₃={result["L3_pos"]:.4f})',
        color='green', markersize=10)
ax.plot(angles, corr_neg_plot, 's-', linewidth=2.5, label=f'φ = -{phi_deg}° (L₃={result["L3_neg"]:.4f})',
        color='orange', markersize=10)

# Fill areas
ax.fill(angles, corr_theory_plot, alpha=0.15, color='royalblue')

# Set labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=14, weight='bold')

# Set radial limits
ax.set_ylim(-1.05, -0.9)
ax.set_yticks([-1.04, -1.02, -1.00, -0.98, -0.96, -0.94, -0.92])
ax.set_yticklabels(['-1.04', '-1.02', '-1.00', '-0.98', '-0.96', '-0.94', '-0.92'], size=10)

# Grid
ax.grid(True, linestyle='--', alpha=0.3)

# Title
title = f'Leggett Correlation Measurements: φ = ±{phi_deg}° (|Ψ⁻⟩ = (|01⟩-|10⟩)/√2)\n'
title += f'QM Theory: L₃ = {result["L3_theory"]:.4f} | Leggett Bound: L₃ ≤ {bound:.4f}\n'
title += f'IonQ Forte - 24 Qubits (12 pairs)'
ax.set_title(title, size=14, weight='bold', pad=20)

# Legend
ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=11, framealpha=0.9)

# Add annotations
textstr = f'Both measurements VIOLATE Leggett bound\n'
textstr += f'Margin (+{phi_deg}°): {result["L3_pos"] - bound:+.4f}\n'
textstr += f'Margin (-{phi_deg}°): {result["L3_neg"] - bound:+.4f}'
ax.text(0.5, -0.15, textstr, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()

# Save
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'ionq_forte_24qb_phi_30.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'Saved to {output_file}')

plt.show()
