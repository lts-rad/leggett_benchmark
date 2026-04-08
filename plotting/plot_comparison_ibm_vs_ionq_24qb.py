import os
#!/usr/bin/env python3
"""
Compare IBM Pittsburgh 24qb vs IonQ Forte 24qb Leggett test results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT.json', 'r') as f:
    ibm_results = json.load(f)

with open('../../production/leggett_results_ionq_forte_sequential_24qb.json', 'r') as f:
    ionq_results = json.load(f)

# Extract angles and L3 values
angles = []
ibm_l3 = []
ionq_l3_pos = []
ionq_l3_neg = []
bounds = []

for ibm_result in ibm_results:
    phi = ibm_result['phi_deg']

    # Find corresponding IonQ result
    ionq_result = None
    for r in ionq_results:
        if abs(r['phi_deg'] - phi) < 0.1:
            ionq_result = r
            break

    if ionq_result:
        angles.append(phi)
        ibm_l3.append(ibm_result['L3'])
        ionq_l3_pos.append(ionq_result['L3_pos'])
        ionq_l3_neg.append(ionq_result['L3_neg'])
        bounds.append(ibm_result['bound'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: L3 vs angle
ax1.plot(angles, ibm_l3, 'o-', label='IBM Pittsburgh 24qb', color='blue', linewidth=2, markersize=8)
ax1.plot(angles, ionq_l3_pos, 's-', label='IonQ Forte 24qb (+φ)', color='green', linewidth=2, markersize=8)
ax1.plot(angles, ionq_l3_neg, '^-', label='IonQ Forte 24qb (-φ)', color='orange', linewidth=2, markersize=8)
ax1.plot(angles, bounds, '--', label='Leggett Bound', color='red', linewidth=2, alpha=0.7)

# Theory line
theory_l3 = [r['L3_theory'] for r in ibm_results]
ax1.plot(angles, theory_l3, ':', label='QM Theory', color='gray', linewidth=2, alpha=0.7)

ax1.set_xlabel('Relative Phase φ (degrees)', fontsize=12, weight='bold')
ax1.set_ylabel('Leggett Parameter L₃', fontsize=12, weight='bold')
ax1.set_title('Leggett Test: IBM Pittsburgh vs IonQ Forte (24 Qubits)', fontsize=14, weight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Right plot: Margin above bound (%)
ibm_margin = [(l3 - b) / b * 100 for l3, b in zip(ibm_l3, bounds)]
ionq_margin_pos = [(l3 - b) / b * 100 for l3, b in zip(ionq_l3_pos, bounds)]
ionq_margin_neg = [(l3 - b) / b * 100 for l3, b in zip(ionq_l3_neg, bounds)]

ax2.plot(angles, ibm_margin, 'o-', label='IBM Pittsburgh 24qb', color='blue', linewidth=2, markersize=8)
ax2.plot(angles, ionq_margin_pos, 's-', label='IonQ Forte 24qb (+φ)', color='green', linewidth=2, markersize=8)
ax2.plot(angles, ionq_margin_neg, '^-', label='IonQ Forte 24qb (-φ)', color='orange', linewidth=2, markersize=8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Leggett Bound')

ax2.set_xlabel('Relative Phase φ (degrees)', fontsize=12, weight='bold')
ax2.set_ylabel('Violation Margin (%)', fontsize=12, weight='bold')
ax2.set_title('Leggett Bound Violation Margin', fontsize=14, weight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'comparison_ibm_vs_ionq_24qb.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'Saved to {output_file}')

# Print summary statistics
print('\n=== Summary Statistics (φ=30°) ===')
idx_30 = angles.index(30)
print(f'IBM Pittsburgh 24qb:')
print(f'  L₃ = {ibm_l3[idx_30]:.4f}')
print(f'  Bound = {bounds[idx_30]:.4f}')
print(f'  Margin = {ibm_margin[idx_30]:+.2f}%')
print(f'\nIonQ Forte 24qb (+φ):')
print(f'  L₃ = {ionq_l3_pos[idx_30]:.4f}')
print(f'  Bound = {bounds[idx_30]:.4f}')
print(f'  Margin = {ionq_margin_pos[idx_30]:+.2f}%')
print(f'\nIonQ Forte 24qb (-φ):')
print(f'  L₃ = {ionq_l3_neg[idx_30]:.4f}')
print(f'  Bound = {bounds[idx_30]:.4f}')
print(f'  Margin = {ionq_margin_neg[idx_30]:+.2f}%')

print('\n=== Average Performance ===')
print(f'IBM Pittsburgh: {np.mean(ibm_margin):+.2f}% ± {np.std(ibm_margin):.2f}%')
print(f'IonQ Forte (+φ): {np.mean(ionq_margin_pos):+.2f}% ± {np.std(ionq_margin_pos):.2f}%')
print(f'IonQ Forte (-φ): {np.mean(ionq_margin_neg):+.2f}% ± {np.std(ionq_margin_neg):.2f}%')

plt.show()
