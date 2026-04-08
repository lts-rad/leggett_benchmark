#!/usr/bin/env python3
"""
Compare IBM Pittsburgh 24qb vs IonQ Forte 24qb (apples to apples).
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT.json', 'r') as f:
    ibm_results = json.load(f)

with open('../../production/leggett_results_ionq_forte_sequential_24qb.json', 'r') as f:
    ionq_results = json.load(f)

# Plot angle
phi_deg_target = 30

# Find IBM results for phi = +30 and -30
ibm_pos = None
ibm_neg = None
for r in ibm_results:
    if abs(r['phi_deg'] - phi_deg_target) < 0.1:
        ibm_pos = r
    elif abs(r['phi_deg'] + phi_deg_target) < 0.1:
        ibm_neg = r

# Find IonQ result (has both +/- in one entry)
ionq_result = None
for r in ionq_results:
    if abs(r['phi_deg'] - phi_deg_target) < 0.1:
        ionq_result = r
        break

if not ibm_pos or not ibm_neg or not ionq_result:
    print(f"Error: Could not find results for phi = ±{phi_deg_target}°")
    exit(1)

phi_deg = phi_deg_target

# Extract correlations
ibm_corr_pos = ibm_pos['correlations']
ibm_corr_neg = ibm_neg['correlations']
ionq_corr_pos = ionq_result['correlations_pos']
ionq_corr_neg = ionq_result['correlations_neg']
corr_theory = ibm_pos['correlations_theory']

# Correlation labels with IBM qubit pairs
# Layout: [67, 68, 29, 30, 54, 55, 93, 92, 109, 110, 2, 3, 13, 14, 112, 113, 153, 152, 75, 74, 94, 95, 107, 97]
# +30° uses pairs 0-5: (67,68), (29,30), (54,55), (93,92), (109,110), (2,3)
# -30° uses pairs 6-11: (13,14), (112,113), (153,152), (75,74), (94,95), (107,97)
labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']
ibm_qubits_pos = ['(67,68)', '(29,30)', '(54,55)', '(93,92)', '(109,110)', '(2,3)']
ibm_qubits_neg = ['(13,14)', '(112,113)', '(153,152)', '(75,74)', '(94,95)', '(107,97)']

# Number of variables
N = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Append first value to close the plot
ibm_corr_pos_plot = ibm_corr_pos + [ibm_corr_pos[0]]
ibm_corr_neg_plot = ibm_corr_neg + [ibm_corr_neg[0]]
ionq_corr_pos_plot = ionq_corr_pos + [ionq_corr_pos[0]]
ionq_corr_neg_plot = ionq_corr_neg + [ionq_corr_neg[0]]
corr_theory_plot = corr_theory + [corr_theory[0]]

# Calculate uniform correlation at Leggett bound
bound = ibm_pos['bound']
corr_at_bound = -bound / 2.0
corr_bound_plot = [corr_at_bound] * (N + 1)

# Create figure with two subplots
fig = plt.figure(figsize=(20, 10))

# Left plot: IBM Pittsburgh TOP6 12qb
ax1 = plt.subplot(121, projection='polar')

ax1.plot(angles, corr_theory_plot, 'o--', linewidth=2, label=f'QM Ideal (L₃={ibm_pos["L3_theory"]:.4f})',
        color='royalblue', markersize=8, alpha=0.7)
ax1.plot(angles, corr_bound_plot, ':', linewidth=3, label=f'Leggett Bound (L₃={bound:.4f})',
        color='gray', alpha=0.7)
ax1.plot(angles, ibm_corr_pos_plot, 'o-', linewidth=2.5, label=f'φ = +{phi_deg}° (L₃={ibm_pos["L3"]:.4f})',
        color='green', markersize=10)
ax1.plot(angles, ibm_corr_neg_plot, 's-', linewidth=2.5, label=f'φ = -{phi_deg}° (L₃={ibm_neg["L3"]:.4f})',
        color='orange', markersize=10)

ax1.fill(angles, corr_theory_plot, alpha=0.15, color='royalblue')

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(labels, size=14, weight='bold')

ax1.set_ylim(-1.05, -0.9)
ax1.set_yticks([-1.04, -1.02, -1.00, -0.98, -0.96, -0.94, -0.92])
ax1.set_yticklabels(['-1.04', '-1.02', '-1.00', '-0.98', '-0.96', '-0.94', '-0.92'], size=10)

ax1.grid(True, linestyle='--', alpha=0.3)

ibm_margin_pos = (ibm_pos["L3"] - bound) / bound * 100
ibm_margin_neg = (ibm_neg["L3"] - bound) / bound * 100

title1 = f'IBM Pittsburgh - 24 Qubits (12 pairs)\n'
title1 += f'φ = ±{phi_deg}° | L₃(+) = {ibm_pos["L3"]:.4f} | L₃(-) = {ibm_neg["L3"]:.4f}\n'
title1 += f'Margin (+): {ibm_margin_pos:+.2f}% | Margin (-): {ibm_margin_neg:+.2f}%'
ax1.set_title(title1, size=13, weight='bold', pad=20)

ax1.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10, framealpha=0.9)

# Right plot: IonQ Forte
ax2 = plt.subplot(122, projection='polar')

ax2.plot(angles, corr_theory_plot, 'o--', linewidth=2, label=f'QM Ideal (L₃={ionq_result["L3_theory"]:.4f})',
        color='royalblue', markersize=8, alpha=0.7)
ax2.plot(angles, corr_bound_plot, ':', linewidth=3, label=f'Leggett Bound (L₃={bound:.4f})',
        color='gray', alpha=0.7)
ax2.plot(angles, ionq_corr_pos_plot, 'o-', linewidth=2.5, label=f'φ = +{phi_deg}° (L₃={ionq_result["L3_pos"]:.4f})',
        color='green', markersize=10)
ax2.plot(angles, ionq_corr_neg_plot, 's-', linewidth=2.5, label=f'φ = -{phi_deg}° (L₃={ionq_result["L3_neg"]:.4f})',
        color='orange', markersize=10)

ax2.fill(angles, corr_theory_plot, alpha=0.15, color='royalblue')

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels, size=14, weight='bold')

ax2.set_ylim(-1.05, -0.9)
ax2.set_yticks([-1.04, -1.02, -1.00, -0.98, -0.96, -0.94, -0.92])
ax2.set_yticklabels(['-1.04', '-1.02', '-1.00', '-0.98', '-0.96', '-0.94', '-0.92'], size=10)

ax2.grid(True, linestyle='--', alpha=0.3)

ionq_margin_pos = (ionq_result["L3_pos"] - bound) / bound * 100
ionq_margin_neg = (ionq_result["L3_neg"] - bound) / bound * 100

title2 = f'IonQ Forte - 24 Qubits (12 pairs)\n'
title2 += f'φ = ±{phi_deg}° | L₃(+) = {ionq_result["L3_pos"]:.4f} | L₃(-) = {ionq_result["L3_neg"]:.4f}\n'
title2 += f'Margin (+): {ionq_margin_pos:+.2f}% | Margin (-): {ionq_margin_neg:+.2f}%'
ax2.set_title(title2, size=13, weight='bold', pad=20)

ax2.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10, framealpha=0.9)

# Overall title
fig.suptitle(f'Leggett Inequality Test Comparison: IBM Pittsburgh 24qb vs IonQ Forte 24qb (φ = ±{phi_deg}°)',
             fontsize=16, weight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_file = f'radar_comparison_ibm_top6_vs_ionq_phi_{phi_deg}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'Saved to {output_file}')

print(f'\n=== Comparison Summary (φ = ±{phi_deg}°) ===')
print(f'\nIBM Pittsburgh 24qb:')
print(f'  L₃(+{phi_deg}°) = {ibm_pos["L3"]:.4f} | Margin: {ibm_margin_pos:+.2f}%')
print(f'  L₃(-{phi_deg}°) = {ibm_neg["L3"]:.4f} | Margin: {ibm_margin_neg:+.2f}%')
print(f'  Average Margin: {(ibm_margin_pos + ibm_margin_neg) / 2:+.2f}%')
print(f'\nIonQ Forte 24qb:')
print(f'  L₃(+{phi_deg}°) = {ionq_result["L3_pos"]:.4f} | Margin: {ionq_margin_pos:+.2f}%')
print(f'  L₃(-{phi_deg}°) = {ionq_result["L3_neg"]:.4f} | Margin: {ionq_margin_neg:+.2f}%')
print(f'  Average Margin: {(ionq_margin_pos + ionq_margin_neg) / 2:+.2f}%')

ionq_avg = (ionq_margin_pos + ionq_margin_neg) / 2
ibm_avg = (ibm_margin_pos + ibm_margin_neg) / 2
improvement = ionq_avg / ibm_avg

print(f'\n=== Performance Ratio ===')
print(f'IonQ is {improvement:.2f}× better than IBM Pittsburgh (24qb vs 24qb)')

plt.show()
