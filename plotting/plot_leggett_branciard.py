import os
#!/usr/bin/env python3
"""
Plot Leggett inequality from Branciard et al. arXiv:0801.2241v2

Shows:
- Leggett bound (the "triangle"): L₃(φ) ≤ 2 - (2/3)|sin(φ/2)|
- QM prediction: L_Ψ⁻(φ) = 2|cos(φ/2)|
- Required visibility on right axis
"""

import numpy as np
import matplotlib.pyplot as plt

# Angle range
phi_deg = np.linspace(-90, 90, 1000)
phi_rad = np.radians(phi_deg)

# Leggett bound: L₃(φ) ≤ 2 - (2/3)|sin(φ/2)|
leggett_bound = 2 - (2/3) * np.abs(np.sin(phi_rad / 2))

# QM prediction for pure singlet: L_Ψ⁻(φ) = 2|cos(φ/2)|
qm_prediction = 2 * np.abs(np.cos(phi_rad / 2))

# Required visibility to violate at each angle
# V_required = (2 - (2/3)|sin(φ/2)|) / (2|cos(φ/2)|)
# Cut off at ±60 degrees as requested
with np.errstate(divide='ignore', invalid='ignore'):
    v_required = leggett_bound / qm_prediction
    v_required = np.where(np.abs(phi_deg) > 60, np.nan, v_required)

# Key angles
phi_max_violation = 2 * np.degrees(np.arctan(1/3))  # ~36.9°
v_threshold_n3 = 2 * np.sqrt(2) / 3  # ~94.3%

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()

# Plot on left axis
# Violation region is BETWEEN Leggett bound and QM prediction (where QM > bound)
ax1.fill_between(phi_deg, leggett_bound, qm_prediction,
                  where=(qm_prediction > leggett_bound),
                  alpha=0.3, color='green', label='Leggett violation region')
ax1.plot(phi_deg, leggett_bound, 'b-', linewidth=2.5, label='Leggett bound: $2 - \\frac{2}{3}|\\sin\\frac{\\varphi}{2}|$')
ax1.plot(phi_deg, qm_prediction, 'r--', linewidth=2.5, label='QM prediction: $2|\\cos\\frac{\\varphi}{2}|$')

# Mark maximum violation point
ax1.axvline(x=phi_max_violation, color='gray', linestyle=':', alpha=0.7)
ax1.axvline(x=-phi_max_violation, color='gray', linestyle=':', alpha=0.7)
ax1.plot([phi_max_violation, -phi_max_violation],
         [2*np.cos(np.radians(phi_max_violation)/2)]*2, 'ro', markersize=8)

# Plot required visibility on right axis
ax2.plot(phi_deg, v_required * 100, 'g-', linewidth=2, label='Required visibility')
ax2.axhline(y=v_threshold_n3 * 100, color='orange', linestyle='--', linewidth=1.5,
            label=f'Min visibility: {v_threshold_n3*100:.1f}%')

# Mark key visibility points (only up to 60 degrees)
key_angles = [15, 25, 30, 36.9, 45, 60]
for ang in key_angles:
    if ang <= 60:
        v_at_ang = (2 - (2/3)*np.sin(np.radians(ang)/2)) / (2*np.cos(np.radians(ang)/2))
        ax2.plot(ang, v_at_ang * 100, 'g^', markersize=8)
        ax2.annotate(f'{v_at_ang*100:.1f}%', (ang, v_at_ang*100),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

# Labels and formatting
ax1.set_xlabel('Angle φ (degrees)', fontsize=14)
ax1.set_ylabel('$L_3(\\varphi)$', fontsize=14, color='blue')
ax2.set_ylabel('Required Visibility (%)', fontsize=14, color='green')

ax1.set_xlim(-90, 90)
ax1.set_ylim(1.3, 2.05)
ax2.set_ylim(85, 101)

ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='green')

# Grid
ax1.grid(True, alpha=0.3)
ax1.set_xticks([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])

# Title
plt.title('Leggett Inequality Test (Branciard et al. arXiv:0801.2241v2)\n' +
          f'Maximum violation at φ ≈ ±{phi_max_violation:.1f}°, requires V > {v_threshold_n3*100:.1f}%',
          fontsize=14)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=11)

# Add annotations
ax1.annotate(f'Max violation\nφ = ±{phi_max_violation:.1f}°',
            xy=(phi_max_violation, 2*np.cos(np.radians(phi_max_violation)/2)),
            xytext=(55, 1.85), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray'))

# Add text box with key thresholds
textstr = '\n'.join([
    'Key visibility thresholds:',
    f'  N=3: V > {v_threshold_n3*100:.1f}% (this test)',
    f'  N=4: V > 91.3%',
    f'  N→∞: V > 86.6%',
    '',
    'At specific angles:',
    f'  φ=30°: V > 95.3%',
    f'  φ=36.9°: V > 94.3%',
    f'  φ=45°: V > 93.5%',
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'leggett_branciard_plot.png'), dpi=150, bbox_inches='tight')
print("Saved leggett_branciard_plot.png")

# Print table of required visibilities
print("\nRequired visibility at each angle:")
print(f"{'Angle':>8} {'V_required':>12} {'L_bound':>10} {'L_QM':>10}")
print("-" * 45)
for ang in [15, 20, 25, 30, 36.9, 45, 60]:
    phi = np.radians(ang)
    bound = 2 - (2/3) * np.sin(phi/2)
    qm = 2 * np.cos(phi/2)
    v_req = bound / qm
    print(f"{ang:>8.1f}° {v_req*100:>11.2f}% {bound:>10.4f} {qm:>10.4f}")

# plt.show()  # Commented out for non-interactive use
