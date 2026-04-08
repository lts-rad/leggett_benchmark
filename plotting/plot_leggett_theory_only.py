#!/usr/bin/env python3
"""
Plot Leggett inequality theoretical bounds only (no experimental data points).
Shows L3 Leggett bound and QM prediction curves.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_leggett_theory():
    """Create plot showing only Leggett bound and QM prediction."""

    # Create smooth curves for theory
    phi_smooth = np.linspace(-90, 90, 200)
    phi_rad_smooth = np.radians(phi_smooth)

    # Leggett bound (upper bound for Leggett model)
    L_bound_smooth = 2 - (2/3) * np.abs(np.sin(phi_rad_smooth/2))

    # Quantum mechanical prediction (pure singlet)
    L_QM_smooth = 2 * np.abs(np.cos(phi_rad_smooth/2))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot theory curves
    ax.plot(phi_smooth, L_bound_smooth, 'k-', linewidth=2, label='Leggett Bound', zorder=1)
    ax.plot(phi_smooth, L_QM_smooth, '--', color='gray', linewidth=2, label='QM Prediction', zorder=2)

    # Styling
    ax.set_xlabel('Angle φ (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('L₃ Parameter', fontsize=14, fontweight='bold')
    ax.set_title('Leggett Inequality: Theory',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)

    # Set axis limits
    ax.set_xlim(-90, 90)
    ax.set_ylim(1.5, 2.05)

    plt.tight_layout()

    # Save figure
    output_file = 'leggett_theory_only_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    plt.show()


if __name__ == "__main__":
    plot_leggett_theory()
