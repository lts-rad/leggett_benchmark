#!/usr/bin/env python3
"""
Compare 12QB vs 24QB BEST_LAYOUT results on IBM Pittsburgh.
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def extract_data(results):
    """Extract angles and L3 values, handling both +/- angles."""
    data = {}
    for r in results:
        phi = r.get('phi_deg')
        if phi is None:
            continue
        if 'phi_rad' in r and r['phi_rad'] < 0:
            phi = -abs(phi)
        L3 = r.get('L3') or r.get('L3_exp')
        if L3 is not None:
            data[phi] = L3
    angles = sorted(data.keys())
    L3_values = [data[a] for a in angles]
    return angles, L3_values


def plot_comparison():
    """Create comparison plot."""

    # Load results
    hw_12qb = load_results('leggett_results_ibm_ibm_pittsburgh_sequential_12qb_BEST_LAYOUT.json')
    hw_24qb = load_results('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT.json')

    # Extract data
    angles_12qb, L3_12qb = extract_data(hw_12qb)
    angles_24qb, L3_24qb = extract_data(hw_24qb)

    # Create smooth curves for theory
    phi_smooth = np.linspace(-90, 90, 200)
    phi_rad_smooth = np.radians(phi_smooth)
    L_bound_smooth = 2 - (2/3) * np.abs(np.sin(phi_rad_smooth/2))
    L_QM_smooth = 2 * np.abs(np.cos(phi_rad_smooth/2))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot theory curves
    ax.plot(phi_smooth, L_bound_smooth, 'k-', linewidth=2.5, label='Leggett Bound', zorder=1)
    ax.plot(phi_smooth, L_QM_smooth, '--', color='gray', linewidth=2, label='QM Prediction', zorder=2)

    # Plot data
    ax.scatter(angles_12qb, L3_12qb, s=120, marker='D', color='blue',
               edgecolors='darkblue', linewidth=2, label='12QB BEST_LAYOUT', zorder=7)
    ax.scatter(angles_24qb, L3_24qb, s=120, marker='s', color='red',
               edgecolors='darkred', linewidth=2, label='24QB BEST_LAYOUT', zorder=8)

    # Styling
    ax.set_xlabel('Angle $\\varphi$ (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('$L_3$ Parameter', fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)

    ax.set_xlim(-90, 90)
    ax.set_ylim(1.2, 2.05)

    plt.tight_layout()

    output_file = '12qb_vs_24qb_best_layout_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print comparison
    print("\n12QB BEST_LAYOUT angles:", angles_12qb)
    print("24QB BEST_LAYOUT angles:", angles_24qb)

    plt.show()


if __name__ == "__main__":
    plot_comparison()
