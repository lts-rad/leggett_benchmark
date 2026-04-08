#!/usr/bin/env python3
"""
Plot IBM Pittsburgh Leggett inequality test results comparison:
- Ideal simulation (noiseless)
- Noise model simulation
- Hardware (default layout)
- Hardware (best layout)
- Hardware (with barriers)
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
        # Handle phi_rad being negative (indicates -phi)
        if 'phi_rad' in r and r['phi_rad'] < 0:
            phi = -abs(phi)
        L3 = r.get('L3') or r.get('L3_exp')
        if L3 is not None:
            data[phi] = L3
    angles = sorted(data.keys())
    L3_values = [data[a] for a in angles]
    return angles, L3_values


def plot_ibm_comparison():
    """Create comparison plot of IBM Pittsburgh results."""

    # Load results - IBM
    ideal_results = load_results('leggett_results_ibm_sequential_SIM_noiseless_12qb.json')
    noise_results = load_results('leggett_results_ibm_ibm_pittsburgh_NOISE_MODEL_12qb.json')
    hw_results = load_results('leggett_results_ibm_ibm_pittsburgh_sequential_12qb.json')
    hw_best_results = load_results('leggett_results_ibm_ibm_pittsburgh_sequential_12qb_BEST_LAYOUT.json')
    hw_barrier_results = load_results('leggett_results_ibm_ibm_pittsburgh_sequential_barrier_12qb.json')

    # Load results - IonQ Forte (24qb)
    ionq_hw_results = load_results('../../leggett_results_ionq_forte_sequential_24qb.json')

    # Load results - IBM 24QB BEST_LAYOUT (10k shots)
    hw_best_24qb_results = load_results('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT_10k.json')

    # Load results - IBM 24QB BEST_LAYOUT rotated (10k shots)
    hw_best_24qb_rotated_results = load_results('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT_rotated_10k.json')

    # Extract data - IBM
    ideal_angles, ideal_L3 = extract_data(ideal_results)
    noise_angles, noise_L3 = extract_data(noise_results)
    hw_angles, hw_L3 = extract_data(hw_results)
    hw_best_angles, hw_best_L3 = extract_data(hw_best_results)
    hw_barrier_angles, hw_barrier_L3 = extract_data(hw_barrier_results)

    # Extract data - IonQ
    ionq_hw_angles, ionq_hw_L3 = extract_data(ionq_hw_results)

    # Extract data - IBM 24QB BEST_LAYOUT (new)
    hw_best_24qb_angles, hw_best_24qb_L3 = extract_data(hw_best_24qb_results)

    # Extract data - IBM 24QB BEST_LAYOUT (rotated/alternated layout)
    hw_best_24qb_rotated_angles, hw_best_24qb_rotated_L3 = extract_data(hw_best_24qb_rotated_results)

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
    ax.plot(phi_smooth, L_bound_smooth, 'k-', linewidth=2.5, label='Leggett Bound', zorder=1)
    ax.plot(phi_smooth, L_QM_smooth, '--', color='gray', linewidth=2, label='QM Prediction', zorder=2)

    # Define colors and markers
    ideal_color = 'blue'
    noise_color = 'orange'
    hw_color = 'blue'
    hw_best_color = 'red'
    hw_barrier_color = 'purple'
    ionq_color = 'green'
    hw_best_24qb_color = 'cyan'
    hw_best_24qb_rotated_color = '#00CED1'  # Dark turquoise (cyan-like blue)

    # Plot data - simulators hollow, hardware filled
    # Ideal simulation (hollow triangle)
    ax.scatter(ideal_angles, ideal_L3, s=100, marker='^', facecolors='none',
               edgecolors=ideal_color, linewidth=2, label='Ideal', zorder=5)

    # Noise model simulation (hollow circle)
    ax.scatter(noise_angles, noise_L3, s=100, marker='o', facecolors='none',
               edgecolors=noise_color, linewidth=2, label='Default Layout Noise Model', zorder=6)

    # Hardware default layout (filled diamond)
    ax.scatter(hw_angles, hw_L3, s=120, marker='D', color=hw_color,
               edgecolors='darkblue', linewidth=2, label='Default Layout', zorder=7)

    # Hardware best layout (filled square)
    ax.scatter(hw_best_angles, hw_best_L3, s=120, marker='s', color=hw_best_color,
               edgecolors='darkred', linewidth=2, label='Optimized Layout', zorder=8)

    # Hardware with barriers (filled pentagon)
    ax.scatter(hw_barrier_angles, hw_barrier_L3, s=120, marker='p', color=hw_barrier_color,
               edgecolors='indigo', linewidth=2, label='Serialized', zorder=9)

    # IonQ Forte hardware (filled diamond)
    ax.scatter(ionq_hw_angles, ionq_hw_L3, s=120, marker='D', color=ionq_color,
               edgecolors='darkgreen', linewidth=2, label='IonQ Forte-1 24QB', zorder=10)

    # IBM 24QB BEST_LAYOUT (filled hexagon)
    ax.scatter(hw_best_24qb_angles, hw_best_24qb_L3, s=120, marker='h', color=hw_best_24qb_color,
               edgecolors='darkcyan', linewidth=2, label='Optimized 24QB', zorder=11)

    # IBM 24QB BEST_LAYOUT ROTATED (filled star)
    ax.scatter(hw_best_24qb_rotated_angles, hw_best_24qb_rotated_L3, s=140, marker='*', color=hw_best_24qb_rotated_color,
               edgecolors='#008B8B', linewidth=1.5, label='Optimized 24QB (Alternate Layout)', zorder=12)

    # Styling
    ax.set_xlabel('Angle $\\varphi$ (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('$L_3$ Parameter', fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower center', framealpha=0.95)

    # Set axis limits
    ax.set_xlim(-90, 90)
    ax.set_ylim(1.5, 2.05)

    plt.tight_layout()

    # Save figure
    output_file = 'ibm_pittsburgh_comparison_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    plt.show()


if __name__ == "__main__":
    plot_ibm_comparison()
