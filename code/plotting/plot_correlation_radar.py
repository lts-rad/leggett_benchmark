#!/usr/bin/env python3
"""
Create radar plots for Leggett correlation measurements.
Shows the 6 correlations (C_a1b1, C_a1b1', C_a2b2, C_a2b2', C_a3b3, C_a3b3') on a radar chart.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def create_radar_plot(results, output_file='correlation_radar.png'):
    """
    Create radar plot showing correlation measurements.

    Args:
        results: List of result dictionaries with 'correlations' field
        output_file: Output filename for the plot
    """

    # Categories for the 6 correlations
    categories = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot each angle's data
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for idx, result in enumerate(results):
        phi_deg = result['phi_deg']
        correlations = result['correlations']

        # Convert correlations from [-1, 1] to [0, 1] for better visualization
        # We'll show absolute values to compare magnitudes
        values = [abs(c) for c in correlations]
        values += values[:1]  # Complete the circle

        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label=f'φ = {phi_deg:+.0f}°',
                color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)

    # Set radial limits (0 to 1 since we're showing absolute values)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add title
    plt.title('Leggett Correlation Measurements\n(Absolute Values)',
              size=16, weight='bold', pad=20)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Radar plot saved to: {output_file}")
    plt.close()

    # Also create a version with signed values (showing both positive and negative)
    create_signed_radar_plot(results, output_file.replace('.png', '_signed.png'))


def create_signed_radar_plot(results, output_file='correlation_radar_signed.png'):
    """
    Create radar plot showing signed correlation measurements.
    Maps [-1, 1] to radial axis where center=0.
    Shows ideal QM prediction and Leggett bound as reference polygons.
    """

    categories = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    # Group results by absolute value of phi (fix sign confusion)
    phi_groups = {}
    for result in results:
        # Use phi_rad to determine actual sign
        phi_rad = result['phi_rad']
        phi_deg = abs(result['phi_deg'])  # Take absolute value
        if phi_rad < 0:
            phi_deg = -phi_deg  # Apply correct sign from radians

        if phi_deg not in phi_groups:
            phi_groups[phi_deg] = result

    # Sort by angle
    sorted_phis = sorted(phi_groups.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_phis)))

    # Plot ideal and bound first (as background)
    for idx, phi_deg in enumerate(sorted_phis):
        result = phi_groups[phi_deg]
        phi_rad = result['phi_rad']
        correlations_theory = result['correlations_theory']

        # Ideal QM prediction (theory)
        values_ideal = [(c + 1) for c in correlations_theory]
        values_ideal += values_ideal[:1]

        ax.plot(angles, values_ideal, ':', linewidth=2.5,
                color=colors[idx], alpha=0.4, label=f'φ = {phi_deg:+.0f}° (QM ideal)')

        # Leggett bound - all correlations should be same for a given phi
        # For bound, we use: bound = 2 - (2/3)*|sin(phi/2)|
        # But this is for L3, not individual correlations
        # For individual correlations, the bound depends on the specific geometry
        # We'll skip plotting individual correlation bounds as they're complex

    # Now plot experimental data (on top, transparent)
    for idx, phi_deg in enumerate(sorted_phis):
        result = phi_groups[phi_deg]
        correlations = result['correlations']

        # Experimental values
        values = [(c + 1) for c in correlations]  # Map to [0, 2]
        values += values[:1]

        # Plot experimental with transparency
        ax.plot(angles, values, 'o-', linewidth=2, label=f'φ = {phi_deg:+.0f}° (exp)',
                color=colors[idx], markersize=8, alpha=0.8)
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=14, weight='bold')

    # Zoom in on the relevant range: correlations are near -1
    # Map [-1, -0.6] to radial range for better visibility
    # In shifted coordinates: -1 -> 0, -0.6 -> 0.4
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(['-1.0', '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4'], size=11)

    ax.grid(True, linestyle='--', alpha=0.3)

    plt.title('Leggett Correlation Measurements\n(Solid = Experiment, Dotted = QM Theory)',
              size=16, weight='bold', pad=20)

    # Reorder legend: group exp and theory together
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Signed radar plot saved to: {output_file}")
    plt.close()


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 plot_correlation_radar.py RESULTS_FILE [OUTPUT_FILE]")
        print("Example: python3 plot_correlation_radar.py leggett_results.json radar.png")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'correlation_radar.png'

    print(f"Loading results from: {input_file}")
    results = load_results(input_file)

    print(f"Loaded {len(results)} angle measurements")
    for r in results:
        print(f"  φ = {r['phi_deg']:+.0f}°: L₃ = {r['L3']:.4f}, violated = {r['violated']}")

    print(f"\nCreating radar plots...")
    create_radar_plot(results, output_file)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
