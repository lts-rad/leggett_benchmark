#!/usr/bin/env python3
"""
Create individual radar plots for each angle showing:
- Experimental correlations
- QM ideal prediction
- Leggett bound
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def detect_bell_state(correlations):
    """
    Detect which Bell state was used based on correlation signs.

    Returns:
        'singlet' for |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 (all negative)
        'psi_plus' for |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 (all positive, opposite of singlet)
        'phi_plus' for |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 (pattern: +,+,-,-,+,+)
        'phi_minus' for |Φ⁻⟩ = (|00⟩ - |11⟩)/√2 (pattern: -,-,+,+,+,+)
        'unknown' otherwise
    """
    num_positive = sum(1 for c in correlations if c > 0)
    num_negative = sum(1 for c in correlations if c < 0)

    # Get sign pattern
    signs = [1 if c > 0 else -1 if c < 0 else 0 for c in correlations]

    if num_negative == 6:
        return 'singlet'  # |Ψ⁻⟩ = (|01⟩-|10⟩)/√2
    elif num_positive == 6:
        return 'psi_plus'  # |Ψ⁺⟩ = (|01⟩+|10⟩)/√2
    elif num_positive == 4 and num_negative == 2:
        # Distinguish |Φ⁺⟩ from |Φ⁻⟩ by looking at first two correlations
        if signs[0] > 0 and signs[1] > 0:
            return 'phi_plus'  # |Φ⁺⟩: C(a₁,b₁) and C(a₁,b₁') are positive
        else:
            return 'phi_minus'  # |Φ⁻⟩: C(a₁,b₁) and C(a₁,b₁') are negative
    else:
        return 'unknown'


def get_theory_correlations_for_bell_state(bell_state, phi_rad):
    """
    Calculate theoretical correlations for a given Bell state and angle.

    For |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2, correlations have opposite sign compared to |Ψ⁻⟩
    """
    # Alice's directions
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])

    # Bob's directions
    b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    # Base correlations (for singlet |Ψ⁻⟩)
    C_a1b1 = -np.dot(a1, b1)
    C_a1b1p = -np.dot(a1, b1_prime)
    C_a2b2 = -np.dot(a2, b2)
    C_a2b2p = -np.dot(a2, b2_prime)
    C_a3b3 = -np.dot(a3, b3)
    C_a3b3p = -np.dot(a3, b3_prime)

    if bell_state == 'singlet':
        # |Ψ⁻⟩: C(a,b) = -a·b
        return [C_a1b1, C_a1b1p, C_a2b2, C_a2b2p, C_a3b3, C_a3b3p]
    elif bell_state == 'psi_plus':
        # |Ψ⁺⟩: First 4 correlations flip sign, last 2 stay negative
        return [-C_a1b1, -C_a1b1p, -C_a2b2, -C_a2b2p, C_a3b3, C_a3b3p]
    elif bell_state == 'phi_plus':
        # |Φ⁺⟩ = (|00⟩+|11⟩)/√2: All positive
        return [-C_a1b1, -C_a1b1p, -C_a2b2, -C_a2b2p, -C_a3b3, -C_a3b3p]
    elif bell_state == 'phi_minus':
        # |Φ⁻⟩ = (|00⟩-|11⟩)/√2: All negative (same as singlet)
        return [C_a1b1, C_a1b1p, C_a2b2, C_a2b2p, C_a3b3, C_a3b3p]
    else:
        # Unknown, return singlet
        return [C_a1b1, C_a1b1p, C_a2b2, C_a2b2p, C_a3b3, C_a3b3p]


def create_individual_radar_plots(results, output_prefix='radar', bell_state='singlet'):
    """
    Create separate radar plot for each angle.

    Shows:
    - Experimental data (solid line with fill)
    - QM ideal prediction (dashed line)
    - Leggett bound (dotted line)

    Args:
        results: List of measurement results
        output_prefix: Prefix for output files
        bell_state: Which Bell state to use ('singlet', 'psi_plus', 'phi_plus', 'phi_minus')
    """

    categories = ['C(a₁,b₁)', "C(a₁,b₁')", 'C(a₂,b₂)', "C(a₂,b₂')", 'C(a₃,b₃)', "C(a₃,b₃')"]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Group results by absolute value of phi (fix sign confusion)
    phi_groups = {}
    for result in results:
        # Use phi_rad to determine actual sign
        phi_rad = result['phi_rad']
        phi_deg = abs(result['phi_deg'])
        if phi_rad < 0:
            phi_deg = -phi_deg

        if phi_deg not in phi_groups:
            phi_groups[phi_deg] = []
        phi_groups[phi_deg].append(result)

    # Create one plot per unique angle
    for phi_deg in sorted(set([abs(p) for p in phi_groups.keys()])):
        # Get both +phi and -phi results if they exist
        results_pos = phi_groups.get(phi_deg, [])
        results_neg = phi_groups.get(-phi_deg, [])

        if not results_pos and not results_neg:
            continue

        # Average if we have multiple measurements at this angle
        # Get reference values
        reference_result = results_pos[0] if results_pos else results_neg[0]
        bound_value = reference_result['bound']
        L3_theory = reference_result['L3_theory']
        phi_rad = reference_result['phi_rad']
        print(f"\n=== Creating plot for phi_deg={phi_deg:.0f} ===")
        print(f"Using phi_rad={phi_rad:.4f} ({np.degrees(phi_rad):.1f}°), phi_rad/2={phi_rad/2:.4f} ({np.degrees(phi_rad/2):.1f}°)")

        # Get correct theory correlations for this Bell state
        correlations_theory = get_theory_correlations_for_bell_state(bell_state, phi_rad)

        # Determine Bell state label
        bell_state_label = {
            'singlet': '|Ψ⁻⟩ = (|01⟩-|10⟩)/√2',
            'psi_plus': '|Ψ⁺⟩ = (|01⟩+|10⟩)/√2',
            'phi_plus': '|Φ⁺⟩ = (|00⟩+|11⟩)/√2',
            'phi_minus': '|Φ⁻⟩ = (|00⟩-|11⟩)/√2',
            'unknown': 'Unknown state'
        }.get(bell_state, 'Unknown')

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # QM ideal values (same for both +phi and -phi)
        values_qm = [(c + 1) for c in correlations_theory]
        values_qm += values_qm[:1]

        # Leggett bound - calculate classical correlation pattern that achieves L3 = bound
        # For local realistic theories, correlations follow C(θ) where θ is angle between directions
        # The bound L3 = 2 - (2/3)|sin(φ/2)| is achieved by classical correlations
        # We'll compute correlations that give this bound value

        # Leggett bound is on L3, not individual correlations
        # Don't plot individual classical correlations since the model is unclear

        # Plot QM ideal (middle layer)
        ax.plot(angles, values_qm, '--', linewidth=3, color='blue',
                alpha=0.7, label=f'QM Ideal (L₃={L3_theory:.3f})')
        ax.fill(angles, values_qm, alpha=0.1, color='blue')

        # Plot experimental data for +phi
        if results_pos:
            avg_correlations_pos = np.mean([r['correlations'] for r in results_pos], axis=0)
            values_exp_pos = [(c + 1) for c in avg_correlations_pos]
            values_exp_pos += values_exp_pos[:1]

            L3_exp_pos = (1/3) * (abs(avg_correlations_pos[0] + avg_correlations_pos[1]) +
                                  abs(avg_correlations_pos[2] + avg_correlations_pos[3]) +
                                  abs(avg_correlations_pos[4] + avg_correlations_pos[5]))

            ax.plot(angles, values_exp_pos, 'o-', linewidth=3, color='green',
                    markersize=10, alpha=0.9, label=f'φ = +{phi_deg:.0f}° (L₃={L3_exp_pos:.3f})')
            ax.fill(angles, values_exp_pos, alpha=0.1, color='green')

        # Plot experimental data for -phi
        if results_neg:
            avg_correlations_neg = np.mean([r['correlations'] for r in results_neg], axis=0)
            values_exp_neg = [(c + 1) for c in avg_correlations_neg]
            values_exp_neg += values_exp_neg[:1]

            L3_exp_neg = (1/3) * (abs(avg_correlations_neg[0] + avg_correlations_neg[1]) +
                                  abs(avg_correlations_neg[2] + avg_correlations_neg[3]) +
                                  abs(avg_correlations_neg[4] + avg_correlations_neg[5]))

            ax.plot(angles, values_exp_neg, 's-', linewidth=3, color='orange',
                    markersize=10, alpha=0.9, label=f'φ = -{phi_deg:.0f}° (L₃={L3_exp_neg:.3f})')
            ax.fill(angles, values_exp_neg, alpha=0.1, color='orange')

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=14, weight='bold')

        # Set radial limits - zoom to data range
        all_values = values_qm.copy()
        if results_pos:
            all_values.extend(values_exp_pos)
        if results_neg:
            all_values.extend(values_exp_neg)

        min_val = min(all_values) - 0.05
        max_val = max(all_values) + 0.05

        ax.set_ylim(min_val, max_val)

        # Create appropriate tick labels
        num_ticks = 7
        tick_vals = np.linspace(min_val, max_val, num_ticks)
        tick_labels = [f'{(v-1):.2f}' for v in tick_vals]  # Convert back to [-1,1] range
        ax.set_yticks(tick_vals)
        ax.set_yticklabels(tick_labels, size=11)

        ax.grid(True, linestyle='--', alpha=0.3)

        # Title
        title = f'Leggett Correlation Measurements: φ = ±{abs(phi_deg):.0f}° ({bell_state_label})\n'
        title += f'QM Theory: L₃ = {L3_theory:.4f} | Leggett Bound: L₃ ≤ {bound_value:.4f}'

        plt.title(title, size=13, weight='bold', pad=20)

        # Legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=12)

        plt.tight_layout()

        output_file = f'{output_prefix}_phi_{abs(phi_deg):.0f}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def main():
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python3 plot_correlation_radar_individual.py RESULTS_FILE [OUTPUT_PREFIX]")
        print("Example: python3 plot_correlation_radar_individual.py results.json radar")
        sys.exit(1)

    input_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'radar'

    # Detect Bell state from filename
    basename = os.path.basename(input_file).lower()
    if 'b01p10' in basename:
        bell_state = 'psi_plus'  # |Ψ⁺⟩ = (|01⟩+|10⟩)/√2
    elif 'b0011' in basename:
        bell_state = 'phi_plus'  # |Φ⁺⟩ = (|00⟩+|11⟩)/√2
    elif 'b00n11' in basename:
        bell_state = 'phi_minus'  # |Φ⁻⟩ = (|00⟩-|11⟩)/√2
    else:
        bell_state = 'singlet'  # |Ψ⁻⟩ = (|01⟩-|10⟩)/√2

    print(f"Loading results from: {input_file}")
    print(f"Detected Bell state: {bell_state}")
    results = load_results(input_file)

    print(f"Loaded {len(results)} measurements")

    print(f"\nCreating individual radar plots...")
    create_individual_radar_plots(results, output_prefix, bell_state)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
