#!/usr/bin/env python3
"""
Radar plot showing top 20 performing pairs from 64-pair analysis at φ = ±30°.
Groups pairs by correlation type to form complete hexagons.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from qiskit_ibm_runtime import QiskitRuntimeService

# Set up high-quality rendering
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica Neue', 'Helvetica', 'Arial']
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.labelweight'] = 'light'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

ADJACENT_PAIRS_64 = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (16, 23), (17, 27), (18, 31), (19, 35), (20, 21), (24, 25), (28, 29), (32, 33),
    (36, 41), (37, 45), (38, 49), (39, 53), (42, 43), (46, 47), (50, 51), (54, 55),
    (56, 63), (57, 67), (58, 71), (59, 75), (60, 61), (64, 65), (68, 69), (72, 73),
    (76, 81), (77, 85), (78, 89), (79, 93), (82, 83), (86, 87), (90, 91), (94, 95),
    (96, 103), (97, 107), (98, 111), (99, 115), (100, 101), (104, 105), (108, 109), (112, 113),
    (116, 121), (117, 125), (118, 129), (119, 133), (122, 123), (126, 127), (130, 131), (134, 135),
    (136, 143), (137, 147), (138, 151), (139, 155), (140, 141), (144, 145), (148, 149), (152, 153)
]


def load_jobs_from_json(json_file):
    """Load unique job IDs and angles from JSON results file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    seen = set()
    jobs = []
    for entry in data:
        job_id = entry['job_id']
        phi_deg = abs(entry['phi_deg'])
        if (job_id, phi_deg) not in seen:
            seen.add((job_id, phi_deg))
            jobs.append((job_id, phi_deg))

    jobs.sort(key=lambda x: x[1])
    return jobs


def extract_pair_correlations(counts, num_pairs=64):
    num_qubits = num_pairs * 2
    pair_corrs = [0.0] * num_pairs
    total = sum(counts.values())
    for bitstring, count in counts.items():
        for p in range(num_pairs):
            a = int(bitstring[num_qubits - 1 - 2*p])
            b = int(bitstring[num_qubits - 2 - 2*p])
            if a == b:
                pair_corrs[p] += count
            else:
                pair_corrs[p] -= count
    return [c / total for c in pair_corrs]


def get_theoretical_correlation(phi_deg, corr_type):
    """Get theoretical correlation for given angle and correlation type."""
    phi_rad = np.radians(phi_deg if corr_type < 6 else -phi_deg)
    return -np.cos(phi_rad / 2)


def main():
    json_file = "leggett_results_ibm_ibm_pittsburgh_64pairs_20251208_141241_10k.json"

    print(f"Loading jobs from: {json_file}")
    jobs = load_jobs_from_json(json_file)

    # Fetch all job results
    service = QiskitRuntimeService()
    all_pair_corrs = {}

    for job_id, phi_deg in jobs:
        print(f"Fetching {phi_deg}°...", end=" ", flush=True)
        job = service.job(job_id)
        result = job.result()
        counts = dict(result[0].data.meas.get_counts())
        all_pair_corrs[phi_deg] = extract_pair_correlations(counts)
    print("Done!\n")

    angles = sorted(all_pair_corrs.keys())

    # Organize pairs by correlation type
    # Each pair measures one of 12 correlation types (0-5 for +phi, 6-11 for -phi)
    pairs_by_type = {i: [] for i in range(12)}

    for p in range(64):
        corr_type = p % 12

        # Calculate average distance from QM theory for this pair
        distances = []
        for phi in angles:
            corr = all_pair_corrs[phi][p]
            expected = get_theoretical_correlation(phi, corr_type)
            distances.append(abs(corr - expected))

        avg_distance = np.mean(distances)
        corr_at_30 = all_pair_corrs[30][p]

        pairs_by_type[corr_type].append({
            'pair_idx': p,
            'qubits': ADJACENT_PAIRS_64[p],
            'corr_type': corr_type,
            'avg_distance': avg_distance,
            'corr_at_30': corr_at_30
        })

    # Sort each type by quality
    for t in range(12):
        pairs_by_type[t].sort(key=lambda x: x['avg_distance'])

    # Build "best" hexagons by selecting best pair for each correlation type
    # We'll show top 3 selections for each of +phi and -phi

    # Labels for 6 correlation types
    labels = ['C(a₁,b₁)', 'C(a₁,b₁\')', 'C(a₂,b₂)', 'C(a₂,b₂\')', 'C(a₃,b₃)', 'C(a₃,b₃\')']
    N = len(labels)
    radar_angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    radar_angles += radar_angles[:1]

    # Theory values for ±30°
    theory_val = -np.cos(np.radians(30) / 2)  # ≈ -0.966
    L3_theory = 2 * abs(theory_val)  # ≈ 1.932

    # Leggett bound at 30°
    L3_bound = 2 - (2/3) * abs(np.sin(np.radians(30)/2))  # ≈ 1.827
    corr_bound = -L3_bound / 2.0  # ≈ -0.914

    # Colors for different selections (10 shades each)
    colors_pos = plt.cm.Blues(np.linspace(0.9, 0.3, 10))  # Dark to light blues
    colors_neg = plt.cm.Purples(np.linspace(0.9, 0.3, 10))  # Dark to light purples

    text_outline = [pe.withStroke(linewidth=2, foreground='white')]

    # Create figure with two subplots: +30° and -30°
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), subplot_kw=dict(projection='polar'), facecolor='white')

    for ax_idx, (ax, phi_sign, title_sign, type_range, colors) in enumerate([
        (axes[0], '+', '+', range(6), colors_pos),
        (axes[1], '-', '−', range(6, 12), colors_neg)
    ]):
        ax.set_facecolor('#FAFBFC')

        # Normalization - zoom in by using tighter range
        center = -1.0
        outer = -0.85  # Zoomed in from -0.6

        # Plot theory hexagon
        theory_plot = [theory_val] * 7
        norm_theory = [(v - center) / (outer - center) for v in theory_plot]
        ax.plot(radar_angles, norm_theory, '--', linewidth=2.5, dashes=(8, 4),
                color='#3498DB', alpha=0.95, zorder=5, label=f'QM Theory (L₃={L3_theory:.3f})')

        # Plot Leggett bound hexagon
        bound_plot = [corr_bound] * 7
        norm_bound = [(v - center) / (outer - center) for v in bound_plot]
        ax.plot(radar_angles, norm_bound, '--', linewidth=2.5, dashes=(4, 2),
                color='#7F8C8D', alpha=0.9, zorder=4, label=f'Leggett Bound (L₃={L3_bound:.3f})')

        # Plot top 10 "best" hexagons (best pair from each of 6 correlation types)
        for rank in range(10):
            correlations = []
            pair_labels = []

            for spoke_idx, corr_type in enumerate(type_range):
                if rank < len(pairs_by_type[corr_type]):
                    best_pair = pairs_by_type[corr_type][rank]
                    correlations.append(best_pair['corr_at_30'])
                    pair_labels.append(f"{best_pair['qubits'][0]}-{best_pair['qubits'][1]}")
                else:
                    correlations.append(-0.9)  # fallback
                    pair_labels.append("N/A")

            # Close the hexagon
            correlations_plot = correlations + [correlations[0]]
            norm_corrs = [(v - center) / (outer - center) for v in correlations_plot]

            # Calculate L3 for this selection
            L3 = (1/3) * (abs(correlations[0] + correlations[1]) +
                         abs(correlations[2] + correlations[3]) +
                         abs(correlations[4] + correlations[5]))

            # Plot hexagon
            color = colors[rank]
            linewidth = max(1.0, 2.5 - rank*0.15)
            markersize = max(4, 10 - rank*0.6)
            # Only add label for first 5 to keep legend manageable
            label = f'Best #{rank+1} (L₃={L3:.3f})' if rank < 5 else None
            ax.plot(radar_angles, norm_corrs, '-', linewidth=linewidth,
                    color=color, alpha=0.9 - rank*0.05, zorder=8-rank,
                    label=label)
            ax.plot(radar_angles[:-1], norm_corrs[:-1], 'o', markersize=markersize,
                    markerfacecolor=color, markeredgecolor='white',
                    markeredgewidth=1.0, alpha=0.95 - rank*0.05, zorder=9-rank)

            # Add qubit pair labels for best selection only
            if rank == 0:
                for i, pair_label in enumerate(pair_labels):
                    ax.annotate(pair_label, (radar_angles[i], norm_corrs[i]),
                                fontsize=7, fontweight='bold', color=color, ha='center', va='bottom',
                                xytext=(0, 12), textcoords='offset points', zorder=20,
                                path_effects=text_outline)

        # Axis labels
        ax.set_xticks(radar_angles[:-1])
        ax.set_xticklabels(labels, size=10, weight='semibold', color='#2C3E50')

        # Y-axis setup
        ax.set_ylim(0, 1.1)
        tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
        tick_labels = [f'{center + t * (outer - center):.2f}' for t in tick_positions]
        ax.set_rticks(tick_positions)
        ax.set_yticklabels(tick_labels, size=9, color='#444444', fontweight='medium')

        # Grid styling
        ax.grid(True, linestyle='-', alpha=0.35, linewidth=1.0, color='#BDC3C7')
        ax.spines['polar'].set_color('#D5DBDB')
        ax.spines['polar'].set_linewidth(1.5)

        # Title
        ax.set_title(f'ϕ = {title_sign}30°', size=16, weight='bold', pad=15, color='#2C3E50')

        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=10)

    # Main title
    fig.suptitle('Best Pair Selections from 64-Pair Analysis at φ = ±30°\n(IBM Pittsburgh, 10k shots)',
                 size=18, weight='bold', color='#2C3E50', y=1.02)

    plt.tight_layout()

    output_file = 'radar_64pairs_top20_30deg.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved to {output_file}')

    # Print best pairs by type
    print("\nBest pair for each correlation type:")
    for t in range(12):
        sign = '+φ' if t < 6 else '-φ'
        spoke = t % 6
        best = pairs_by_type[t][0]
        print(f"  Type {t:2} ({sign} spoke {spoke}): {str(best['qubits']):<12} "
              f"C@30°={best['corr_at_30']:.4f} avgΔ={best['avg_distance']:.4f}")

    plt.close()


if __name__ == "__main__":
    main()
