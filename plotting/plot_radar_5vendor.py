import os
#!/usr/bin/env python3
"""
Radar plot comparing L3 values across 5 quantum vendors/emulators
for Leggett inequality testing at multiple angles.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_l3_by_angle(data, target_angles):
    """Extract L3 values for specific angles, averaging +/- pairs"""
    l3_values = {}
    for angle in target_angles:
        # Find both +angle and -angle entries
        pos_entries = [d for d in data if abs(d['phi_deg'] - angle) < 0.1]
        neg_entries = [d for d in data if abs(d['phi_deg'] + angle) < 0.1]

        all_l3 = []
        for e in pos_entries + neg_entries:
            all_l3.append(e['L3'])

        if all_l3:
            l3_values[angle] = np.mean(all_l3)

    return l3_values

def main():
    # Target angles
    angles = [15, 25, 30, 45, 60]

    # Theoretical values
    theory = {}
    bounds = {}
    for angle in angles:
        phi = np.radians(angle)
        theory[angle] = 2 * abs(np.cos(phi))
        bounds[angle] = 2 / (1 + abs(np.sin(phi)))

    # Load data files
    ibm_data = load_json('../paper/ibm/leggett_results_ibm_ibm_pittsburgh_NOISE_MODEL_12qb_BEST_LAYOUT.json')
    ionq_data = load_json('leggett_results_ionq_forte_NOISE_MODEL_24qb.json')
    rigetti_data = load_json('leggett_results_rigetti_ankaa3_emulator_NOISE_BEST.json')
    iqm_data = load_json('leggett_results_iqm_emerald_emulator_NOISE_BEST.json')
    quantinuum_data = load_json('leggett_results_azure_sim_h2-1_sequential_24qb.json')

    # Extract L3 values
    ibm_l3 = get_l3_by_angle(ibm_data, angles)
    ionq_l3 = get_l3_by_angle(ionq_data, angles)
    rigetti_l3 = get_l3_by_angle(rigetti_data, angles)
    iqm_l3 = get_l3_by_angle(iqm_data, angles)
    quantinuum_l3 = get_l3_by_angle(quantinuum_data, angles)

    # Prepare radar chart
    categories = [f'±{a}°' for a in angles]
    N = len(categories)

    # Create angle positions for radar chart
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    theta += theta[:1]  # Complete the loop

    # Prepare data lists (add first value at end to close the polygon)
    theory_vals = [theory[a] for a in angles] + [theory[angles[0]]]
    bounds_vals = [bounds[a] for a in angles] + [bounds[angles[0]]]
    ibm_vals = [ibm_l3.get(a, 0) for a in angles] + [ibm_l3.get(angles[0], 0)]
    ionq_vals = [ionq_l3.get(a, 0) for a in angles] + [ionq_l3.get(angles[0], 0)]
    rigetti_vals = [rigetti_l3.get(a, 0) for a in angles] + [rigetti_l3.get(angles[0], 0)]
    iqm_vals = [iqm_l3.get(a, 0) for a in angles] + [iqm_l3.get(angles[0], 0)]
    quantinuum_vals = [quantinuum_l3.get(a, 0) for a in angles] + [quantinuum_l3.get(angles[0], 0)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot bounds (filled area below)
    ax.fill(theta, bounds_vals, alpha=0.15, color='red', label='L3 Bound (Classical)')
    ax.plot(theta, bounds_vals, color='red', linewidth=2, linestyle='--')

    # Plot theory
    ax.plot(theta, theory_vals, color='black', linewidth=2, linestyle='-', label='QM Theory', marker='o', markersize=6)

    # Plot vendor data
    ax.plot(theta, quantinuum_vals, color='purple', linewidth=2, linestyle='-', label='Quantinuum H2-1 (Emulator)', marker='s', markersize=8)
    ax.plot(theta, ionq_vals, color='blue', linewidth=2, linestyle='-', label='IonQ Forte-1 (Noise Sim)', marker='^', markersize=8)
    ax.plot(theta, ibm_vals, color='green', linewidth=2, linestyle='-', label='IBM Pittsburgh (Noise Sim)', marker='D', markersize=8)
    ax.plot(theta, iqm_vals, color='orange', linewidth=2, linestyle='-', label='IQM Emerald (Emulator)', marker='v', markersize=8)
    ax.plot(theta, rigetti_vals, color='cyan', linewidth=2, linestyle='-', label='Rigetti Ankaa-3 (Emulator)', marker='p', markersize=8)

    # Set category labels
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    # Set radial limits
    ax.set_ylim(1.5, 2.0)
    ax.set_yticks([1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

    # Title and legend
    ax.set_title('Leggett L3 Values: 5-Vendor Comparison\n(Noise Simulations & Emulators)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'leggett_radar_5vendor.png'), dpi=150, bbox_inches='tight')
    plt.savefig('leggett_radar_5vendor.pdf', bbox_inches='tight')
    print("Saved: leggett_radar_5vendor.png and .pdf")

    # Print summary table
    print("\n" + "="*80)
    print("L3 Summary (Noise Simulations)")
    print("="*80)
    print(f"{'Angle':<8} {'Bound':<8} {'Theory':<8} {'IBM':<8} {'IonQ':<8} {'Rigetti':<8} {'IQM':<8} {'Quantinuum':<8}")
    print("-"*80)
    for a in angles:
        print(f"±{a}°{'':<4} {bounds[a]:<8.3f} {theory[a]:<8.3f} {ibm_l3.get(a,0):<8.3f} {ionq_l3.get(a,0):<8.3f} {rigetti_l3.get(a,0):<8.3f} {iqm_l3.get(a,0):<8.3f} {quantinuum_l3.get(a,0):<8.3f}")

    # Calculate Fc percentages
    print("\n" + "="*80)
    print("Fc% (L3_noise / L3_theory * 100)")
    print("="*80)
    print(f"{'Angle':<8} {'IBM':<8} {'IonQ':<8} {'Rigetti':<8} {'IQM':<8} {'Quantinuum':<8}")
    print("-"*80)
    for a in angles:
        ibm_fc = ibm_l3.get(a, 0) / theory[a] * 100
        ionq_fc = min(ionq_l3.get(a, 0) / theory[a] * 100, 100)  # Cap at 100%
        rigetti_fc = rigetti_l3.get(a, 0) / theory[a] * 100
        iqm_fc = iqm_l3.get(a, 0) / theory[a] * 100
        quantinuum_fc = min(quantinuum_l3.get(a, 0) / theory[a] * 100, 100)  # Cap at 100%
        print(f"±{a}°{'':<4} {ibm_fc:<8.1f} {ionq_fc:<8.1f} {rigetti_fc:<8.1f} {iqm_fc:<8.1f} {quantinuum_fc:<8.1f}")

if __name__ == "__main__":
    main()
