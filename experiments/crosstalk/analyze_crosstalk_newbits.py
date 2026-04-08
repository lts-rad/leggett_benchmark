#!/usr/bin/env python3
"""
Analyze crosstalk data using BEST_LAYOUT_newbits qubit pairs.

Layout: [86, 87, 113, 114, 99, 115, 60, 61, 37, 45, 46, 47, 11, 18, 147, 148, 100, 101, 40, 41, 97, 107, 84, 85]

+phi pairs (0-5): (86,87), (113,114), (99,115), (60,61), (37,45), (46,47)
-phi pairs (6-11): (11,18), (147,148), (100,101), (40,41), (97,107), (84,85)
"""

import json
import numpy as np

# Load BEST_LAYOUT_newbits results
with open('leggett_results_ibm_ibm_pittsburgh_sequential_24qb_BEST_LAYOUT_newbits.json') as f:
    data_leggett = json.load(f)

# Load crosstalk experiment results
with open('crosstalk_yy_experiment.json') as f:
    data_crosstalk = json.load(f)

print("=" * 70)
print("BEST_LAYOUT_newbits QUBIT PAIRS")
print("=" * 70)
layout = [86, 87, 113, 114, 99, 115, 60, 61, 37, 45, 46, 47, 11, 18, 147, 148, 100, 101, 40, 41, 97, 107, 84, 85]
pairs = [(layout[i], layout[i+1]) for i in range(0, len(layout), 2)]
print(f"\nAll 12 pairs:")
for i, p in enumerate(pairs):
    phi_sign = "+phi" if i < 6 else "-phi"
    print(f"  Pair {i}: {p} ({phi_sign})")

print("\n" + "=" * 70)
print("LEGGETT RESULTS (phi=±30°)")
print("=" * 70)

for entry in data_leggett:
    if abs(entry['phi_deg']) == 30:
        print(f"\nphi = {entry['phi_deg']:+d}°:")
        print(f"  L3 = {entry['L3']:.4f} (bound: {entry['bound']:.4f}, violated: {entry['violated']})")
        print(f"  Correlations:")
        if entry['phi_deg'] > 0:
            pair_list = pairs[:6]
        else:
            pair_list = pairs[6:]
        for i, (corr, pair) in enumerate(zip(entry['correlations'], pair_list)):
            print(f"    {pair}: {corr:.3f}")

print("\n" + "=" * 70)
print("CROSSTALK EXPERIMENT RESULTS")
print("=" * 70)

print(f"\nTarget pair: {tuple(data_crosstalk['target_pair'])}")
print(f"Shots: {data_crosstalk['num_shots']}")

print("\nGroup results:")
for group_name in ['group_A', 'group_B', 'group_C', 'group_D']:
    g = data_crosstalk['groups'][group_name]
    other_pairs = [tuple(p) for p in g['pairs'] if p != [11, 18]]
    print(f"\n  {group_name}:")
    print(f"    Other pairs: {other_pairs}")
    print(f"    (11,18) YY correlation: {g['target_yy']:.3f}")

print(f"\nSummary:")
print(f"  Mean (11,18) YY across groups: {data_crosstalk['summary']['mean']:.3f}")
print(f"  Std: {data_crosstalk['summary']['std']:.3f}")

# Verify pairs match
print("\n" + "=" * 70)
print("VERIFICATION: Do crosstalk pairs match BEST_LAYOUT_newbits?")
print("=" * 70)

crosstalk_pairs = set()
for group_name in ['group_A', 'group_B', 'group_C', 'group_D']:
    g = data_crosstalk['groups'][group_name]
    for p in g['pairs']:
        crosstalk_pairs.add(tuple(sorted(p)))

layout_pairs = set(tuple(sorted(p)) for p in pairs)

print(f"\nCrosstalk experiment pairs: {sorted(crosstalk_pairs)}")
print(f"\nBEST_LAYOUT_newbits pairs: {sorted(layout_pairs)}")
print(f"\nMatch: {crosstalk_pairs == layout_pairs}")
