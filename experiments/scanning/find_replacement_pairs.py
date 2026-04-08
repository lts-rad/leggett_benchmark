#!/usr/bin/env python3
"""
Find replacement pairs for the underperforming qubits.
"""

import json

# Load viable pairs from scan
with open('viable_pairs_analysis.json', 'r') as f:
    analysis = json.load(f)

# Pairs to keep (good performers)
keep_pairs = [
    (29, 30),   # φ+30 a1b1'
    (93, 92),   # φ+30 a2b2'
    (112, 113), # φ-30 a1b1'
    (107, 97),  # φ-30 a3b3'
    (94, 95),   # φ-30 a3b3
]

# Qubits already in use
used_qubits = set()
for pair in keep_pairs:
    used_qubits.add(pair[0])
    used_qubits.add(pair[1])

print("="*70)
print("FINDING REPLACEMENT PAIRS")
print("="*70)
print(f"\nKeeping these 5 pairs:")
for pair in keep_pairs:
    print(f"  {pair}")
print(f"\nUsed qubits: {sorted(used_qubits)}")
print(f"\nNeed to find 7 replacement pairs...")

# Find best available pairs that don't use any of the kept qubits
available_pairs = []
for pair_data in analysis['viable_pairs']:
    q1, q2 = pair_data['qubits']
    # Skip if either qubit is already in use
    if q1 in used_qubits or q2 in used_qubits:
        continue
    available_pairs.append(pair_data)

print(f"\nFound {len(available_pairs)} available pairs (no qubit overlap with kept pairs)")

# Sort by estimated L3 (descending)
available_pairs.sort(key=lambda x: x['estimated_L3'], reverse=True)

print(f"\n{'='*70}")
print("TOP 10 REPLACEMENT CANDIDATES")
print(f"{'='*70}")
print(f"{'Rank':>4} {'Qubits':>12} {'Est L₃':>10} {'Avg Err':>10}")
print("-"*70)

for i, pair in enumerate(available_pairs[:10]):
    q1, q2 = pair['qubits']
    print(f"{i+1:>4} ({q1:3d},{q2:3d})     {pair['estimated_L3']:>10.4f} {pair['avg_error']:>10.4f}")

# Propose new layout
print(f"\n{'='*70}")
print("PROPOSED NEW LAYOUT (Top 7 replacements + 5 kept)")
print(f"{'='*70}")

new_layout_pairs = keep_pairs + [tuple(p['qubits']) for p in available_pairs[:7]]
new_layout_flat = []
for pair in new_layout_pairs:
    new_layout_flat.extend(pair)

print(f"\nNew 24-qubit layout (12 pairs):")
print(f"  Flat list: {new_layout_flat}")
print(f"\n  Pairs:")
for i, pair in enumerate(new_layout_pairs):
    source = "KEPT" if pair in keep_pairs else "NEW"
    print(f"    {i:2d}: {pair}  [{source}]")

# Save proposed layout
output = {
    'kept_pairs': keep_pairs,
    'replacement_pairs': [tuple(p['qubits']) for p in available_pairs[:7]],
    'new_layout_pairs': new_layout_pairs,
    'new_layout_flat': new_layout_flat
}

with open('proposed_new_layout.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved proposed layout to proposed_new_layout.json")
