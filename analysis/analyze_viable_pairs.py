#!/usr/bin/env python3
"""
Analyze qubit pair scan to determine which pairs would violate Leggett inequality.
"""

import json
import numpy as np

# Load scan results
with open('qubit_pair_scan_phi30.json', 'r') as f:
    results = json.load(f)

print("="*70)
print("ANALYZING VIABLE PAIRS FOR LEGGETT VIOLATION")
print("="*70)

phi_deg = 30
phi_rad = np.radians(phi_deg)

# Theoretical values at φ=30°
# Alice on y-axis: a2 = [0, 1, 0]
# Bob at ±φ/2: b2 = [0, cos(φ/2), sin(φ/2)]
a2 = np.array([0, 1, 0])
b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])

corr_b2_theory = -np.dot(a2, b2)
corr_b2p_theory = -np.dot(a2, b2_prime)

print(f"\nTheoretical correlations at φ={phi_deg}°:")
print(f"  C(a2, b2)  = {corr_b2_theory:.6f}")
print(f"  C(a2, b2') = {corr_b2p_theory:.6f}")

# For full Leggett test at φ=30°:
# L₃ = (|C1| + |C2| + |C3| + |C4| + |C5| + |C6|) / 3
# Since all 6 correlations should be approximately equal:
L3_theory = 2 * abs(corr_b2_theory)
# Leggett bound from actual test data at φ=30°
bound = 1.8274539699316528

print(f"\nFull Leggett test expectations:")
print(f"  L₃ (theory) = {L3_theory:.6f}")
print(f"  L₃ bound    = {bound:.6f}")
print(f"  Need L₃ > {bound:.4f} for violation")

# Analyze each pair
# We measured 2 out of 6 correlations
# Extrapolate: if avg of these 2 is X, then L₃ ≈ 6*X
viable_pairs = []

print(f"\nScanning {len(results)} pairs...")
print(f"{'Qubits':>12} {'Avg Corr':>10} {'Est L₃':>10} {'Violates?':>12} {'Avg Err':>10}")
print("-"*70)

for r in results:
    q1, q2 = r['qubits']
    c1, c2 = r['correlations']
    avg_error = r['avg_error']

    # Extrapolate to full L₃ (assuming all 6 correlations similar)
    # L₃ = (|C1| + |C2| + ... + |C6|) / 3
    # We measured 2, so: estimated_L₃ = (|C1| + |C2|) * (6/2) / 3 = (|C1| + |C2|)
    estimated_L3 = abs(c1) + abs(c2)

    # Check if would violate
    would_violate = estimated_L3 > bound

    if would_violate:
        viable_pairs.append({
            'qubits': [q1, q2],
            'estimated_L3': estimated_L3,
            'avg_error': avg_error,
            'correlations': r['correlations']
        })

# Sort by estimated L₃ (descending)
viable_pairs.sort(key=lambda x: x['estimated_L3'], reverse=True)

print(f"\nFound {len(viable_pairs)} pairs that would violate Leggett bound!")

# Count unique qubits
unique_qubits = set()
for pair in viable_pairs:
    unique_qubits.add(pair['qubits'][0])
    unique_qubits.add(pair['qubits'][1])

print(f"These pairs involve {len(unique_qubits)} unique qubits")

print(f"\n{'='*70}")
print(f"TOP 30 VIABLE PAIRS (by estimated L₃)")
print(f"{'='*70}")
print(f"{'Rank':>4} {'Qubits':>12} {'Est L₃':>10} {'Margin':>10} {'Avg Err':>10}")
print("-"*70)

for i, pair in enumerate(viable_pairs[:30]):
    q1, q2 = pair['qubits']
    margin = pair['estimated_L3'] - bound
    print(f"{i+1:>4} ({q1:3d},{q2:3d})     {pair['estimated_L3']:>10.4f} {margin:>+10.4f} {pair['avg_error']:>10.4f}")

print(f"\n{'='*70}")
print(f"BOTTOM 20 VIABLE PAIRS (weakest violations)")
print(f"{'='*70}")
print(f"{'Rank':>4} {'Qubits':>12} {'Est L₃':>10} {'Margin':>10} {'Avg Err':>10}")
print("-"*70)

for i, pair in enumerate(viable_pairs[-20:]):
    q1, q2 = pair['qubits']
    rank = len(viable_pairs) - 20 + i + 1
    margin = pair['estimated_L3'] - bound
    print(f"{rank:>4} ({q1:3d},{q2:3d})     {pair['estimated_L3']:>10.4f} {margin:>+10.4f} {pair['avg_error']:>10.4f}")

# Save analysis
output = {
    'phi_deg': phi_deg,
    'L3_theory': L3_theory,
    'L3_bound': bound,
    'total_pairs_tested': len(results),
    'viable_pairs_count': len(viable_pairs),
    'unique_qubits_count': len(unique_qubits),
    'unique_qubits': sorted(list(unique_qubits)),
    'viable_pairs': viable_pairs
}

with open('viable_pairs_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Total pairs tested:        {len(results)}")
print(f"Viable pairs (L₃ > bound): {len(viable_pairs)} ({100*len(viable_pairs)/len(results):.1f}%)")
print(f"Unique qubits involved:    {len(unique_qubits)} qubits")
print(f"\nSaved detailed analysis to viable_pairs_analysis.json")

# Show qubit usage histogram
print(f"\n{'='*70}")
print("QUBIT USAGE IN VIABLE PAIRS")
print(f"{'='*70}")

qubit_count = {}
for pair in viable_pairs:
    for q in pair['qubits']:
        qubit_count[q] = qubit_count.get(q, 0) + 1

# Sort by usage
sorted_qubits = sorted(qubit_count.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 20 most-used qubits in viable pairs:")
print(f"{'Qubit':>6} {'Count':>8} {'% of viable pairs':>20}")
print("-"*40)
for q, count in sorted_qubits[:20]:
    pct = 100 * count / len(viable_pairs)
    print(f"{q:>6} {count:>8} {pct:>19.1f}%")
