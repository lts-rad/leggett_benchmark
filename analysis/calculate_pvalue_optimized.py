#!/usr/bin/env python3
"""
Calculate p-value (statistical significance) for Leggett inequality violation
from IBM Pittsburgh optimized layout hardware results.
"""

import json
import numpy as np
from scipy import stats

# Load hardware results
with open('leggett_results_ibm_ibm_pittsburgh_OPTIMIZED_layout_20251202_114427.json', 'r') as f:
    data = json.load(f)

num_shots = data['num_shots']
bound = data['positive']['bound']

print("="*70)
print("STATISTICAL SIGNIFICANCE ANALYSIS - LEGGETT VIOLATION")
print("="*70)
print(f"\nHardware: IBM Pittsburgh (Optimized Layout)")
print(f"Shots: {num_shots}")
print(f"Leggett Bound: L₃ ≤ {bound:.4f}")
print()

# For each angle
for angle_key, angle_name in [('positive', '+30°'), ('negative', '-30°')]:
    result = data[angle_key]

    L3_measured = result['L3']
    correlations = result['correlations']

    print(f"{'='*70}")
    print(f"Angle: φ = {angle_name}")
    print(f"{'='*70}")

    # Measured L3
    print(f"Measured L₃: {L3_measured:.4f}")
    print(f"Margin above bound: {L3_measured - bound:+.4f}")

    # Estimate standard error of each correlation
    # For correlation C measured from counts, σ_C ≈ 1/√N
    sigma_C = 1.0 / np.sqrt(num_shots)
    print(f"\nStandard error of each correlation: σ_C ≈ {sigma_C:.4f}")

    # L3 = (1/3) * Σ|C_i| for i=1 to 6
    # Assuming independent measurements:
    # σ_L3 = (1/3) * √(Σ σ_C²) = (1/3) * √(6 * σ_C²) = (1/3) * √6 * σ_C
    sigma_L3 = (1.0/3.0) * np.sqrt(6) * sigma_C
    print(f"Standard error of L₃: σ_L3 ≈ {sigma_L3:.4f}")

    # Calculate z-score (standard deviations above bound)
    z_score = (L3_measured - bound) / sigma_L3
    print(f"\nZ-score: {z_score:.2f}σ")

    # One-tailed p-value (testing if L3 > bound)
    # H0: L3 ≤ bound (Leggett/hidden variables)
    # H1: L3 > bound (quantum mechanics)
    p_value = 1 - stats.norm.cdf(z_score)

    print(f"P-value: {p_value:.4f}")

    # Interpretation
    if p_value < 0.001:
        sig = "***HIGHLY SIGNIFICANT*** (p < 0.001)"
    elif p_value < 0.01:
        sig = "**VERY SIGNIFICANT** (p < 0.01)"
    elif p_value < 0.05:
        sig = "*SIGNIFICANT* (p < 0.05)"
    elif p_value < 0.1:
        sig = "marginally significant (p < 0.1)"
    else:
        sig = "NOT SIGNIFICANT (p ≥ 0.1)"

    print(f"Significance: {sig}")

    # Confidence interval (95%)
    ci_lower = L3_measured - 1.96 * sigma_L3
    ci_upper = L3_measured + 1.96 * sigma_L3
    print(f"\n95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

    if ci_lower > bound:
        print(f"✓ Lower bound of CI is ABOVE Leggett bound (strong violation)")
    else:
        print(f"✗ Lower bound of CI overlaps Leggett bound (weak violation)")

    print()

print("="*70)
print("SUMMARY")
print("="*70)
print("\nInterpretation:")
print("- p < 0.05: Statistically significant violation of Leggett bound")
print("- p < 0.01: Very strong evidence for quantum mechanics")
print("- p < 0.001: Overwhelming evidence for quantum mechanics")
print("\nFor a conclusive result, we need p < 0.05 AND")
print("95% CI lower bound > Leggett bound")
print("="*70)
