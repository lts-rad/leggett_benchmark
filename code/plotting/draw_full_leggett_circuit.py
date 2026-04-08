#!/usr/bin/env python3
"""
Draw the FULL 24-qubit Leggett inequality test circuit.
Shows all 12 singlet pairs in parallel.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from leggett import create_leggett_circuit_for_angle

def draw_full_circuit():
    """Draw the complete 24-qubit circuit."""

    # Create the full symmetric circuit for phi = 30 degrees
    phi_rad = np.radians(30)
    qc = create_leggett_circuit_for_angle(phi_rad)

    print("="*70)
    print("FULL 24-QUBIT LEGGETT CIRCUIT (SYMMETRIC COMPILATION)")
    print("="*70)
    print(f"\nCircuit statistics:")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Depth: {qc.depth()}")
    print(f"  Total gates: {sum(qc.count_ops().values())}")
    print(f"\nGate breakdown:")
    for gate, count in sorted(qc.count_ops().items()):
        print(f"    {gate:8s}: {count:3d}")

    # Draw full circuit as text
    print("\n" + "="*70)
    print("FULL 24-QUBIT CIRCUIT DIAGRAM (TEXT)")
    print("="*70)
    print("\nNote: This shows all 12 singlet pairs in parallel")
    print("Pairs 0-5 (qubits 0-11) test +φ, Pairs 6-11 (qubits 12-23) test -φ\n")

    print(qc.draw(output='text', fold=150))

    # Try to save as PNG
    print("\n" + "="*70)
    print("SAVING CIRCUIT DIAGRAM")
    print("="*70)

    try:
        import matplotlib.pyplot as plt

        # Draw with matplotlib - full circuit
        print("\nGenerating high-resolution PNG of full 24-qubit circuit...")
        fig = qc.draw(output='mpl', style='iqp', fold=-1)
        plt.gcf().set_size_inches(24, 16)  # Large figure for 24 qubits
        plt.tight_layout()
        plt.savefig('leggett_circuit_full_24qb.png', dpi=300, bbox_inches='tight')
        print("✓ Saved full circuit to: leggett_circuit_full_24qb.png")
        print("  (24 qubits, all 12 singlet pairs shown)")
        plt.close()

    except ImportError as e:
        print(f"\nNote: Could not create PNG diagram - missing library")
        print(f"Error: {e}")
        print("\nInstall with: pip install pylatexenc")
    except Exception as e:
        print(f"\nNote: Could not create PNG diagram")
        print(f"Error: {e}")

    # Print detailed structure
    print("\n" + "="*70)
    print("CIRCUIT STRUCTURE EXPLANATION")
    print("="*70)

    print("""
This circuit tests the Leggett inequality using 12 independent singlet pairs
running IN PARALLEL (all 24 qubits measured simultaneously in one shot).

PARALLEL STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All 12 pairs are prepared and measured at the SAME TIME:

┌─────────────────────────────────────────────────────────────────┐
│ Pair 0 (q0,q1):   |Ψ⁻⟩ → Alice(a₁) ⊗ Bob(b₁)     → Measure    │
│ Pair 1 (q2,q3):   |Ψ⁻⟩ → Alice(a₁) ⊗ Bob(b₁')    → Measure    │
│ Pair 2 (q4,q5):   |Ψ⁻⟩ → Alice(a₂) ⊗ Bob(b₂)     → Measure    │
│ Pair 3 (q6,q7):   |Ψ⁻⟩ → Alice(a₂) ⊗ Bob(b₂')    → Measure    │
│ Pair 4 (q8,q9):   |Ψ⁻⟩ → Alice(a₃) ⊗ Bob(b₃)     → Measure    │
│ Pair 5 (q10,q11): |Ψ⁻⟩ → Alice(a₃) ⊗ Bob(b₃')    → Measure    │
│                                                                 │
│ Pair 6 (q12,q13): |Ψ⁻⟩ → Alice(a₁) ⊗ Bob(b₁)     → Measure    │
│ Pair 7 (q14,q15): |Ψ⁻⟩ → Alice(a₁) ⊗ Bob(b₁')    → Measure    │
│ Pair 8 (q16,q17): |Ψ⁻⟩ → Alice(a₂) ⊗ Bob(b₂)     → Measure    │
│ Pair 9 (q18,q19): |Ψ⁻⟩ → Alice(a₂) ⊗ Bob(b₂')    → Measure    │
│ Pair 10 (q20,q21): |Ψ⁻⟩ → Alice(a₃) ⊗ Bob(b₃)    → Measure    │
│ Pair 11 (q22,q23): |Ψ⁻⟩ → Alice(a₃) ⊗ Bob(b₃')   → Measure    │
└─────────────────────────────────────────────────────────────────┘

↓ All measured simultaneously ↓

Result: 24-bit string, e.g., "101001011010101001010110"
        ├──────────────────┘ └──────────────────┤
        Pairs 6-11 (−φ)      Pairs 0-5 (+φ)


MEASUREMENT DIRECTIONS (for φ = 30°):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alice always measures along Cartesian axes:
  a₁ = [1, 0, 0]  (X-axis)
  a₂ = [0, 1, 0]  (Y-axis)
  a₃ = [0, 0, 1]  (Z-axis)

Bob measures at angles ±φ/2 = ±15° from Alice's axes:
  For +φ (pairs 0-5):
    b₁  = [cos(15°), sin(15°), 0]     ≈ [0.966, 0.259, 0]
    b₁' = [cos(15°), -sin(15°), 0]    ≈ [0.966, -0.259, 0]
    b₂  = [0, cos(15°), sin(15°)]     ≈ [0, 0.966, 0.259]
    b₂' = [0, cos(15°), -sin(15°)]    ≈ [0, 0.966, -0.259]
    b₃  = [sin(15°), 0, cos(15°)]     ≈ [0.259, 0, 0.966]
    b₃' = [-sin(15°), 0, cos(15°)]    ≈ [-0.259, 0, 0.966]

  For −φ (pairs 6-11): Same but with φ → −φ


SYMMETRIC COMPILATION (arXiv:2407.14782v1):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each measurement uses SYMMETRIC gate compilation:

  Standard (asymmetric):  RZ(−φ) RY(−θ)
  Symmetric (this work):  RZ(−φ) RZ(π/2) RX(−θ) RZ(−π/2)
                                 └──────┬──────┘
                                   Symmetric RY

This ensures balanced Bloch sphere trajectories during measurement,
preventing correlation suppression in open quantum systems.


ADVANTAGE OF 24-QUBIT PARALLEL DESIGN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All pairs decohere for SAME time (no sequential measurement bias)
✓ Single job tests BOTH +φ and −φ (efficient use of quantum resources)
✓ 6 correlations per angle extracted from one 24-bit measurement
✓ Total data: 12 correlations (6 for +φ, 6 for −φ) per circuit run
""")

    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print("""
Text diagram: Printed above
PNG diagram:  leggett_circuit_full_24qb.png (if pylatexenc installed)

To install PNG support:
  pip install pylatexenc

Then run again:
  python3 draw_full_leggett_circuit.py
""")

if __name__ == "__main__":
    draw_full_circuit()
