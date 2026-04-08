#!/usr/bin/env python3
"""
Draw the mid-circuit measurement Leggett inequality test circuit.
Shows 2 qubits reused 6 times with reset and measurement between each use.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ibm'))
from ibm_leggett_2qb_single import create_leggett_circuit_midcircuit
import matplotlib.pyplot as plt


def draw_midcircuit_circuit():
    """Draw the 2-qubit mid-circuit measurement circuit."""

    # Create mid-circuit circuit for phi = 30 degrees
    phi_rad = np.radians(30)
    qc = create_leggett_circuit_midcircuit(phi_rad)

    print("="*70)
    print("MID-CIRCUIT LEGGETT CIRCUIT (2-QUBIT SEQUENTIAL)")
    print("="*70)
    print(f"\nCircuit statistics:")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Classical registers: {len(qc.cregs)}")
    print(f"  Depth: {qc.depth()}")
    print(f"  Total gates: {sum(qc.count_ops().values())}")
    print(f"\nGate breakdown:")
    for gate, count in sorted(qc.count_ops().items()):
        print(f"    {gate:8s}: {count:3d}")

    # Draw full circuit as text
    print("\n" + "="*70)
    print("2-QUBIT SEQUENTIAL CIRCUIT DIAGRAM (TEXT)")
    print("="*70)
    print("\nNote: This shows 2 qubits reused 6 times")
    print("Each section: Reset → Singlet → Measure → Store in c_i\n")

    print(qc.draw(output='text', fold=120))

    # Try to save as PNG
    print("\n" + "="*70)
    print("SAVING CIRCUIT DIAGRAM")
    print("="*70)

    try:
        # Draw with matplotlib - full circuit
        print("\nGenerating high-resolution PNG of mid-circuit measurement circuit...")
        fig = qc.draw(output='mpl', style='iqp', fold=-1)
        plt.gcf().set_size_inches(20, 8)  # Wide figure for sequential operations
        plt.tight_layout()
        plt.savefig('leggett_circuit_midcircuit_2qb.png', dpi=300, bbox_inches='tight')
        print("✓ Saved mid-circuit circuit to: leggett_circuit_midcircuit_2qb.png")
        print("  (2 qubits, 6 sequential measurements with reset)")
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
This circuit tests the Leggett inequality using 2 qubits that are REUSED 6 times
via mid-circuit measurements (sequential approach for one angle φ).

SEQUENTIAL STRUCTURE (for φ = 30°):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────────────────────────────────────────────────────────┐
│ Iteration 1: Measure C(a₁, b₁)                                    │
│   - Create singlet |Ψ⁻⟩ on qubits 0-1                             │
│   - Rotate q0 to measure a₁ = [1,0,0] (X-axis)                    │
│   - Rotate q1 to measure b₁ = [cos(15°), sin(15°), 0]            │
│   - Measure → store in c0                                         │
├────────────────────────────────────────────────────────────────────┤
│ Reset q0, q1 (back to |00⟩)                                       │
├────────────────────────────────────────────────────────────────────┤
│ Iteration 2: Measure C(a₁, b₁')                                   │
│   - Create singlet |Ψ⁻⟩ on qubits 0-1                             │
│   - Rotate q0 to measure a₁ = [1,0,0] (X-axis)                    │
│   - Rotate q1 to measure b₁' = [cos(15°), -sin(15°), 0]          │
│   - Measure → store in c1                                         │
├────────────────────────────────────────────────────────────────────┤
│ Reset q0, q1 (back to |00⟩)                                       │
├────────────────────────────────────────────────────────────────────┤
│ Iteration 3: Measure C(a₂, b₂)                                    │
│   - Create singlet |Ψ⁻⟩ on qubits 0-1                             │
│   - Rotate q0 to measure a₂ = [0,1,0] (Y-axis)                    │
│   - Rotate q1 to measure b₂ = [0, cos(15°), sin(15°)]            │
│   - Measure → store in c2                                         │
├────────────────────────────────────────────────────────────────────┤
│ Reset q0, q1 (back to |00⟩)                                       │
├────────────────────────────────────────────────────────────────────┤
│ Iteration 4: Measure C(a₂, b₂')                                   │
│   - Create singlet |Ψ⁻⟩ on qubits 0-1                             │
│   - Rotate q0 to measure a₂ = [0,1,0] (Y-axis)                    │
│   - Rotate q1 to measure b₂' = [0, cos(15°), -sin(15°)]          │
│   - Measure → store in c3                                         │
├────────────────────────────────────────────────────────────────────┤
│ Reset q0, q1 (back to |00⟩)                                       │
├────────────────────────────────────────────────────────────────────┤
│ Iteration 5: Measure C(a₃, b₃)                                    │
│   - Create singlet |Ψ⁻⟩ on qubits 0-1                             │
│   - Rotate q0 to measure a₃ = [0,0,1] (Z-axis)                    │
│   - Rotate q1 to measure b₃ = [sin(15°), 0, cos(15°)]            │
│   - Measure → store in c4                                         │
├────────────────────────────────────────────────────────────────────┤
│ Reset q0, q1 (back to |00⟩)                                       │
├────────────────────────────────────────────────────────────────────┤
│ Iteration 6: Measure C(a₃, b₃')                                   │
│   - Create singlet |Ψ⁻⟩ on qubits 0-1                             │
│   - Rotate q0 to measure a₃ = [0,0,1] (Z-axis)                    │
│   - Rotate q1 to measure b₃' = [-sin(15°), 0, cos(15°)]          │
│   - Measure → store in c5                                         │
└────────────────────────────────────────────────────────────────────┘

Result per shot: 6 measurement outcomes in c0-c5, each 2 bits
  Example: "00 11 01 10 01 11" → 6 correlations from one circuit run


COMPARISON: PARALLEL vs SEQUENTIAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parallel (24-qubit):
  ✓ All pairs measured at SAME time → equal decoherence
  ✓ Faster: One job per angle
  ✗ Requires 24 qubits
  ✗ Higher error rate (24-qubit connectivity)

Sequential (2-qubit mid-circuit):
  ✓ Only 2 qubits needed
  ✓ Better connectivity (adjacent qubits)
  ✗ Pairs measured at DIFFERENT times → variable decoherence
  ✗ 6x longer coherence time needed
  ✗ Cumulative errors from resets


BARRIERS IN THIS CIRCUIT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Barrier before each singlet preparation (after reset)
  • Barrier before each measurement (after rotations)
  • Ensures operations complete before next stage
  • measure_all() NOT used - manual measurement into named registers
""")

    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print("""
Text diagram: Printed above
PNG diagram:  leggett_circuit_midcircuit_2qb.png (if pylatexenc installed)

To install PNG support:
  pip install pylatexenc

Then run again:
  python3 draw_midcircuit_leggett.py
""")


if __name__ == "__main__":
    draw_midcircuit_circuit()
