import os
#!/usr/bin/env python3
"""
Draw the Leggett inequality test circuit structure.
Creates a visual diagram showing the pattern of the 12-singlet circuit.
"""

import numpy as np
from qiskit import QuantumCircuit
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from leggett import create_leggett_circuit_for_angle
import matplotlib.pyplot as plt

def draw_circuit_pattern():
    """Draw a 4-qubit sample showing the circuit pattern."""

    # Create the full symmetric circuit for phi = 30 degrees
    phi_rad = np.radians(30)
    qc_full = create_leggett_circuit_for_angle(phi_rad)

    print("="*70)
    print("LEGGETT CIRCUIT STRUCTURE")
    print("="*70)
    print(f"\nFull circuit:")
    print(f"  Qubits: {qc_full.num_qubits}")
    print(f"  Depth: {qc_full.depth()}")
    print(f"  Gates: {qc_full.count_ops()}")

    # Create a smaller circuit showing just the pattern (first 2 pairs)
    print("\nCreating 4-qubit diagram showing the repeating pattern...")

    qc_small = QuantumCircuit(4)

    # Singlet pair 0 (qubits 0-1) measuring (a1, b1)
    # a1 = [1,0,0] (X-axis), b1 = [cos(15°), sin(15°), 0]
    qc_small.x(1)
    qc_small.h(0)
    qc_small.cx(0, 1)
    qc_small.z(1)

    # Alice measurement (X-axis): theta=90°, phi=0°
    # Symmetric compilation: RZ(π/2) RX(-90°) RZ(-π/2)
    qc_small.rz(np.pi/2, 0)
    qc_small.rx(-np.pi/2, 0)
    qc_small.rz(-np.pi/2, 0)

    # Bob measurement (15°): theta=90°, phi=15°
    # RZ(-15°) then RZ(π/2) RX(-90°) RZ(-π/2)
    qc_small.rz(-np.radians(15), 1)
    qc_small.rz(np.pi/2, 1)
    qc_small.rx(-np.pi/2, 1)
    qc_small.rz(-np.pi/2, 1)

    qc_small.barrier()

    # Singlet pair 1 (qubits 2-3) measuring (a1, b1')
    # a1 = [1,0,0] (X-axis), b1' = [cos(15°), -sin(15°), 0]
    qc_small.x(3)
    qc_small.h(2)
    qc_small.cx(2, 3)
    qc_small.z(3)

    # Alice measurement (X-axis)
    qc_small.rz(np.pi/2, 2)
    qc_small.rx(-np.pi/2, 2)
    qc_small.rz(-np.pi/2, 2)

    # Bob measurement (-15° in XY plane)
    qc_small.rz(np.radians(15), 3)  # Note: positive for b1'
    qc_small.rz(np.pi/2, 3)
    qc_small.rx(-np.pi/2, 3)
    qc_small.rz(-np.pi/2, 3)

    qc_small.measure_all()

    # Draw with text output (works without pylatexenc)
    print("\nCircuit diagram (text):")
    print(qc_small.draw(output='text', fold=120))

    # Try to draw with matplotlib
    try:
        fig = qc_small.draw(output='mpl', style='iqp', fold=-1)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'leggett_circuit_pattern_4qb.png'), dpi=300, bbox_inches='tight')
        print("\n✓ Saved circuit diagram to: leggett_circuit_pattern_4qb.png")
        plt.close()
    except Exception as e:
        print(f"\nNote: Could not create matplotlib diagram: {e}")
        print("You can install pylatexenc with: pip install pylatexenc")

    # Print structure explanation
    print("\n" + "="*70)
    print("FULL 24-QUBIT CIRCUIT STRUCTURE")
    print("="*70)

    print("""
The circuit contains 12 independent singlet pairs (24 qubits total):

Pairs 0-5 (qubits 0-11): Test angle +φ = +30°
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pair 0 (q0,q1):   Alice→a₁ (X-axis), Bob→b₁ (15°)
  Pair 1 (q2,q3):   Alice→a₁ (X-axis), Bob→b₁' (-15°)
  Pair 2 (q4,q5):   Alice→a₂ (Y-axis), Bob→b₂
  Pair 3 (q6,q7):   Alice→a₂ (Y-axis), Bob→b₂'
  Pair 4 (q8,q9):   Alice→a₃ (Z-axis), Bob→b₃
  Pair 5 (q10,q11): Alice→a₃ (Z-axis), Bob→b₃'

Pairs 6-11 (qubits 12-23): Test angle -φ = -30°
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pair 6 (q12,q13): Alice→a₁ (X-axis), Bob→b₁ (-15°)
  Pair 7 (q14,q15): Alice→a₁ (X-axis), Bob→b₁' (15°)
  Pair 8 (q16,q17): Alice→a₂ (Y-axis), Bob→b₂
  Pair 9 (q18,q19): Alice→a₂ (Y-axis), Bob→b₂'
  Pair 10 (q20,q21): Alice→a₃ (Z-axis), Bob→b₃
  Pair 11 (q22,q23): Alice→a₃ (Z-axis), Bob→b₃'

Each singlet pair follows the SAME pattern:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. X(Bob)         - Initialize Bob to |1⟩
  2. H(Alice)       - Put Alice in superposition
  3. CX(Alice→Bob)  - Entangle
  4. Z(Bob)         - Create singlet |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2

  5. SYMMETRIC MEASUREMENT on Alice:
     RZ(−φ_alice) RZ(π/2) RX(−θ_alice) RZ(−π/2)

  6. SYMMETRIC MEASUREMENT on Bob:
     RZ(−φ_bob) RZ(π/2) RX(−θ_bob) RZ(−π/2)

  7. Measure both qubits in computational basis
""")

    print("\nGate breakdown for full 24-qubit circuit:")
    print("-"*70)
    total_gates = sum(qc_full.count_ops().values())
    print(f"  Total gates: {total_gates}")
    for gate, count in sorted(qc_full.count_ops().items()):
        print(f"    {gate:8s}: {count:3d}")

    print("\n" + "="*70)
    print("KEY FEATURES OF SYMMETRIC COMPILATION")
    print("="*70)
    print("""
1. RY gates compiled as: RZ(π/2) RX(θ) RZ(−π/2)
   - Sandwiches physical RX with virtual RZ gates
   - Ensures balanced Bloch sphere trajectories
   - Prevents preferential paths through |0⟩ vs |1⟩

2. RZ gates kept as-is (virtual, frame-based)
   - Zero duration on IBM hardware
   - Just updates phase reference for next gate

3. Based on arXiv:2407.14782v1
   - Avoids trajectory imbalance in open systems
   - Should preserve quantum correlations
   - Critical for Bell/Leggett tests on superconducting qubits
""")

if __name__ == "__main__":
    draw_circuit_pattern()
