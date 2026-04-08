#!/usr/bin/env python3
"""
Test singlet state preparation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


print("="*70)
print("TESTING SINGLET STATE PREPARATION")
print("="*70)

# Test the preparation we're using
qc = QuantumCircuit(2)
qc.x(1)  # Start with |01⟩
qc.h(0)
qc.cx(0, 1)
qc.z(1)

# Get statevector
state = Statevector(qc)
print(f"\nStatevector:")
print(f"  |00⟩: {state.data[0]}")
print(f"  |01⟩: {state.data[1]}")
print(f"  |10⟩: {state.data[2]}")
print(f"  |11⟩: {state.data[3]}")

print(f"\nExpected singlet |ψ⁻⟩ = (|01⟩ - |10⟩)/√2:")
print(f"  |00⟩: 0.000")
print(f"  |01⟩: +0.707")
print(f"  |10⟩: -0.707")
print(f"  |11⟩: 0.000")

# Measure in Z-Z basis
qc_measure = qc.copy()
qc_measure.measure_all()

simulator = AerSimulator()
result = simulator.run(qc_measure, shots=10000).result()
counts = result.get_counts()

print(f"\nMeasurement in Z-basis (10000 shots):")
for bitstring in ['00', '01', '10', '11']:
    count = counts.get(bitstring, 0)
    print(f"  {bitstring}: {count:5d} ({count/100:.1f}%)")

# Calculate Z-Z correlation with BOTH possible conventions
print(f"\n{'='*70}")
print("Z-Z CORRELATION WITH DIFFERENT CONVENTIONS")
print(f"{'='*70}")

# Convention 1: same=+1, diff=-1
corr1 = 0.0
for bitstring, count in counts.items():
    alice_bit = int(bitstring[1])
    bob_bit = int(bitstring[0])
    if alice_bit == bob_bit:
        corr1 += count
    else:
        corr1 -= count
corr1 /= 10000

# Convention 2: same=-1, diff=+1 (what we're using)
corr2 = 0.0
for bitstring, count in counts.items():
    alice_bit = int(bitstring[1])
    bob_bit = int(bitstring[0])
    if alice_bit == bob_bit:
        corr2 -= count
    else:
        corr2 += count
corr2 /= 10000

print(f"\nConvention 1 (same=+1, diff=-1): {corr1:+.3f}")
print(f"Convention 2 (same=-1, diff=+1): {corr2:+.3f}")

print(f"\nFor singlet, Z-Z correlation should be:")
print(f"  C(Z,Z) = -1.000 (perfectly anti-correlated)")

if abs(corr1 - (-1.0)) < 0.05:
    print(f"\n✓ Convention 1 gives correct result!")
    print(f"  We should use: same bits → +1, different bits → -1")
elif abs(corr2 - (-1.0)) < 0.05:
    print(f"\n✓ Convention 2 gives correct result!")
    print(f"  We should use: same bits → -1, different bits → +1")
