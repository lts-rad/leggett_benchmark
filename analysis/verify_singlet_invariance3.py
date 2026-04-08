#!/usr/bin/env python3
"""
Test how drift effect depends on measurement angle.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def create_singlet():
    qc = QuantumCircuit(2)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)
    return qc

def measure_correlation(qc_base, alice_angle, bob_angle, drift_fn=None, shots=100000):
    qc = qc_base.copy()
    if drift_fn:
        drift_fn(qc)
    qc.ry(-alice_angle, 0)
    qc.ry(-bob_angle, 1)
    qc.measure_all()
    
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    
    same = counts.get('00', 0) + counts.get('11', 0)
    diff = counts.get('01', 0) + counts.get('10', 0)
    return (same - diff) / shots

qc_singlet = create_singlet()

print("How drift effect depends on measurement angle")
print("=" * 80)

# Test Rz drift at different measurement angles
print("\n30° Rz drift on Alice only:")
print(f"{'Alice angle':<15} {'Bob angle':<15} {'Theory':>10} {'Measured':>10} {'Error %':>10}")
print("-" * 60)

for alice_deg in [0, 30, 45, 60, 90]:
    bob_deg = alice_deg + 30  # Always 30° apart
    alice_rad = np.radians(alice_deg)
    bob_rad = np.radians(bob_deg)
    
    theory = -np.cos(bob_rad - alice_rad)
    no_drift = measure_correlation(qc_singlet, alice_rad, bob_rad, None)
    with_drift = measure_correlation(qc_singlet, alice_rad, bob_rad, 
                                     lambda qc: qc.rz(np.radians(30), 0))
    
    err_pct = 100 * abs(with_drift - no_drift) / abs(no_drift) if no_drift != 0 else 0
    print(f"{alice_deg}°{'':<12} {bob_deg}°{'':<12} {theory:>10.4f} {with_drift:>10.4f} {err_pct:>9.1f}%")

print("\n30° Rx drift on Alice only:")
print(f"{'Alice angle':<15} {'Bob angle':<15} {'Theory':>10} {'Measured':>10} {'Error %':>10}")
print("-" * 60)

for alice_deg in [0, 30, 45, 60, 90]:
    bob_deg = alice_deg + 30
    alice_rad = np.radians(alice_deg)
    bob_rad = np.radians(bob_deg)
    
    theory = -np.cos(bob_rad - alice_rad)
    no_drift = measure_correlation(qc_singlet, alice_rad, bob_rad, None)
    with_drift = measure_correlation(qc_singlet, alice_rad, bob_rad,
                                     lambda qc: qc.rx(np.radians(30), 0))
    
    err_pct = 100 * abs(with_drift - no_drift) / abs(no_drift) if no_drift != 0 else 0
    print(f"{alice_deg}°{'':<12} {bob_deg}°{'':<12} {theory:>10.4f} {with_drift:>10.4f} {err_pct:>9.1f}%")

print("\n30° Ry drift on Alice only:")
print(f"{'Alice angle':<15} {'Bob angle':<15} {'Theory':>10} {'Measured':>10} {'Error %':>10}")
print("-" * 60)

for alice_deg in [0, 30, 45, 60, 90]:
    bob_deg = alice_deg + 30
    alice_rad = np.radians(alice_deg)
    bob_rad = np.radians(bob_deg)
    
    theory = -np.cos(bob_rad - alice_rad)
    no_drift = measure_correlation(qc_singlet, alice_rad, bob_rad, None)
    with_drift = measure_correlation(qc_singlet, alice_rad, bob_rad,
                                     lambda qc: qc.ry(np.radians(30), 0))
    
    err_pct = 100 * abs(with_drift - no_drift) / abs(no_drift) if no_drift != 0 else 0
    print(f"{alice_deg}°{'':<12} {bob_deg}°{'':<12} {theory:>10.4f} {with_drift:>10.4f} {err_pct:>9.1f}%")
