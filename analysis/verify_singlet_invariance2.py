#!/usr/bin/env python3
"""
Verify singlet invariance with proper Leggett-style measurement.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator

def create_singlet():
    """Create singlet state |01⟩ - |10⟩"""
    qc = QuantumCircuit(2)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)
    return qc

def measure_correlation(qc_base, alice_angle, bob_angle, drift_fn=None, shots=100000):
    """Measure correlation with Alice at angle a, Bob at angle b"""
    qc = qc_base.copy()
    
    if drift_fn:
        drift_fn(qc)
    
    # Alice measures at her angle, Bob at his
    qc.ry(-alice_angle, 0)
    qc.ry(-bob_angle, 1)
    qc.measure_all()
    
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    
    same = counts.get('00', 0) + counts.get('11', 0)
    diff = counts.get('01', 0) + counts.get('10', 0)
    return (same - diff) / shots

def test_leggett_correlations():
    """Test correlations at different angles with and without drift"""
    
    qc_singlet = create_singlet()
    
    # Leggett test uses Alice at 0, Bob at phi
    phi_deg = 30
    phi_rad = np.radians(phi_deg)
    
    alice_angle = 0
    bob_angle = phi_rad
    
    # QM prediction: -cos(bob - alice) = -cos(phi)
    theory = -np.cos(bob_angle - alice_angle)
    
    print(f"Leggett-style correlation: Alice at 0°, Bob at {phi_deg}°")
    print(f"QM theory: C = -cos({phi_deg}°) = {theory:.4f}")
    print("=" * 70)
    
    drift_cases = [
        ("No drift", None),
        ("Rz(30°) on both", lambda qc: (qc.rz(np.radians(30), 0), qc.rz(np.radians(30), 1))),
        ("Rx(30°) on both", lambda qc: (qc.rx(np.radians(30), 0), qc.rx(np.radians(30), 1))),
        ("Ry(30°) on both", lambda qc: (qc.ry(np.radians(30), 0), qc.ry(np.radians(30), 1))),
        ("Rz(30°) on Alice only", lambda qc: qc.rz(np.radians(30), 0)),
        ("Rx(30°) on Alice only", lambda qc: qc.rx(np.radians(30), 0)),
        ("Ry(30°) on Alice only", lambda qc: qc.ry(np.radians(30), 0)),
        ("Rz(30°) on Bob only", lambda qc: qc.rz(np.radians(30), 1)),
        ("Rx(30°) on Bob only", lambda qc: qc.rx(np.radians(30), 1)),
        ("Ry(30°) on Bob only", lambda qc: qc.ry(np.radians(30), 1)),
    ]
    
    print(f"{'Drift':<30} {'Correlation':>12} {'vs Theory':>12} {'Error %':>10}")
    print("-" * 70)
    
    for name, drift_fn in drift_cases:
        corr = measure_correlation(qc_singlet, alice_angle, bob_angle, drift_fn)
        diff = corr - theory
        err_pct = 100 * abs(diff / theory) if theory != 0 else 0
        print(f"{name:<30} {corr:>12.4f} {diff:>+12.4f} {err_pct:>9.1f}%")

if __name__ == "__main__":
    test_leggett_correlations()
