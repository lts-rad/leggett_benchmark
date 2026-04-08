#!/usr/bin/env python3
"""
Verify claims about singlet state invariance under identical unitaries.
No hand-waving - just simulation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator

def create_singlet():
    """Create singlet state |01⟩ - |10⟩ (unnormalized)"""
    qc = QuantumCircuit(2)
    qc.x(1)           # |01⟩
    qc.h(0)           # (|0⟩ + |1⟩)|1⟩ / sqrt(2)
    qc.cx(0, 1)       # |01⟩ + |10⟩ -> need phase flip
    qc.z(0)           # |01⟩ - |10⟩
    return qc

def test_rotation_invariance():
    """Test if singlet is invariant under identical rotations on both qubits"""
    
    # Create singlet
    qc_singlet = create_singlet()
    sv_singlet = Statevector.from_instruction(qc_singlet)
    
    print("Original singlet state:")
    print(sv_singlet)
    print()
    
    # Test various identical rotations
    test_cases = [
        ("Rx(30°) on both", lambda qc: (qc.rx(np.radians(30), 0), qc.rx(np.radians(30), 1))),
        ("Ry(45°) on both", lambda qc: (qc.ry(np.radians(45), 0), qc.ry(np.radians(45), 1))),
        ("Rz(60°) on both", lambda qc: (qc.rz(np.radians(60), 0), qc.rz(np.radians(60), 1))),
        ("Rx(30°)+Ry(45°)+Rz(60°) on both", lambda qc: (
            qc.rx(np.radians(30), 0), qc.rx(np.radians(30), 1),
            qc.ry(np.radians(45), 0), qc.ry(np.radians(45), 1),
            qc.rz(np.radians(60), 0), qc.rz(np.radians(60), 1)
        )),
        ("Rx(30°) on qubit 0 ONLY", lambda qc: qc.rx(np.radians(30), 0)),
        ("Ry(45°) on qubit 1 ONLY", lambda qc: qc.ry(np.radians(45), 1)),
        ("Rx(30°) on q0, Rx(-30°) on q1", lambda qc: (qc.rx(np.radians(30), 0), qc.rx(np.radians(-30), 1))),
    ]
    
    print("Testing rotation invariance:")
    print("=" * 70)
    print(f"{'Rotation':<40} {'Fidelity':>12} {'Invariant?':>12}")
    print("-" * 70)
    
    for name, apply_rotation in test_cases:
        qc = create_singlet()
        apply_rotation(qc)
        sv_after = Statevector.from_instruction(qc)
        
        # Fidelity = 1 means same state (up to global phase)
        fid = state_fidelity(sv_singlet, sv_after)
        invariant = "YES" if fid > 0.9999 else "NO"
        
        print(f"{name:<40} {fid:>12.6f} {invariant:>12}")
    
    print()
    print("=" * 70)
    print("Fidelity = 1.0 means state unchanged (up to global phase)")
    print("=" * 70)

def test_correlation_effect():
    """Test effect on actual correlations measured at angle phi"""
    from qiskit_aer import AerSimulator
    
    phi_deg = 30
    phi_rad = np.radians(phi_deg)
    shots = 100000
    
    def measure_correlation(qc_base, drift_fn=None):
        """Measure correlation at angle phi"""
        qc = qc_base.copy()
        
        # Apply drift if specified
        if drift_fn:
            drift_fn(qc)
        
        # Measure in rotated basis (angle phi from Z)
        qc.ry(-phi_rad, 0)  # Alice
        qc.ry(-phi_rad, 1)  # Bob
        qc.measure_all()
        
        sim = AerSimulator()
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()
        
        # Correlation: P(same) - P(different)
        same = counts.get('00', 0) + counts.get('11', 0)
        diff = counts.get('01', 0) + counts.get('10', 0)
        return (same - diff) / shots
    
    qc_singlet = create_singlet()
    
    # QM theory for singlet at angle phi: -cos(phi)
    theory = -np.cos(phi_rad)
    
    print(f"\nCorrelation measurements at φ = {phi_deg}°:")
    print("=" * 70)
    print(f"QM theory: C = -cos({phi_deg}°) = {theory:.4f}")
    print("-" * 70)
    
    drift_cases = [
        ("No drift", None),
        ("Rz(30°) on both", lambda qc: (qc.rz(np.radians(30), 0), qc.rz(np.radians(30), 1))),
        ("Rx(30°) on both", lambda qc: (qc.rx(np.radians(30), 0), qc.rx(np.radians(30), 1))),
        ("Ry(30°) on both", lambda qc: (qc.ry(np.radians(30), 0), qc.ry(np.radians(30), 1))),
        ("Rz(30°) on q0 only", lambda qc: qc.rz(np.radians(30), 0)),
        ("Rx(30°) on q0 only", lambda qc: qc.rx(np.radians(30), 0)),
        ("Ry(30°) on q0 only", lambda qc: qc.ry(np.radians(30), 0)),
    ]
    
    print(f"{'Drift':<30} {'Correlation':>12} {'vs Theory':>12}")
    print("-" * 70)
    
    for name, drift_fn in drift_cases:
        corr = measure_correlation(qc_singlet, drift_fn)
        diff = corr - theory
        print(f"{name:<30} {corr:>12.4f} {diff:>+12.4f}")

if __name__ == "__main__":
    test_rotation_invariance()
    test_correlation_effect()
