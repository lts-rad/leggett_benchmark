#!/usr/bin/env python3
"""
Predict visibility for Leggett measurements on IBM Pittsburgh.

1. Download backend properties (error rates, coupling map)
2. Transpile each measurement basis to find worst-case gate count
3. For each directly-coupled qubit pair, predict visibility based on:
   - Readout error / SPAM
   - Two-qubit gate error
   - Single-qubit gate errors
   - T1/T2 decoherence during circuit execution
"""

import numpy as np
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService


def create_singlet_with_measurement(basis_a, basis_b):
    """
    Create a circuit that prepares singlet and measures in specified bases.

    Args:
        basis_a: Measurement basis for Alice ('X', 'Y', 'Z' or Bloch vector)
        basis_b: Measurement basis for Bob ('X', 'Y', 'Z' or Bloch vector)

    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(2, 2)

    # Create singlet |Psi^-⟩ = (|01⟩ - |10⟩)/sqrt(2)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)

    # Apply measurement rotations for Alice (qubit 0)
    if isinstance(basis_a, str):
        if basis_a == 'X':
            qc.h(0)
        elif basis_a == 'Y':
            qc.sdg(0)
            qc.h(0)
        # Z: no rotation
    else:
        # Bloch vector [x, y, z] - convert to rotation
        theta, phi = bloch_to_angles(basis_a)
        qc.rz(-phi, 0)
        qc.ry(-theta, 0)

    # Apply measurement rotations for Bob (qubit 1)
    if isinstance(basis_b, str):
        if basis_b == 'X':
            qc.h(1)
        elif basis_b == 'Y':
            qc.sdg(1)
            qc.h(1)
        # Z: no rotation
    else:
        theta, phi = bloch_to_angles(basis_b)
        qc.rz(-phi, 1)
        qc.ry(-theta, 1)

    qc.measure([0, 1], [0, 1])
    return qc


def bloch_to_angles(vec):
    """Convert Bloch sphere vector to spherical angles (theta, phi)."""
    x, y, z = vec
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    return theta, phi


def get_leggett_measurement_bases(phi_rad):
    """
    Get the 6 measurement basis pairs for Leggett test at angle phi.

    Returns list of (name, alice_basis, bob_basis) tuples.
    """
    # Alice's measurement directions (fixed)
    a1 = np.array([1, 0, 0])  # X
    a2 = np.array([0, 1, 0])  # Y
    a3 = np.array([0, 0, 1])  # Z

    # Bob's measurement directions (depend on phi)
    b1 = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2 = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3 = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    return [
        ('a1_b1', a1, b1),
        ('a1_b1p', a1, b1_prime),
        ('a2_b2', a2, b2),
        ('a2_b2p', a2, b2_prime),
        ('a3_b3', a3, b3),
        ('a3_b3p', a3, b3_prime),
    ]


def analyze_transpiled_circuit(qc_transpiled, backend=None):
    """
    Analyze a transpiled circuit for gate counts and duration.

    Returns dict with gate counts and duration.
    """
    ops = qc_transpiled.count_ops()

    # Common IBM native gates
    n_1q = 0
    n_2q = 0

    for gate, count in ops.items():
        if gate in ['sx', 'x', 'rz', 'id', 'u1', 'u2', 'u3']:
            n_1q += count
        elif gate in ['cx', 'ecr', 'cz']:
            n_2q += count
        elif gate == 'measure':
            pass  # Don't count measurements as gates
        elif gate == 'barrier':
            pass
        else:
            # Unknown gate - assume single qubit
            n_1q += count

    # Get circuit duration if available
    duration = None
    if backend is not None:
        try:
            from qiskit.transpiler import InstructionDurations
            durations = InstructionDurations.from_backend(backend)
            # Schedule the circuit to get duration
            from qiskit import transpile
            qc_scheduled = transpile(qc_transpiled, backend=backend, scheduling_method='asap')
            if qc_scheduled.duration is not None:
                # Duration is in dt units, convert to seconds
                dt = backend.dt  # dt in seconds
                duration = qc_scheduled.duration * dt
        except Exception as e:
            pass

    return {
        'ops': dict(ops),
        'n_1q': n_1q,
        'n_2q': n_2q,
        'depth': qc_transpiled.depth(),
        'duration': duration,  # in seconds, or None if not available
    }


def get_backend_error_data(backend):
    """
    Extract error rates from backend.

    Returns:
        qubit_errors: dict mapping qubit -> {readout_error, t1, t2, sx_error, x_error}
        edge_errors: dict mapping (q1, q2) -> {cx_error or ecr_error, zz_coupling}
        coupling_map: list of [q1, q2] edges
    """
    target = backend.target
    qubit_errors = {}
    edge_errors = {}
    zz_data = {}  # ZZ crosstalk data
    num_qubits = backend.num_qubits

    print(f"  Available gates: {list(target.operation_names)}")

    # Extract ZZ crosstalk from backend.properties().general
    try:
        props = backend.properties()
        if props and hasattr(props, 'general'):
            for item in props.general:
                name = item.name
                if name.startswith('zz_'):
                    # Parse qubit pair from name like 'zz_8182' or 'zz_100101'
                    pair_str = name[3:]  # Remove 'zz_' prefix
                    # Try to split into two qubit numbers
                    # Handle cases like '8182' (81,82) and '100101' (100,101)
                    for split_pos in range(1, len(pair_str)):
                        try:
                            q1 = int(pair_str[:split_pos])
                            q2 = int(pair_str[split_pos:])
                            if 0 <= q1 < num_qubits and 0 <= q2 < num_qubits and q1 != q2:
                                zz_hz = item.value * 1e9  # Convert GHz to Hz
                                zz_data[(q1, q2)] = zz_hz
                                zz_data[(q2, q1)] = zz_hz  # Symmetric
                                break
                        except ValueError:
                            continue
            print(f"  Extracted ZZ crosstalk for {len(zz_data)//2} qubit pairs")
    except Exception as e:
        print(f"  Warning: Could not extract ZZ data: {e}")

    # Extract single-qubit properties
    for q in range(num_qubits):
        qubit_errors[q] = {
            'readout_error': 0.01,
            'sx_error': 0.001,
            'x_error': 0.001,
            'rz_error': 0.0,
            't1': 200e-6,  # Default 200 µs in seconds
            't2': 200e-6,  # Default 200 µs in seconds
        }

        # T1 and T2 from qubit_properties
        try:
            qprops = backend.qubit_properties(q)
            if qprops.t1 is not None:
                qubit_errors[q]['t1'] = qprops.t1  # Already in seconds
            if qprops.t2 is not None:
                qubit_errors[q]['t2'] = qprops.t2  # Already in seconds
        except Exception:
            pass

        # Readout error
        try:
            if 'measure' in target.operation_names:
                meas_props = target['measure'].get((q,))
                if meas_props and meas_props.error is not None:
                    qubit_errors[q]['readout_error'] = meas_props.error
        except Exception as e:
            pass

        # SX gate error
        try:
            if 'sx' in target.operation_names:
                sx_props = target['sx'].get((q,))
                if sx_props and sx_props.error is not None:
                    qubit_errors[q]['sx_error'] = sx_props.error
        except Exception:
            pass

        # X gate error
        try:
            if 'x' in target.operation_names:
                x_props = target['x'].get((q,))
                if x_props and x_props.error is not None:
                    qubit_errors[q]['x_error'] = x_props.error
        except Exception:
            pass

    # Get coupling map from backend directly
    coupling_map = []
    try:
        cm = backend.coupling_map
        if cm:
            coupling_map = list(cm.get_edges())
            print(f"  Got {len(coupling_map)} edges from coupling_map")
    except Exception as e:
        print(f"  Warning: Could not get coupling_map: {e}")

    # Determine 2Q gate type
    if 'ecr' in target.operation_names:
        two_q_gate = 'ecr'
    elif 'cx' in target.operation_names:
        two_q_gate = 'cx'
    elif 'cz' in target.operation_names:
        two_q_gate = 'cz'
    else:
        two_q_gate = None
        print(f"  Warning: No known 2Q gate found!")

    print(f"  2Q gate type: {two_q_gate}")

    # Extract 2Q gate errors
    if two_q_gate:
        try:
            gate_ops = target[two_q_gate]
            for qargs in gate_ops:
                if qargs and len(qargs) == 2:
                    q1, q2 = qargs
                    props = gate_ops[qargs]
                    error = props.error if props and props.error is not None else 0.01
                    duration = props.duration if props and props.duration else 0
                    # Get ZZ crosstalk for this pair
                    zz = zz_data.get((q1, q2), 0)
                    edge_errors[(q1, q2)] = {
                        'gate': two_q_gate,
                        'error': error,
                        'duration': duration,
                        'zz_hz': zz,  # ZZ crosstalk in Hz
                    }
                    # Also add to coupling map if not already there
                    if [q1, q2] not in coupling_map:
                        coupling_map.append([q1, q2])
        except Exception as e:
            print(f"  Warning: Error extracting 2Q gate data: {e}")

    print(f"  Total edges: {len(coupling_map)}")
    print(f"  Edge errors extracted: {len(edge_errors)}")

    # Sample a few qubit errors for debugging
    sample_qubits = [0, 1, 2] if num_qubits > 2 else list(range(num_qubits))
    for q in sample_qubits:
        print(f"  Qubit {q}: RO={qubit_errors[q]['readout_error']:.4f}, "
              f"SX={qubit_errors[q]['sx_error']:.6f}")

    return qubit_errors, edge_errors, coupling_map


def predict_visibility(qubit_errors, edge_errors, q1, q2, n_1q, n_2q, circuit_duration=None):
    """
    Predict visibility for a qubit pair based on error rates and decoherence.

    Model:
    V = V_readout * V_gates * V_T2 * V_T1 * V_ZZ

    Where:
    - V_readout = (1 - ro1) * (1 - ro2)
    - V_gates = (1 - 1q_error)^n_1q * (1 - 2q_error)^n_2q
    - V_T2 = exp(-t/T2_q1) * exp(-t/T2_q2)  [dephasing during circuit]
    - V_T1 = exp(-t/T1_q1) * exp(-t/T1_q2)  [amplitude damping]
    - V_ZZ = cos(2π * ZZ_freq * t)  [ZZ crosstalk phase error]

    Args:
        qubit_errors: dict with per-qubit error data including t1, t2
        edge_errors: dict with 2Q gate errors and zz_hz
        q1, q2: qubit indices
        n_1q, n_2q: gate counts
        circuit_duration: total circuit time in seconds (including measurement)
    """
    # Get readout errors
    ro1 = qubit_errors.get(q1, {}).get('readout_error', 0.01)
    ro2 = qubit_errors.get(q2, {}).get('readout_error', 0.01)

    # Get average single-qubit gate error
    sx1 = qubit_errors.get(q1, {}).get('sx_error', 0.001)
    sx2 = qubit_errors.get(q2, {}).get('sx_error', 0.001)
    avg_1q_error = (sx1 + sx2) / 2

    # Get two-qubit gate error and ZZ crosstalk (check both directions)
    edge_data = edge_errors.get((q1, q2)) or edge_errors.get((q2, q1))
    if edge_data:
        twoq_error = edge_data['error']
        zz_hz = edge_data.get('zz_hz', 0)
    else:
        twoq_error = 0.01  # Default if not found
        zz_hz = 0

    # Get T1/T2 times (in seconds)
    t1_q1 = qubit_errors.get(q1, {}).get('t1', 200e-6)
    t1_q2 = qubit_errors.get(q2, {}).get('t1', 200e-6)
    t2_q1 = qubit_errors.get(q1, {}).get('t2', 200e-6)
    t2_q2 = qubit_errors.get(q2, {}).get('t2', 200e-6)

    # Readout contribution: probability both qubits read correctly
    readout_fidelity = (1 - ro1) * (1 - ro2)

    # Gate error contribution
    gate_fidelity = ((1 - avg_1q_error) ** n_1q) * ((1 - twoq_error) ** n_2q)

    # Decoherence contributions (only if circuit duration is known)
    if circuit_duration is not None and circuit_duration > 0:
        t = circuit_duration

        # Proper decoherence model (avoiding double-counting):
        # 1/T2 = 1/(2*T1) + 1/T_φ, so T2 already includes T1 contribution to coherence
        #
        # We use T1 for amplitude damping + pure dephasing T_φ:
        #   T_φ = 1 / (1/T2 - 1/(2*T1))
        #
        # V_T1 = exp(-t/T1) for amplitude damping (|1⟩ → |0⟩)
        # V_Tφ = exp(-t/T_φ) for pure dephasing

        # T1 amplitude damping
        v_t1_q1 = np.exp(-t / t1_q1) if t1_q1 > 0 else 0
        v_t1_q2 = np.exp(-t / t1_q2) if t1_q2 > 0 else 0
        t1_fidelity = v_t1_q1 * v_t1_q2

        # Pure dephasing T_φ (T2 contribution minus T1 contribution)
        # 1/T_φ = 1/T2 - 1/(2*T1)
        def calc_tphi(t1, t2):
            if t1 <= 0 or t2 <= 0:
                return 0
            rate_phi = (1/t2) - (1/(2*t1))
            if rate_phi <= 0:
                # T2 is limited by T1 only, no pure dephasing
                return float('inf')
            return 1 / rate_phi

        tphi_q1 = calc_tphi(t1_q1, t2_q1)
        tphi_q2 = calc_tphi(t1_q2, t2_q2)

        v_tphi_q1 = np.exp(-t / tphi_q1) if tphi_q1 > 0 and tphi_q1 != float('inf') else 1.0
        v_tphi_q2 = np.exp(-t / tphi_q2) if tphi_q2 > 0 and tphi_q2 != float('inf') else 1.0
        tphi_fidelity = v_tphi_q1 * v_tphi_q2

        # Combined T1 + T_φ gives same result as T2 (for verification)
        t2_fidelity = t1_fidelity * tphi_fidelity  # This should equal exp(-t/T2) product

        # ZZ crosstalk: causes phase accumulation
        # Phase error = 2π * ZZ_freq * t
        # For Bell state correlations, this causes visibility loss ≈ cos(phase_error)
        if zz_hz != 0:
            phase_error = 2 * np.pi * abs(zz_hz) * t
            zz_fidelity = np.cos(phase_error)
            # Clamp to [0, 1] in case of large phase errors
            zz_fidelity = max(0, zz_fidelity)
        else:
            zz_fidelity = 1.0
    else:
        t2_fidelity = 1.0
        t1_fidelity = 1.0
        tphi_fidelity = 1.0
        zz_fidelity = 1.0

    # Combined visibility estimate
    # Using T1 (amplitude damping) + T_φ (pure dephasing) to avoid double-counting
    # V_decoherence = V_T1 × V_Tφ = V_T2 (mathematically equivalent)
    visibility = readout_fidelity * gate_fidelity * t2_fidelity * zz_fidelity

    # Calculate T_φ for display
    def calc_tphi_display(t1, t2):
        if t1 <= 0 or t2 <= 0:
            return 0
        rate_phi = (1/t2) - (1/(2*t1))
        if rate_phi <= 0:
            return float('inf')
        return 1 / rate_phi

    tphi_q1_val = calc_tphi_display(t1_q1, t2_q1) if circuit_duration else 0
    tphi_q2_val = calc_tphi_display(t1_q2, t2_q2) if circuit_duration else 0

    return {
        'visibility': visibility,
        'readout_fidelity': readout_fidelity,
        'gate_fidelity': gate_fidelity,
        't2_fidelity': t2_fidelity,
        't1_fidelity': t1_fidelity,
        'tphi_fidelity': tphi_fidelity if circuit_duration else 1.0,
        'zz_fidelity': zz_fidelity,
        'ro_q1': ro1,
        'ro_q2': ro2,
        'avg_1q_error': avg_1q_error,
        'twoq_error': twoq_error,
        'zz_khz': zz_hz / 1000,  # Convert to kHz for display
        't1_q1': t1_q1 * 1e6,  # Convert to µs for display
        't1_q2': t1_q2 * 1e6,
        't2_q1': t2_q1 * 1e6,
        't2_q2': t2_q2 * 1e6,
        'tphi_q1': tphi_q1_val * 1e6 if tphi_q1_val != float('inf') else float('inf'),
        'tphi_q2': tphi_q2_val * 1e6 if tphi_q2_val != float('inf') else float('inf'),
        'n_1q': n_1q,
        'n_2q': n_2q,
        'circuit_duration_us': circuit_duration * 1e6 if circuit_duration else None,
    }


def main():
    import sys

    backend_name = "ibm_pittsburgh"
    phi_deg = 30  # Default test angle

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend_name = sys.argv[i + 1]
        elif arg == '--phi' and i + 1 < len(sys.argv):
            phi_deg = float(sys.argv[i + 1])

    phi_rad = np.radians(phi_deg)

    print("=" * 80)
    print(f"VISIBILITY PREDICTION FOR LEGGETT MEASUREMENTS")
    print("=" * 80)
    print(f"\nBackend: {backend_name}")
    print(f"Test angle: {phi_deg}°")

    # Connect to IBM and get backend
    print(f"\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    print(f"Backend: {backend.name}, {backend.num_qubits} qubits")

    # Get error data
    print(f"\nExtracting error rates...")
    qubit_errors, edge_errors, coupling_map = get_backend_error_data(backend)

    print(f"Found {len(coupling_map)} direct qubit connections")

    # Get measurement bases for Leggett test
    bases = get_leggett_measurement_bases(phi_rad)

    print(f"\n" + "=" * 80)
    print("STEP 1: ANALYZE GATE COUNTS FOR EACH MEASUREMENT BASIS")
    print("=" * 80)

    # Transpile each basis and find worst case
    basis_analysis = []

    for name, basis_a, basis_b in bases:
        qc = create_singlet_with_measurement(basis_a, basis_b)

        # Transpile for the backend (use a representative edge)
        # Just need gate decomposition, not specific mapping
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)

        analysis = analyze_transpiled_circuit(qc_transpiled, backend=backend)
        analysis['name'] = name
        basis_analysis.append(analysis)

        duration_str = f", duration: {analysis['duration']*1e6:.3f} µs" if analysis['duration'] else ""
        print(f"\n  {name}:")
        print(f"    1Q gates: {analysis['n_1q']}, 2Q gates: {analysis['n_2q']}, depth: {analysis['depth']}{duration_str}")
        print(f"    Ops: {analysis['ops']}")

    # Find worst case
    worst_case = max(basis_analysis, key=lambda x: x['n_1q'] + 2*x['n_2q'])
    print(f"\n  WORST CASE: {worst_case['name']}")
    print(f"    1Q gates: {worst_case['n_1q']}, 2Q gates: {worst_case['n_2q']}")

    # Get circuit duration for T1/T2 decoherence
    # IMPORTANT: Only count gate time, NOT measurement time!
    # T1/T2 decoherence applies while qubit is in superposition.
    # Once measurement starts, the state collapses - measurement errors
    # are captured by SPAM/readout error, not T1/T2.

    circuit_duration_full = worst_case.get('duration')  # May include measurement

    # Calculate gate-only duration for decoherence
    # Typical IBM Heron gate times:
    # - 1Q gates (sx, rz): ~30-50ns
    # - 2Q gates (cz): ~68-84ns (Heron is fast!)
    # Note: rz is virtual (0 time), but we count conservatively
    gate_duration = (worst_case['n_1q'] * 40e-9 + worst_case['n_2q'] * 80e-9)

    if circuit_duration_full:
        print(f"    Full circuit duration (from scheduler): {circuit_duration_full*1e6:.3f} µs")
    print(f"    Gate-only duration (for T1/T2): {gate_duration*1e6:.3f} µs")

    # Use gate-only duration for decoherence calculation
    circuit_duration = gate_duration

    # Use worst case for visibility prediction
    n_1q_worst = worst_case['n_1q']
    n_2q_worst = worst_case['n_2q']

    print(f"\n" + "=" * 80)
    print("STEP 2: PREDICT VISIBILITY FOR ALL DIRECT QUBIT PAIRS")
    print("=" * 80)

    # Get unique undirected edges
    edges_set = set()
    for q1, q2 in coupling_map:
        edge = tuple(sorted([q1, q2]))
        edges_set.add(edge)

    edges = sorted(list(edges_set))
    print(f"\nAnalyzing {len(edges)} unique qubit pairs...")

    predictions = []

    for q1, q2 in edges:
        pred = predict_visibility(qubit_errors, edge_errors, q1, q2, n_1q_worst, n_2q_worst,
                                  circuit_duration=circuit_duration)
        pred['q1'] = q1
        pred['q2'] = q2
        predictions.append(pred)

    # Sort by visibility (best first)
    predictions.sort(key=lambda x: x['visibility'], reverse=True)

    if not predictions:
        print("\nERROR: No predictions generated! Check coupling map extraction.")
        return []

    # Print top 20 and bottom 10
    print(f"\n{'='*130}")
    print("TOP 20 QUBIT PAIRS (HIGHEST PREDICTED VISIBILITY)")
    print("="*130)
    print(f"{'Rank':>4} {'Pair':>10} {'Visibility':>10} {'RO':>7} {'Gate':>7} {'T2':>7} {'T1':>7} {'ZZ':>7} {'T2_A':>6} {'T2_B':>6} {'T1_A':>6} {'T1_B':>6}")
    print("-" * 130)

    for i, pred in enumerate(predictions[:20]):
        print(f"{i+1:>4} ({pred['q1']:>3},{pred['q2']:>3}) "
              f"{pred['visibility']:>10.4f} {pred['readout_fidelity']:>7.4f} "
              f"{pred['gate_fidelity']:>7.4f} {pred['t2_fidelity']:>7.4f} "
              f"{pred['t1_fidelity']:>7.4f} {pred['zz_fidelity']:>7.4f} "
              f"{pred['t2_q1']:>6.0f} {pred['t2_q2']:>6.0f} "
              f"{pred['t1_q1']:>6.0f} {pred['t1_q2']:>6.0f}")

    print(f"\n{'='*130}")
    print("BOTTOM 10 QUBIT PAIRS (LOWEST PREDICTED VISIBILITY)")
    print("="*130)
    print(f"{'Rank':>4} {'Pair':>10} {'Visibility':>10} {'RO':>7} {'Gate':>7} {'T2':>7} {'T1':>7} {'ZZ':>7} {'T2_A':>6} {'T2_B':>6} {'T1_A':>6} {'T1_B':>6}")
    print("-" * 130)

    for i, pred in enumerate(predictions[-10:]):
        rank = len(predictions) - 10 + i + 1
        print(f"{rank:>4} ({pred['q1']:>3},{pred['q2']:>3}) "
              f"{pred['visibility']:>10.4f} {pred['readout_fidelity']:>7.4f} "
              f"{pred['gate_fidelity']:>7.4f} {pred['t2_fidelity']:>7.4f} "
              f"{pred['t1_fidelity']:>7.4f} {pred['zz_fidelity']:>7.4f} "
              f"{pred['t2_q1']:>6.0f} {pred['t2_q2']:>6.0f} "
              f"{pred['t1_q1']:>6.0f} {pred['t1_q2']:>6.0f}")

    # Statistics
    visibilities = [p['visibility'] for p in predictions]
    print(f"\n{'='*80}")
    print("VISIBILITY STATISTICS")
    print("="*80)
    print(f"  Best:   {max(visibilities):.4f}")
    print(f"  Worst:  {min(visibilities):.4f}")
    print(f"  Mean:   {np.mean(visibilities):.4f}")
    print(f"  Median: {np.median(visibilities):.4f}")
    print(f"  Std:    {np.std(visibilities):.4f}")

    # Thresholds
    V_ENTANGLED = 1/3
    V_CHSH = 1/np.sqrt(2)
    V_LEGGETT_COMPAT = (1 + 1/np.sqrt(2)) / 2
    V_LEGGETT_VIOLATE = np.sqrt(3) / 2

    n_entangled = sum(1 for v in visibilities if v > V_ENTANGLED)
    n_chsh = sum(1 for v in visibilities if v > V_CHSH)
    n_leggett_violate = sum(1 for v in visibilities if v > V_LEGGETT_VIOLATE)

    print(f"\n  Pairs with V > {V_ENTANGLED:.3f} (entangled):        {n_entangled}/{len(predictions)}")
    print(f"  Pairs with V > {V_CHSH:.3f} (CHSH violation):    {n_chsh}/{len(predictions)}")
    print(f"  Pairs with V > {V_LEGGETT_VIOLATE:.3f} (Leggett violation): {n_leggett_violate}/{len(predictions)}")

    # Save results
    output_file = f"visibility_predictions_{backend_name}.json"
    print(f"\nSaving results to {output_file}...")

    results = {
        'backend': backend_name,
        'phi_deg': phi_deg,
        'worst_case_basis': worst_case['name'],
        'n_1q_worst': n_1q_worst,
        'n_2q_worst': n_2q_worst,
        'circuit_duration_us': circuit_duration * 1e6 if circuit_duration else None,
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions,
        'statistics': {
            'best': max(visibilities),
            'worst': min(visibilities),
            'mean': np.mean(visibilities),
            'median': np.median(visibilities),
            'std': np.std(visibilities),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Done!")

    return predictions


if __name__ == "__main__":
    main()
