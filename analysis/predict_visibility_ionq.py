import os
#!/usr/bin/env python3
"""
Visibility Prediction for IonQ Forte-1

Based on published characterization data (Dec 2025):
- 1Q Gate Error: 0.020% (median)
- 2Q Gate Error: 0.440% (median)
- SPAM Error: 0.550% (median)
- T1: 188 s
- T2: 0.95 s
- 1Q Gate: 63 µs
- 2Q Gate: 650 µs
- Readout: 250 µs
- Reset: 150 µs

Unlike IBM, IonQ has:
1. All-to-all connectivity (no layout optimization needed)
2. Only aggregate error rates (no per-qubit variation data)
3. Much longer coherence times (seconds vs microseconds)
4. Slower gates but higher fidelity
"""

import numpy as np
import json
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# IonQ Forte-1 characterization data (Dec 2025)
IONQ_FORTE = {
    # Error rates (as probabilities)
    'e_1q': 0.00020,      # 0.020% median 1Q gate error
    'e_2q': 0.00440,      # 0.440% median 2Q gate error
    'e_spam': 0.00550,    # 0.550% SPAM error per qubit

    # Coherence times (seconds)
    'T1': 188.0,
    'T2': 0.95,

    # Gate times (microseconds)
    't_1q': 63,           # µs
    't_2q': 650,          # µs
    't_readout': 250,     # µs
    't_reset': 150,       # µs
}

# Noise model parameters from IonQ API (depolarization rates)
IONQ_NOISE_MODEL = {
    'r_1q': 0.000267,     # 1Q depolarization rate
    'r_2q': 0.004949,     # 2Q depolarization rate
}


def calculate_circuit_time(n_1q, n_2q):
    """
    Calculate gate-only circuit execution time in microseconds.

    IMPORTANT: Only count gate time, NOT measurement time!
    T1/T2 decoherence applies while qubit is in superposition.
    Once measurement starts, the state collapses - measurement errors
    are captured by SPAM/readout error, not T1/T2.
    """
    t = (n_1q * IONQ_FORTE['t_1q'] +
         n_2q * IONQ_FORTE['t_2q'])
    return t


def decoherence_fidelity(t_us, T1_s, T2_s):
    """
    Calculate decoherence fidelity using proper Kraus operator model.

    Args:
        t_us: circuit time in microseconds
        T1_s: T1 time in seconds
        T2_s: T2 time in seconds

    Returns:
        Fidelity factor (0 to 1)
    """
    t = t_us * 1e-6  # Convert to seconds
    T1 = T1_s
    T2 = T2_s

    # Amplitude damping probability
    p1 = 1 - np.exp(-t / T1)

    # Dephasing probability
    if T2 <= T1:
        p_phase = 1 - np.exp(-t / T2)
    else:
        # Pure dephasing time from 1/T2 = 1/(2*T1) + 1/T_phi
        T_phi = 1 / ((1/T2) - (1/(2*T1)))
        p_phase = (1 - p1) * (1 - np.exp(-t / T_phi))

    # Fidelity: (1-p1)(1-p_phase) + p1*0.5 (50% credit when damped)
    F = (1 - p1) * (1 - p_phase) + p1 * 0.5

    return F


def predict_visibility_ionq(n_1q=12, n_2q=1, verbose=True):
    """
    Predict visibility for IonQ Forte-1.

    Model: V = V_SPAM × V_gate × V_decoherence

    Note: No ZZ crosstalk term for trapped ions (different error model).
    IonQ errors are dominated by coherent errors that can partially cancel.

    Args:
        n_1q: Number of 1Q gates per qubit
        n_2q: Number of 2Q gates
        verbose: Print detailed breakdown
    """
    # Circuit time
    t_circuit = calculate_circuit_time(n_1q, n_2q)

    # V_SPAM: Both qubits need correct SPAM
    # SPAM error applies to each qubit
    V_SPAM = (1 - IONQ_FORTE['e_spam']) ** 2

    # V_gate: Gate errors
    # 1Q gates: n_1q gates per qubit × 2 qubits
    # 2Q gates: n_2q gates total
    V_gate_1q = (1 - IONQ_FORTE['e_1q']) ** (n_1q * 2)
    V_gate_2q = (1 - IONQ_FORTE['e_2q']) ** n_2q
    V_gate = V_gate_1q * V_gate_2q

    # V_decoherence: Both qubits decay
    F_qubit = decoherence_fidelity(t_circuit, IONQ_FORTE['T1'], IONQ_FORTE['T2'])
    V_decoherence = F_qubit ** 2

    # Total visibility
    V_total = V_SPAM * V_gate * V_decoherence

    if verbose:
        print("=" * 70)
        print("IONQ FORTE-1 VISIBILITY PREDICTION")
        print("=" * 70)
        print(f"\nCircuit parameters:")
        print(f"  1Q gates per qubit: {n_1q}")
        print(f"  2Q gates: {n_2q}")
        print(f"  Circuit time: {t_circuit:.1f} µs = {t_circuit/1000:.3f} ms")
        print(f"\nCharacterization data:")
        print(f"  1Q gate error: {IONQ_FORTE['e_1q']*100:.3f}%")
        print(f"  2Q gate error: {IONQ_FORTE['e_2q']*100:.3f}%")
        print(f"  SPAM error: {IONQ_FORTE['e_spam']*100:.3f}%")
        print(f"  T1: {IONQ_FORTE['T1']:.0f} s")
        print(f"  T2: {IONQ_FORTE['T2']:.2f} s")
        print(f"\n" + "-" * 70)
        print("VISIBILITY BREAKDOWN")
        print("-" * 70)
        print(f"  V_SPAM:        {V_SPAM*100:.3f}%  (1 - {IONQ_FORTE['e_spam']*100:.3f}%)² = {(1-V_SPAM)*100:.3f}% loss")
        print(f"  V_gate_1q:     {V_gate_1q*100:.3f}%  (1 - {IONQ_FORTE['e_1q']*100:.3f}%)^{n_1q*2} = {(1-V_gate_1q)*100:.3f}% loss")
        print(f"  V_gate_2q:     {V_gate_2q*100:.3f}%  (1 - {IONQ_FORTE['e_2q']*100:.3f}%)^{n_2q} = {(1-V_gate_2q)*100:.3f}% loss")
        print(f"  V_gate (total):{V_gate*100:.3f}%")
        print(f"  V_decoherence: {V_decoherence*100:.3f}%  (t={t_circuit/1000:.3f}ms vs T2={IONQ_FORTE['T2']*1000:.0f}ms)")
        print(f"\n  V_TOTAL:       {V_total*100:.2f}%")

        # Loss breakdown
        loss_spam = (1 - V_SPAM) * 100
        loss_gate = (1 - V_gate) * 100
        loss_decoherence = (1 - V_decoherence) * 100
        total_loss = (1 - V_total) * 100

        print(f"\n" + "-" * 70)
        print("LOSS BREAKDOWN (approximate)")
        print("-" * 70)
        if total_loss > 0:
            print(f"  SPAM:         {loss_spam:.3f}%  ({loss_spam/total_loss*100:.1f}%)")
            print(f"  Gates:        {loss_gate:.3f}%  ({loss_gate/total_loss*100:.1f}%)")
            print(f"  Decoherence:  {loss_decoherence:.3f}%  ({loss_decoherence/total_loss*100:.1f}%)")
            print(f"  ─────────────────────")
            print(f"  TOTAL:        {total_loss:.2f}%")

    return {
        'V_total': V_total,
        'V_SPAM': V_SPAM,
        'V_gate': V_gate,
        'V_gate_1q': V_gate_1q,
        'V_gate_2q': V_gate_2q,
        'V_decoherence': V_decoherence,
        't_circuit_us': t_circuit,
    }


def compare_with_experimental():
    """Compare predicted visibility with experimental IonQ Forte results."""

    # Load experimental data
    try:
        with open(os.path.join(_DATA_DIR, 'leggett_results_ionq_forte_sequential_24qb.json') , 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Experimental data file not found")
        return

    print("\n" + "=" * 70)
    print("COMPARISON WITH EXPERIMENTAL DATA")
    print("=" * 70)

    for entry in data:
        phi = entry['phi_deg']
        # Theory predicts all correlations = cos(phi/2) for singlet
        theory_corr = entry['correlations_theory'][0]

        # Average experimental correlation (across 6 measurements × 2 signs)
        exp_corr_pos = np.mean(np.abs(entry['correlations_pos']))
        exp_corr_neg = np.mean(np.abs(entry['correlations_neg']))
        exp_corr = (exp_corr_pos + exp_corr_neg) / 2

        # Visibility = |experimental| / |theory|
        visibility = exp_corr / abs(theory_corr)

        print(f"  φ = {phi:+5.0f}°: V = {visibility*100:.2f}%  (exp={exp_corr:.4f}, theory={abs(theory_corr):.4f})")

    # Average visibility across all angles
    all_vis = []
    for entry in data:
        theory = abs(entry['correlations_theory'][0])
        for corr_list in [entry['correlations_pos'], entry['correlations_neg']]:
            for c in corr_list:
                all_vis.append(abs(c) / theory)

    avg_vis = np.mean(all_vis)
    std_vis = np.std(all_vis)

    print(f"\n  Average visibility: {avg_vis*100:.2f}% ± {std_vis*100:.2f}%")

    return avg_vis


def main():
    # Leggett circuit has approximately:
    # - 12 1Q gates per qubit (worst case from IBM analysis)
    # - 1 2Q gate (CZ/MS)
    print("\n" + "=" * 70)
    print("LEGGETT CIRCUIT ANALYSIS")
    print("=" * 70)

    result = predict_visibility_ionq(n_1q=12, n_2q=1, verbose=True)

    # Compare with IBM
    print("\n" + "=" * 70)
    print("COMPARISON: IBM PITTSBURGH vs IONQ FORTE")
    print("=" * 70)

    print("\n                         IBM Pittsburgh     IonQ Forte-1")
    print("-" * 60)
    print(f"  2Q gate error:             0.40%              0.44%")
    print(f"  1Q gate error:             0.02%              0.02%")
    print(f"  SPAM error:                0.6-1.5%           0.55%")
    print(f"  T2:                        150-250 µs         950,000 µs")
    print(f"  Circuit time:              2.7 µs             {result['t_circuit_us']:.0f} µs")
    print(f"  t/T2 ratio:                ~1.5%              ~0.0001%")
    print("-" * 60)
    print(f"  Predicted V (best):        97.5%              {result['V_total']*100:.1f}%")
    print(f"  Experimental V:            ~95%               ~98%")
    print("-" * 60)
    print("\nKey insight: IonQ's much longer T2 makes decoherence negligible,")
    print("despite having ~500× slower gates. Gate errors and SPAM dominate.")

    # Compare with experimental
    exp_vis = compare_with_experimental()

    if exp_vis:
        gap = result['V_total'] - exp_vis
        print(f"\n" + "=" * 70)
        print("PREDICTION vs EXPERIMENT")
        print("=" * 70)
        print(f"  Predicted:    {result['V_total']*100:.2f}%")
        print(f"  Experimental: {exp_vis*100:.2f}%")
        print(f"  Gap:          {gap*100:+.2f}%")
        if gap > 0:
            print(f"\n  Unexplained loss: {gap*100:.2f}%")
            print("  Possible sources:")
            print("    - Coherent errors (not captured by depolarizing model)")
            print("    - Crosstalk (IonQ reports <1.1% residual action)")
            print("    - Calibration drift during experiment")
            print("    - Shot noise and statistical variation")


if __name__ == "__main__":
    main()
