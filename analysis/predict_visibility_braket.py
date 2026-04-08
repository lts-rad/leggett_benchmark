#!/usr/bin/env python3
"""
Calculate visibility predictions from Braket device JSON files.
"""

import json
import numpy as np

def analyze_device(json_path, device_name, top_n=12):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract single-qubit properties into dict by qubit id
    qubit_props = {}
    for qubit_id, props in data['oneQubitProperties'].items():
        qid = int(qubit_id)
        t1_val = props.get('T1', {}).get('value')
        t2_val = props.get('T2', {}).get('value')
        qubit_props[qid] = {
            't1': t1_val if t1_val and t1_val > 0 else 50e-6,
            't2': t2_val if t2_val and t2_val > 0 else 20e-6,
            'readout': 0.95,
            'oneq_fid': 0.999,
        }
        for fid in props.get('oneQubitFidelity', []):
            fid_type = fid['fidelityType']['name']
            if fid_type == 'READOUT':
                qubit_props[qid]['readout'] = fid['fidelity']
            elif fid_type in ['RANDOMIZED_BENCHMARKING', 'SIMULTANEOUS_RANDOMIZED_BENCHMARKING']:
                qubit_props[qid]['oneq_fid'] = fid['fidelity']

    # Calculate per-pair visibility predictions
    pair_predictions = []
    n_1q = 12
    n_2q = 1
    circuit_time = 0.6e-6

    if 'twoQubitProperties' in data:
        for edge_id, props in data['twoQubitProperties'].items():
            # Parse edge like "0-1" or "1-2"
            parts = edge_id.replace('-', '_').split('_')
            if len(parts) == 2:
                try:
                    q1, q2 = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
            else:
                continue

            if q1 not in qubit_props or q2 not in qubit_props:
                continue

            # Get 2Q gate fidelity
            twoq_fid = 0.98
            for fid in props.get('twoQubitGateFidelity', []):
                twoq_fid = fid['fidelity']
                break

            # Calculate visibility for this pair
            qp1 = qubit_props[q1]
            qp2 = qubit_props[q2]

            # Skip pairs with bad T2 values
            if qp1['t2'] <= 0 or qp2['t2'] <= 0 or qp1['t2'] < 1e-9 or qp2['t2'] < 1e-9:
                continue

            F_readout = qp1['readout'] * qp2['readout']
            avg_1q_error = (1 - qp1['oneq_fid'] + 1 - qp2['oneq_fid']) / 2
            F_gate = ((1 - avg_1q_error) ** n_1q) * (twoq_fid ** n_2q)
            F_T2 = np.exp(-circuit_time / qp1['t2']) * np.exp(-circuit_time / qp2['t2'])

            V = F_readout * F_gate * F_T2

            pair_predictions.append({
                'q1': q1, 'q2': q2,
                'visibility': V,
                'readout_fidelity': F_readout,
                'gate_fidelity': F_gate,
                't2_fidelity': F_T2,
                'twoq_fid': twoq_fid,
            })

    # Sort by visibility
    pair_predictions.sort(key=lambda x: x['visibility'], reverse=True)

    print(f"\n{'='*60}")
    print(f"{device_name}")
    print(f"{'='*60}")
    print(f"Qubits: {len(qubit_props)}, Edges: {len(pair_predictions)}")

    # Stats for all pairs
    all_vis = [p['visibility'] for p in pair_predictions]
    print(f"\nAll pairs: V_median={np.median(all_vis):.4f}, V_mean={np.mean(all_vis):.4f}")

    # Top N pairs
    top_pairs = pair_predictions[:top_n]
    if top_pairs:
        print(f"\nTop {top_n} pairs:")
        for i, p in enumerate(top_pairs[:6]):
            print(f"  {i+1}. ({p['q1']},{p['q2']}): V={p['visibility']:.4f} "
                  f"(RO={p['readout_fidelity']:.3f}, Gate={p['gate_fidelity']:.3f}, T2={p['t2_fidelity']:.3f})")
        if len(top_pairs) > 6:
            print(f"  ...")

        top_vis = [p['visibility'] for p in top_pairs]
        top_ro = [p['readout_fidelity'] for p in top_pairs]
        top_gate = [p['gate_fidelity'] for p in top_pairs]
        top_t2 = [p['t2_fidelity'] for p in top_pairs]

        print(f"\n--- Top {top_n} Averages ---")
        print(f"F_readout:   {np.mean(top_ro):.4f} ± {np.std(top_ro):.4f}")
        print(f"F_gate:      {np.mean(top_gate):.4f} ± {np.std(top_gate):.4f}")
        print(f"F_T2:        {np.mean(top_t2):.4f} ± {np.std(top_t2):.4f}")
        print(f"V_predicted: {np.mean(top_vis):.4f} ± {np.std(top_vis):.4f}")

        return {
            'device': device_name,
            'F_readout': np.mean(top_ro),
            'F_gate': np.mean(top_gate),
            'F_T2': np.mean(top_t2),
            'V_predicted': np.mean(top_vis),
            'F_readout_std': np.std(top_ro),
            'F_gate_std': np.std(top_gate),
            'F_T2_std': np.std(top_t2),
            'V_std': np.std(top_vis),
        }

    return None


def main():
    rigetti = analyze_device('braket_ankaa_rigetti.json', 'Rigetti Ankaa-3', top_n=12)
    iqm = analyze_device('braket_emerald_iqm.json', 'IQM Emerald', top_n=12)

    print(f"\n{'='*60}")
    print("LATEX TABLE VALUES (Top 12 pairs)")
    print(f"{'='*60}")
    if rigetti and iqm:
        print(f"                  Rigetti    IQM")
        print(f"F_readout:        {rigetti['F_readout']:.3f}      {iqm['F_readout']:.3f}")
        print(f"F_gate:           {rigetti['F_gate']:.3f}      {iqm['F_gate']:.3f}")
        print(f"F_T2:             {rigetti['F_T2']:.3f}      {iqm['F_T2']:.3f}")
        print(f"V_predicted:      {rigetti['V_predicted']:.3f}      {iqm['V_predicted']:.3f}")


if __name__ == "__main__":
    main()
