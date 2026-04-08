#!/usr/bin/env python3
"""
Reprocess existing b01p10 IBM data by fetching raw counts from IBM and
recalculating correlations with the correct formula for |Ψ⁺⟩.
"""

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qiskit_ibm_runtime import QiskitRuntimeService
from leggett import extract_correlations_from_counts_six, calc_leggett_for_angle

def reprocess_job(job_id, phi_deg, phi_rad, num_shots):
    """
    Fetch raw data from IBM job and reprocess with correct correlation formula.
    """
    # Connect to IBM
    service = QiskitRuntimeService()

    print(f"Fetching job {job_id} for φ={phi_deg}°...")
    job = service.job(job_id)
    result = job.result()

    # Extract counts
    pub_result = result[0]
    counts_array = pub_result.data.meas.get_counts()

    # Convert to standard counts dictionary
    counts = {}
    for bitstring, count in counts_array.items():
        counts[bitstring] = count

    # Extract correlations with CORRECT formula for |Ψ⁺⟩
    correlations = extract_correlations_from_counts_six(counts, num_shots, bell_state='psi_plus')

    # Calculate L3 and theory with CORRECT formula for |Ψ⁺⟩
    result_data = calc_leggett_for_angle(correlations, phi_rad, bell_state='psi_plus')

    return {
        'phi_deg': phi_deg,
        'phi_rad': phi_rad,
        **result_data,
        'job_id': job_id,
        'num_shots': num_shots,
        'unique_bitstrings': len(counts),
        'timestamp': job.metrics()['timestamps']['finished']
    }


def main():
    # Load the INCORRECT JSON file to get job IDs
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'leggett_results_ibm_ibm_pittsburgh_sequential_b01p10_12qb.json'
    output_file = input_file.replace('.json', '_CORRECTED.json')

    print(f"Loading job IDs from: {input_file}")
    with open(input_file, 'r') as f:
        old_data = json.load(f)

    print(f"\nReprocessing {len(old_data)} jobs with corrected |Ψ⁺⟩ formula...")

    new_results = []

    for i, entry in enumerate(old_data, 1):
        job_id = entry['job_id']
        phi_deg = entry['phi_deg']
        phi_rad = entry['phi_rad']
        num_shots = entry['num_shots']

        print(f"\n[{i}/{len(old_data)}] Processing φ={phi_deg}°...")

        try:
            result = reprocess_job(job_id, phi_deg, phi_rad, num_shots)
            new_results.append(result)

            # Show comparison
            print(f"  OLD correlations: {entry['correlations']}")
            print(f"  NEW correlations: {result['correlations']}")
            print(f"  OLD theory:       {entry['correlations_theory']}")
            print(f"  NEW theory:       {result['correlations_theory']}")
            print(f"  OLD L3: {entry['L3']:.4f}, NEW L3: {result['L3']:.4f}")
            print(f"  OLD violated: {entry['violated']}, NEW violated: {result['violated']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping this job...")
            continue

    # Sort by angle
    new_results.sort(key=lambda x: x['phi_deg'])

    # Save corrected results
    print(f"\n\nSaving corrected results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(new_results, f, indent=2, default=str)

    print(f"\nDone! Reprocessed {len(new_results)}/{len(old_data)} jobs successfully.")

    # Summary
    violations_old = sum(1 for r in old_data if r['violated'] == 'True' or r['violated'] == True)
    violations_new = sum(1 for r in new_results if r['violated'])

    print(f"\nViolations:")
    print(f"  OLD: {violations_old}/{len(old_data)}")
    print(f"  NEW: {violations_new}/{len(new_results)}")


if __name__ == "__main__":
    main()
