<img width="6221" height="4785" alt="radar_all_angles_combo_24qb" src="https://github.com/user-attachments/assets/bd2653fa-a481-4470-8c76-3f339131beab" /># Leggett Inequality Tests on Quantum Hardware

Testing the Leggett inequality (arXiv:0801.2241v2, Branciard et al.) across multiple quantum computing platforms: IBM, IonQ, Quantinuum, Rigetti, and IQM.

<img width="6366" height="4686" alt="radar_rotated_vs_new_comparison" src="https://github.com/user-attachments/assets/297327a6-3c31-4e2d-85a6-6cb0d353c52c" />

## Project Structure

```
leggett.py              Core library: circuit construction, correlation extraction, L3 calculation
tomography.py           Two-qubit state tomography: density matrix reconstruction, tangle/concurrence

experiments/

  ibm/                  IBM Quantum experiments (sequential, mid-circuit, layout optimization)
  ionq/                 IonQ Forte-1 experiments
  azure/                Azure Quantum (Quantinuum H2, Pasqal, Rigetti QVM)
  braket/               AWS Braket (IQM Emerald, Rigetti Ankaa3)
  iqm/                  IQM Emerald direct access
  crosstalk/            Crosstalk investigation & mitigation experiments
  scanning/             Qubit pair scanning & selection
  tomography/           State tomography experiments

analysis/               Visibility computation, p-values, pair analysis, paper tables
plotting/               Radar plots, vendor comparisons, circuit diagrams
tests/                  Unit tests for singlet preparation and rotation math
data/                   JSON result files and device characterization data
plots/                  Generated plot images
docs/                   Paper draft, notes, and research documentation
```

## Key Results

| Platform              | V_predicted | V_experimental | Violations |
|-----------------------|-------------|----------------|------------|
| IBM Pittsburgh (best) | 0.985       | 0.966          | 7/8 (88%)  |
| IonQ Forte-1          | 0.979       | 0.984          | 10/10      |

## Circuit Approaches

- **Sequential (24qb)**: 12 singlet pairs measured simultaneously (6 for +phi, 6 for -phi)
- **Sequential (12qb)**: 6 pairs for +phi only, separate runs for each angle
- **Mid-circuit**: 2 qubits reused via mid-circuit measurement and reset
- **Large-scale**: 64-70 pairs for redundant correlation averaging

## Running Experiments

All experiment scripts are run from the repo root:

```bash
python experiments/ibm/ibm_leggett_test_sequential.py
python experiments/ionq/ionq_leggett_test.py
```
