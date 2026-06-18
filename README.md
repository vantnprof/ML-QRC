# ML-QRC

Parameter-efficient multilinear readout for Quantum Reservoir Computing
(QRC).

This repository studies ML-QRC, a Tucker-factored readout layer that keeps the
QRC measurement tensor structured across time steps, qubits, and observables
instead of flattening it directly into one dense linear head.

## What Is In This Repository

- [`notebooks/task1_mackey_glass_ml_qrc.ipynb`](notebooks/task1_mackey_glass_ml_qrc.ipynb):
  Task 1 experiments for Mackey-Glass chaotic time-series forecasting.
- [`notebooks/task2_tfim_ml_qrc.ipynb`](notebooks/task2_tfim_ml_qrc.ipynb):
  Task 2 experiments for transverse-field Ising model (TFIM) phase
  classification.
- [`notebooks/`](notebooks/): additional studies, including qubit scaling, mode
  ablations, noise robustness, and baseline isolation.
- [`scripts/paper_studies/common.py`](scripts/paper_studies/common.py): shared
  utilities used by the notebooks.
- [`requirements.txt`](requirements.txt): Python dependencies.

## Project Summary

Standard QRC pipelines collect expectation values over `T` time steps, `n`
qubits, and `F` observables, then flatten the resulting `T x n x F` tensor into
a dense readout. ML-QRC replaces that flat readout with independent mode-wise
projections over the time, qubit, and observable axes before a final prediction
head.

The project evaluates this idea on two primary tasks:

- Mackey-Glass one-step forecasting, where ML-QRC is competitive with Ridge
  regression while using fewer readout parameters.
- TFIM quantum phase classification, where a smaller ML-QRC configuration
  matches the full MLP baseline at much lower parameter cost.

The repository also includes qubit-count scaling, projection-dimension sweeps,
and mode ablation studies to characterize when the tensor structure helps most.

## Setup

Create an environment and install the dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then start Jupyter from the repository root so notebook imports resolve:

```bash
jupyter notebook
```

The notebooks use NumPy, pandas, matplotlib, scikit-learn, PennyLane, PyTorch,
SciPy, and Jupyter.

## Main Notebooks

### Task 1: Mackey-Glass Forecasting

Run
[`notebooks/task1_mackey_glass_ml_qrc.ipynb`](notebooks/task1_mackey_glass_ml_qrc.ipynb)
to reproduce the Mackey-Glass study. The notebook covers:

- the Mackey-Glass benchmark with a `1000 / 3000 / 1000`
  washout-train-test split;
- one-step forecasting with `T = 50` and the main `n = 6` reservoir;
- full-data and 10 percent low-data regimes;
- Linear, Ridge, MLP, size-matched MLP-S, Ridge-PCA, Tensor-Ridge, and ML-QRC
  readouts;
- fixed-`d = 8` qubit-count scaling for `n in {4, 6, 8, 10}`.

Outputs are written under `notebooks/artifacts/task1_mg_v2/`.

### Task 2: TFIM Phase Classification

Run
[`notebooks/task2_tfim_ml_qrc.ipynb`](notebooks/task2_tfim_ml_qrc.ipynb)
to reproduce the TFIM study. The notebook covers:

- TFIM ground-state data for ferromagnetic and paramagnetic couplings;
- `T = 50`, depolarizing noise level `p = 0.01`, and the main `n = 6`
  reservoir;
- full-data and 50 percent low-data regimes;
- Linear, Ridge, MLP, size-matched MLP-S, Ridge-PCA, Tensor-Ridge, and ML-QRC
  readouts;
- ML-QRC projection-dimension sweeps and fixed-`d = 8` qubit-count scaling.

Outputs are written under `notebooks/artifacts/task3_tfim/`.

## Generated Outputs

Notebook runs generate cached feature banks, CSV tables, and PDF figures under
`notebooks/artifacts/`. These files are reproducible experiment outputs and are
not required to understand or edit the source notebooks.

## Contact

For questions, contact: vantn.prof@gmail.com
