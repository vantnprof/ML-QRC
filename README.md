# ML-QRC: Multilinear Quantum Reservoir Computing

**Parameter-Efficient Multilinear Readout for Quantum Reservoir Computing**

Research implementation by **Van Tien Nguyen** and **Panagiotis Markopoulos**.

[Research poster](presentation/ML-QRC_QCE_QML2026_Poster.pdf) · [Experiment notebooks](notebooks/) · [Shared utilities](scripts/paper_studies/common.py)

ML-QRC is a hybrid quantum-classical method that learns a compact readout from a fixed quantum reservoir. It preserves the measurement tensor's time, qubit, and observable axes, learns a projection along each axis, and predicts from the resulting representation using a classical neural head.

- **PennyLane** constructs the reservoir's `StronglyEntanglingLayers` circuit and the Pauli observable matrices; **PyTorch** performs batched statevector evolution and trains the readout.
- The reservoir remains fixed. Only the classical projections and prediction head are trained.
- Experiments cover Mackey–Glass forecasting, transverse-field Ising model (TFIM) phase classification, projection dimensions, qubit scaling, mode ablations, and noise robustness.
- In the main configuration, ML-QRC uses **33,513 readout parameters**, compared with **403,329** for the dense MLP at the same hidden width: approximately **12× fewer parameters**.

## Installation

Use **Python 3.11** and run the commands below from the repository root in a Bash-compatible shell:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --prefix "$VIRTUAL_ENV" \
  --name ml-qrc --display-name "Python (ML-QRC)"
```

The requirements include PennyLane, PyTorch, NumPy, SciPy, pandas, matplotlib, scikit-learn, and the Jupyter notebook and execution tools. CPU execution is supported; the experiment code selects CUDA when it is available to PyTorch.

Dependencies are not version-locked. For a recorded experiment, save the installed versions alongside its outputs:

```bash
python -m pip freeze > environment.txt
```

## Quick start

Check the PennyLane reservoir, tensor features, and trainable PyTorch readout with a small local example. This generates its inputs internally and does not require external datasets or a quantum hardware account:

```bash
python - <<'PY'
import numpy as np
import pennylane as qml
import torch
from scripts.paper_studies.common import (
    MLQRCRegressor, build_scalar_qrc_features, count_params, set_seed,
)

set_seed(0)
windows = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
features = build_scalar_qrc_features(
    windows, qml=qml, n_qubits=2, n_layers=1,
    t_steps=4, reservoir_seed=42, batch_size=4,
)
model = MLQRCRegressor(4, 2, 3, 2, 2, 2, hidden_dim=8)
predictions = model(torch.from_numpy(features))
predictions.square().mean().backward()
assert features.shape == (4, 4, 2, 3)
assert predictions.shape == (4,) and torch.isfinite(predictions).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in model.parameters())
print(f"PennyLane: {qml.__version__}")
print(f"Feature tensor: {features.shape}")
print(f"Predictions: {tuple(predictions.shape)}")
print(f"Readout parameters: {count_params(model)}")
PY
```

Expected shapes are `(4, 4, 2, 3)` for the feature tensor and `(4,)` for predictions, with `99` readout parameters. This checks feature extraction and backpropagation; it does not reproduce the research benchmarks.

To run the experiments interactively:

```bash
jupyter notebook
```

Open a main notebook below, select **Python (ML-QRC)**, and run its cells in order. The kernel registration above is stored inside `.venv`. Some notebooks include a `%pip install` cell; it is redundant after the installation above.

## Method

For each sample, the reservoir produces a tensor `X` with shape `T × n × F`, where `T` is the input-window length, `n` is the number of qubits, and `F = n + n(n − 1)/2` counts the local `Z_i` and pairwise `Z_i Z_j` observables.

ML-QRC applies three learned mode projections:

```text
Input → fixed PennyLane reservoir → measurement tensor X (T × n × F)
      → time, qubit, and observable projections → Y (d_T × d_n × d_F)
      → flatten → hidden layer with ReLU → prediction
```

For equal projection dimensions `d_T = d_n = d_F = d`, hidden width `H`, and a single output, the parameter counts are:

```text
Dense MLP:  H T n F + 2H + 1
ML-QRC:    d(T + n + F) + H d³ + 2H + 1
```

The main setting uses `T = 50`, `n = 6`, `F = 21`, `d = 8`, and `H = 64`. The projections map `50 × 6 × 21` features to `8 × 8 × 8`. Although the qubit mode expands from 6 to 8, the total representation shrinks from 6,300 to 512 entries.

## Experiments

### Mackey–Glass forecasting

Run [`notebooks/task1_mackey_glass_ml_qrc.ipynb`](notebooks/task1_mackey_glass_ml_qrc.ipynb).

The notebook generates the Mackey–Glass sequence internally and evaluates one-step forecasting with a `1000 / 3000 / 1000` washout/train/test split. It compares full training data with a 10% regime of 300 training samples and reports normalized mean squared error (NMSE).

The comparisons include Linear, Ridge, dense MLP, PCA-compressed MLP-S, Ridge-PCA, Tensor-Ridge, and ML-QRC, plus classical ESN and NG-RC references. It also sweeps `d ∈ {4, 8, 16, 32}` and qubit counts `n ∈ {4, 6, 8, 10}` at fixed `d = 8`.

Outputs are saved to `notebooks/artifacts/task1_mg_v2/`.

### TFIM phase classification

Run [`notebooks/task2_tfim_ml_qrc.ipynb`](notebooks/task2_tfim_ml_qrc.ipynb).

The notebook constructs TFIM ground states internally and samples computational-basis sequences for ferromagnetic and paramagnetic phase classification. It uses 16 coupling values over `g/J ∈ [0.2, 1.8]`, 200 sequences per coupling, and a 70/30 train/test split. The low-data regime retains half of the training set; the metric is classification accuracy.

The study compares the same seven QRC readouts, sweeps projection dimensions, and evaluates fixed-`d = 8` qubit scaling. The main reservoir uses `T = 50`, `n = 6`, and `p_dep = 0.01`.

Outputs are saved to `notebooks/artifacts/task3_tfim/`. The directory retains its historical name even though this is Task 2.

### Additional studies

Run the two main notebooks before the baseline-isolation and mode-ablation studies, which require their cached feature banks and result tables.

| Notebook | Study | Output directory under `notebooks/artifacts/` |
| --- | --- | --- |
| [Baseline isolation](notebooks/baseline_isolation_tensor_vs_parameter_count.ipynb) | Compare flat PCA + Ridge with tensor HOSVD + Ridge to separate compression and learned tensor structure | `baseline_isolation/` |
| [Mode ablation](notebooks/mode_ablation_ml_qrc.ipynb) | Freeze the time, qubit, or observable projection using an identity-padded map | `mode_ablation/` |
| [Noise robustness](notebooks/noise_robustness_ml_qrc.ipynb) | Compare Ridge and ML-QRC on Mackey–Glass at `p_dep ∈ {0, 0.005, 0.01, 0.02, 0.05}` | `noise_robustness/` |
| [Qubit-count scaling](notebooks/qubit_count_scaling_ml_qrc.ipynb) | Standalone Mackey–Glass scaling study with `n ∈ {4, 6, 8, 10}` and `d = 8` | `qubit_scaling/` |

The main feature builders model noise by rescaling local and pairwise expectation values. The noise-robustness notebook instead samples single-qubit Pauli errors during reservoir evolution, perturbing the state trajectory itself. These are distinct simulation models.

### Noninteractive execution

To execute the main notebooks from the command line and retain their outputs in separate notebook files:

```bash
mkdir -p notebooks/artifacts/executed
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=ml-qrc \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir=notebooks/artifacts/executed \
  notebooks/task1_mackey_glass_ml_qrc.ipynb

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=ml-qrc \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir=notebooks/artifacts/executed \
  notebooks/task2_tfim_ml_qrc.ipynb
```

Full runs include multiple seeds, feature generation, and model sweeps. Larger reservoirs require substantially more memory and computation because statevectors and the dense matrices grow with qubit count. Adjust each notebook's `CONFIG` for exploratory runs; reduced settings do not reproduce the reported benchmarks.

## Results reported in the poster

The following values are transcribed from the [research poster](presentation/ML-QRC_QCE_QML2026_Poster.pdf), for `n = 6`, `d = 8`, and 10 evaluation seeds. Neural-readout values are mean ± standard deviation. They have not been recomputed by the quick start.

| Readout | Parameters | MG NMSE, full data ↓ | MG NMSE, 10% data ↓ | TFIM accuracy, full data ↑ | TFIM accuracy, 50% data ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 6,301 | 0.0085 | 0.7572 | 69.9% | 72.6 ± 1.0% |
| Dense MLP | 403,329 | 0.0060 ± 0.0012 | 1.6105 ± 0.2529 | 69.5 ± 0.9% | 67.4 ± 1.4% |
| ML-QRC | 33,513 | 0.0086 ± 0.0013 | 0.9684 ± 0.1581 | 70.5 ± 1.5% | 66.6 ± 1.8% |

ML-QRC reduces the dense neural readout's parameter count while achieving a similar full-data MG NMSE to Ridge and competitive full-data TFIM accuracy. Ridge remains stronger in the reported low-data settings. The parameter reduction concerns the classical readout; it does not establish a quantum runtime advantage. Finite-shot and hardware validation remain future work.

## Outputs and reproducibility

Experiments write cached NumPy feature banks, CSV metrics, and PDF/PNG figures under `notebooks/artifacts/`:

| Directory | Selected outputs |
| --- | --- |
| `task1_mg_v2/` | `table1_main.csv`, `table2_main.csv`, `table9_d_sweep.csv`, `table10_qubit_scaling.csv`, `task1_run_details.csv`, cached `.npy` features |
| `task3_tfim/` | `table_phase.csv`, `table10_d_sweep.csv`, `table11_qubit_scaling_tfim.csv`, `task2_run_details.csv`, cached `.npy` features |
| `baseline_isolation/` | Task-specific baseline CSV tables and comparison figures |
| `mode_ablation/` | `table_mode_ablation.csv`, per-seed scores, summaries, and figures |
| `noise_robustness/` | Noise-sweep scores, summaries, cached features, and figures |

The main notebooks default to evaluation seeds 0–9, reservoir seed 42, Adam training for 300 epochs, batch size 32, learning rate `1e-3`, and weight decay `1e-5`. The noise study uses five evaluation seeds. Inspect each notebook's `CONFIG` for its complete settings.

Caches and output filenames encode only part of the configuration. When changing feature-generation settings, remove the affected cache or use a new cache path, and preserve previous results before rerunning. Reruns can overwrite tables and figures. Numerical results may vary across dependency versions and CPU/CUDA execution.

PCA dimensions and fixed HOSVD ranks are capped by the available training samples and tensor-mode sizes. In particular, the fixed tensor baseline uses feasible ranks `(8, 6, 8)` for the main six-qubit configuration.

**Configuration note:** the current main notebooks request 522 PCA components for MLP-S, yielding 33,537 neural parameters when that rank is feasible. The local manuscript describes PCA-512 for that baseline; 512 components would yield 32,897 neural parameters. Low-data runs cap the PCA rank further. The table above quotes the poster's results; consult the effective dimensions and parameter counts produced by your run when comparing them.

## Repository layout

```text
notebooks/
  task1_mackey_glass_ml_qrc.ipynb           Main forecasting experiment
  task2_tfim_ml_qrc.ipynb                   Main classification experiment
  baseline_isolation_tensor_vs_parameter_count.ipynb
  mode_ablation_ml_qrc.ipynb
  noise_robustness_ml_qrc.ipynb
  qubit_count_scaling_ml_qrc.ipynb
  *executed.ipynb                          Saved executed notebook copies
  artifacts/                              Generated caches, tables, and figures
scripts/
  paper_studies/common.py                  Shared reservoir and readout utilities
presentation/
  ML-QRC_QCE_QML2026_Poster.pdf             Research poster
requirements.txt                          Runtime and notebook dependencies
```

## Citation

If you use this implementation, cite the repository:

```bibtex
@inproceedings{nguyen2026mlqrc,
  author    = {Nguyen, Van Tien and Markopoulos, Panagiotis},
  title     = {Parameter-Efficient Multilinear Readout for Quantum Reservoir Computing},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Quantum Computing and Engineering (QCE), 4th International Workshop on Quantum Machine Learning: From Research to Practice (QML@QCE)},
  address   = {Toronto, ON, Canada},
  publisher = {IEEE},
  month     = sep,
  year      = {2026},
  note      = {To appear}
}
```

## Contact

**Van Tien Nguyen**

[tien.nguyen@utsa.edu](mailto:tien.nguyen@utsa.edu) · [vantn.prof@gmail.com](mailto:vantn.prof@gmail.com)

[Personal website](https://vantnprof.github.io)
