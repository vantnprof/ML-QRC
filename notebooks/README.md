# ML-QRC Manuscript Notebooks

These are the notebooks required for the experiments shown in
`ML_QRC__QCE_QML2026_.pdf`.

## Required Notebooks

Run these in order from the repository root or from this directory:

| Order | Notebook | Manuscript output |
|---:|---|---|
| 1 | `task1_mackey_glass_ml_qrc.ipynb` | Table I and Table III: Mackey-Glass baselines and d-sweep |
| 2 | `task2_tfim_ml_qrc.ipynb` | Table II and Table IV: TFIM baselines and d-sweep |
| 3 | `qubit_count_scaling_ml_qrc.ipynb` | Table V: parameter scalability |
| 4 | `mode_ablation_ml_qrc.ipynb` | Table VI: mode ablation |
| 5 | `noise_robustness_ml_qrc.ipynb` | Table VII: depolarizing-noise sweep |
| 6 | `baseline_isolation_tensor_vs_parameter_count.ipynb` | Supporting baseline-isolation checks discussed in Section IV-D3 |

The notebooks write generated files under `notebooks/artifacts/`.
PyTorch training and QRC feature construction use CUDA automatically when
`torch.cuda.is_available()` is true, and fall back to CPU otherwise. Each
notebook prints the selected `Torch device` near the top.

## Supporting Code

The only retained Python support file is `../scripts/paper_studies/common.py`,
which provides shared numerical helpers used by the Task 1 notebook. The Python
notebook-builder scripts were removed; notebooks are now maintained directly.

## Removed From The Submission Set

The cleanup removed historical backups, ReLU/FC exploratory variants, MNIST
exploratory notebooks, future-direction notebooks for n=8 validation, row-
sequential MNIST, hardware validation, finite-shot noise, and the generated
artifact caches. Those topics remain future work in the current manuscript and
are not needed to reproduce Tables I-VII.

## Current Consistency Note

Before submission, rerun the required notebooks and compare the regenerated
tables against the current PDF. In the pre-cleanup artifacts, Task 2 and the
Task 1 MLP-S row had drifted from the PDF values; that scientific mismatch
should be resolved by rerunning the intended protocol or updating the manuscript
tables before final submission.
