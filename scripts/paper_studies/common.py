from __future__ import annotations

from pathlib import Path
import os
import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def notebook_artifact_dir(name: str) -> Path:
    path = repo_root() / "notebooks" / "artifacts" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_runtime(cache_subdir: str = "cache") -> dict[str, object]:
    root = repo_root()
    cache_root = root / "notebooks" / "artifacts" / cache_subdir
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root.resolve())
    os.environ["MPLCONFIGDIR"] = str((cache_root / "mplconfig").resolve())

    import matplotlib.pyplot as plt
    import pennylane as qml

    plt.rcParams["figure.dpi"] = 1000
    plt.rcParams["savefig.dpi"] = 1000
    plt.rcParams["figure.figsize"] = (3, 2.5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    pd.set_option("display.precision", 6)
    torch.set_num_threads(1)
    return {"plt": plt, "qml": qml, "device": default_torch_device()}


def default_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((yt - yt.mean()) ** 2)
    if denom <= 1e-12:
        return float(np.mean((yt - yp) ** 2))
    return float(np.sum((yt - yp) ** 2) / denom)


class Standardizer:
    def fit(self, x: np.ndarray):
        flat = x.reshape(x.shape[0], -1).astype(np.float64)
        self.mu = flat.mean(axis=0)
        self.sig = flat.std(axis=0)
        self.sig[self.sig < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        flat = x.reshape(shape[0], -1).astype(np.float64)
        out = (flat - self.mu) / self.sig
        return out.reshape(shape).astype(np.float32)


def fit_ols(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    a = np.hstack([x_train.astype(np.float64), np.ones((len(x_train), 1), dtype=np.float64)])
    return np.linalg.lstsq(a, y_train.astype(np.float64), rcond=None)[0].astype(np.float32)


def predict_ols(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.hstack([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)])
    return (a @ weights.astype(np.float64)).astype(np.float32)


def fit_ridge_cv_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alphas: Iterable[float],
    val_frac: float = 0.2,
):
    split = max(1, int(len(x_train) * (1.0 - val_frac)))
    x_fit = x_train[:split].astype(np.float64)
    y_fit = y_train[:split].astype(np.float64)
    x_val = x_train[split:].astype(np.float64)
    y_val = y_train[split:].astype(np.float64)

    mu_x = x_fit.mean(axis=0)
    mu_y = float(y_fit.mean())
    xc = x_fit - mu_x
    yc = y_fit - mu_y

    best_alpha = None
    best_val = np.inf
    for alpha in alphas:
        lam = np.linalg.solve(xc @ xc.T + alpha * np.eye(len(x_fit)), yc)
        w = xc.T @ lam
        b = mu_y - mu_x @ w
        val = nmse(y_val, x_val @ w + b)
        if val < best_val:
            best_val = val
            best_alpha = float(alpha)

    mu_x = x_train.astype(np.float64).mean(axis=0)
    mu_y = float(y_train.mean())
    xc = x_train.astype(np.float64) - mu_x
    yc = y_train.astype(np.float64) - mu_y
    lam = np.linalg.solve(xc @ xc.T + best_alpha * np.eye(len(x_train)), yc)
    w = xc.T @ lam
    b = mu_y - mu_x @ w
    return np.append(w, b).astype(np.float32), best_alpha


class RegressionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLQRCRegressor(nn.Module):
    def __init__(self, t_dim: int, n_dim: int, f_dim: int, d_t: int, d_n: int, d_f: int):
        super().__init__()
        self.u_t = nn.Parameter(torch.randn(d_t, t_dim) * 0.05)
        self.u_n = nn.Parameter(torch.randn(d_n, n_dim) * 0.05)
        self.u_f = nn.Parameter(torch.randn(d_f, f_dim) * 0.05)
        self.head = nn.Linear(d_t * d_n * d_f, 1)

    def forward(self, x):
        y = torch.einsum("btnf,dt->bdnf", x, self.u_t)
        y = torch.einsum("bdnf,in->bdif", y, self.u_n)
        y = torch.einsum("bdif,jf->bdij", y, self.u_f)
        return self.head(y.reshape(y.shape[0], -1)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_regressor(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float | None = None,
    patience: int = 25,
) -> nn.Module:
    device = default_torch_device()
    model = model.to(device)

    if val_frac and len(x_train) > 20:
        split = max(1, int(len(x_train) * (1.0 - val_frac)))
        x_fit = x_train[:split]
        y_fit = y_train[:split]
        x_val = torch.tensor(x_train[split:], dtype=torch.float32, device=device)
        y_val = torch.tensor(y_train[split:], dtype=torch.float32, device=device)
    else:
        x_fit = x_train
        y_fit = y_train
        x_val = None
        y_val = None

    ds = TensorDataset(
        torch.tensor(x_fit, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = np.inf
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        if x_val is None:
            continue
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(x_val), y_val).item())
        if val + 1e-8 < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_regressor(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    return model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def unfold(x: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(x, mode, 0), (x.shape[mode], -1))


def truncated_left_singular_vectors(mat: np.ndarray, rank: int) -> np.ndarray:
    u, _, _ = np.linalg.svd(mat, full_matrices=False)
    return u[:, : min(rank, u.shape[1])].astype(np.float32)


def apply_mode_product(x: np.ndarray, factor: np.ndarray, mode: int) -> np.ndarray:
    moved = np.moveaxis(x, mode, 0)
    flat = moved.reshape(moved.shape[0], -1)
    projected = factor.T @ flat
    shape = (factor.shape[1],) + moved.shape[1:]
    return np.moveaxis(projected.reshape(shape), 0, mode)


def tensor_ridge_project_train(x_train: np.ndarray, target_ranks: tuple[int, int, int]):
    t_rank = min(target_ranks[0], x_train.shape[1])
    n_rank = min(target_ranks[1], x_train.shape[2])
    f_rank = min(target_ranks[2], x_train.shape[3])
    u_t = truncated_left_singular_vectors(unfold(x_train, 1), t_rank)
    u_n = truncated_left_singular_vectors(unfold(x_train, 2), n_rank)
    u_f = truncated_left_singular_vectors(unfold(x_train, 3), f_rank)
    return u_t, u_n, u_f


def tensor_project(x: np.ndarray, u_t: np.ndarray, u_n: np.ndarray, u_f: np.ndarray) -> np.ndarray:
    out = apply_mode_product(x, u_t, 1)
    out = apply_mode_product(out, u_n, 2)
    out = apply_mode_product(out, u_f, 3)
    return out


def generate_mackey_glass(num_points: int, beta: float, gamma: float, q: int, tau: float, step: float, x0: float) -> np.ndarray:
    delay = int(round(tau / step))
    total = delay + num_points
    values = np.full(total + 1, x0, dtype=np.float64)

    def rhs(x_now: float, x_delay: float) -> float:
        return beta * x_delay / (1.0 + x_delay**q) - gamma * x_now

    for idx in range(delay, total):
        x_now = values[idx]
        x_d0 = values[idx - delay]
        x_dh = 0.5 * (values[idx - delay] + values[idx - delay + 1])
        x_d1 = values[idx - delay + 1]
        k1 = rhs(x_now, x_d0)
        k2 = rhs(x_now + 0.5 * step * k1, x_dh)
        k3 = rhs(x_now + 0.5 * step * k2, x_dh)
        k4 = rhs(x_now + step * k3, x_d1)
        values[idx + 1] = x_now + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return values[delay + 1 : delay + 1 + num_points].astype(np.float32)


def build_reservoir_unitary(qml, n_qubits: int, n_layers: int, reservoir_seed: int):
    rng = np.random.default_rng(reservoir_seed)
    weights = rng.uniform(0.0, 2.0 * np.pi, size=(n_layers, n_qubits, 3)).astype(np.float32)
    return torch.tensor(
        qml.matrix(qml.StronglyEntanglingLayers(weights, wires=range(n_qubits)), wire_order=range(n_qubits)),
        dtype=torch.complex64,
    )


def observable_matrices(qml, n_qubits: int):
    def op_mat(op):
        return torch.tensor(qml.matrix(op, wire_order=range(n_qubits)), dtype=torch.complex64)

    out = [op_mat(qml.PauliZ(i)) for i in range(n_qubits)]
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            out.append(op_mat(qml.PauliZ(i) @ qml.PauliZ(j)))
    return out


def batched_ry(states: torch.Tensor, angles: torch.Tensor, wire: int, n_qubits: int) -> torch.Tensor:
    c = torch.cos(angles / 2).to(torch.complex64)
    s = torch.sin(angles / 2).to(torch.complex64)
    gate = torch.stack(
        [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
        dim=-2,
    )
    shaped = states.reshape(states.shape[0], *([2] * n_qubits))
    shaped = torch.movedim(shaped, 1 + wire, -1)
    shaped = torch.einsum("bij,b...j->b...i", gate, shaped)
    shaped = torch.movedim(shaped, -1, 1 + wire)
    return shaped.reshape(states.shape[0], -1)


def build_scalar_qrc_features(
    windows: np.ndarray,
    *,
    qml,
    n_qubits: int,
    n_layers: int,
    t_steps: int,
    reservoir_seed: int,
    cache_path: Path | None = None,
    batch_size: int = 256,
    p_dep: float = 0.0,
):
    f_local = n_qubits
    f_pair = n_qubits * (n_qubits - 1) // 2
    f_dim = f_local + f_pair
    expected_shape = (len(windows), t_steps, n_qubits, f_dim)
    if cache_path and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == expected_shape:
            return cached.astype(np.float32)

    device = default_torch_device()
    reservoir = build_reservoir_unitary(qml, n_qubits, n_layers, reservoir_seed).to(device)
    observables = [obs.to(device) for obs in observable_matrices(qml, n_qubits)]
    zero_state = torch.zeros(2**n_qubits, dtype=torch.complex64, device=device)
    zero_state[0] = 1.0
    pi_val = torch.tensor(float(np.pi), dtype=torch.float32, device=device)
    qi = torch.arange(n_qubits, device=device)
    scale_local = float(1.0 - 4.0 * p_dep / 3.0)
    scale_pair = float(scale_local**2)

    window_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    chunks = []
    for start in range(0, len(window_tensor), batch_size):
        batch = window_tensor[start : start + batch_size]
        bs = batch.shape[0]
        states = zero_state.unsqueeze(0).expand(bs, -1).clone()
        feat = torch.zeros(bs, t_steps, n_qubits, f_dim, dtype=torch.float32, device=device)
        for t_idx in range(t_steps):
            angles = pi_val * batch[:, t_idx]
            for wire in range(n_qubits):
                states = batched_ry(states, angles, wire, n_qubits)
            states = states @ reservoir.T
            evs = torch.stack(
                [torch.real(torch.sum(states.conj() * (states @ obs.T), dim=1)) for obs in observables],
                dim=1,
            )
            evs[:, :f_local] *= scale_local
            evs[:, f_local:] *= scale_pair
            feat[:, t_idx, qi, qi] = evs[:, :f_local]
            pair = 0
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    col = f_local + pair
                    feat[:, t_idx, i, col] = evs[:, col]
                    feat[:, t_idx, j, col] = evs[:, col]
                    pair += 1
        chunks.append(feat.cpu().numpy())
    out = np.concatenate(chunks, axis=0).astype(np.float32)
    if cache_path:
        np.save(cache_path, out)
    return out


def flatten_features(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32)
