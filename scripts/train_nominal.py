#!/usr/bin/env python3
"""Train a strict cyclic PD response operator with hard force closure.

The network predicts the closed-loop rail state, rail acceleration, and shell
strain energy at a requested time.  Motor thrust and mechanical shell
resistance are not independent neural heads:

    pdot_hat = PD_projection(p_hat, u_hat, reference)
    R_hat = u_hat - m_r * pddot_hat

The network predicts displacement error and thrust, then the velocity error is
projected analytically onto the archived PD law.  This removes the inconsistent
independent state/force labels used by earlier Phase-10 models.
All normalization and optimization use trajectory split 0 only, checkpoint
selection uses split 1, and split 2 is evaluated once after training.
"""

from __future__ import annotations

import argparse
import secrets
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from response_data import (
    _attach_static_context_arrays,
    _load_dataset,
    _metrics,
    _safe_std,
)
from response_data import _terminal_reference
from response_data import _attach_pid_control_arrays

N_RAILS = 12


class CircularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = torch.nn.functional.silu(self.norm1(value))
        value = self.conv1(value.transpose(1, 2)).transpose(1, 2)
        value = self.dropout(value)
        value = torch.nn.functional.silu(self.norm2(value))
        value = self.conv2(value.transpose(1, 2)).transpose(1, 2)
        return residual + value


class PhysicsConstrainedResponseOperator(nn.Module):
    """Cyclic-equivariant rail state model with an invariant energy head."""

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            CircularResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.energy_local_encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.energy_global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.energy_blocks = nn.ModuleList(
            CircularResidualBlock(hidden_dim, dropout)
            for _ in range(max(2, num_blocks // 2))
        )
        self.energy_norm = nn.LayerNorm(hidden_dim)
        self.displacement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.velocity_prior_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.acceleration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.force_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.energy_state_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        pooled_dim = 4 * hidden_dim
        self.energy_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, local_features: torch.Tensor, global_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_latent = self.global_encoder(global_features)
        node = self.local_encoder(local_features) + global_latent[:, None, :]
        for block in self.blocks:
            node = block(node)
        node = self.final_norm(node)
        displacement = self.displacement_head(node).squeeze(-1)
        velocity_prior = self.velocity_prior_head(node).squeeze(-1)
        acceleration = self.acceleration_head(node).squeeze(-1)
        force_residual = self.force_residual_head(node).squeeze(-1)
        energy_global = self.energy_global_encoder(global_features)
        energy_node = (
            self.energy_local_encoder(local_features) + energy_global[:, None, :]
        )
        energy_node = energy_node + self.energy_state_projection(
            torch.stack((displacement, velocity_prior, force_residual), dim=-1)
        )
        for block in self.energy_blocks:
            energy_node = block(energy_node)
        energy_node = self.energy_norm(energy_node)
        pooled = torch.cat(
            (
                energy_node.mean(dim=1),
                energy_node.amax(dim=1),
                torch.sqrt(torch.mean(energy_node**2, dim=1) + 1.0e-12),
                energy_global,
            ),
            dim=1,
        )
        energy = self.energy_head(pooled)
        return displacement, velocity_prior, acceleration, energy, force_residual


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in source.items():
                if value is None or value == "":
                    continue
                row[key] = int(value) if key == "epoch" else float(value)
            rows.append(row)
    return rows


def _next_aligned(
    values: np.ndarray, trajectory_id: np.ndarray, step_index: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = values.copy()
    trajectory = np.asarray(trajectory_id, dtype=np.int64).reshape(-1)
    step = np.asarray(step_index, dtype=np.int64).reshape(-1)
    for tid in np.unique(trajectory):
        rows = np.flatnonzero(trajectory == int(tid))
        rows = rows[np.argsort(step[rows])]
        if rows.size <= 1:
            continue
        consecutive = np.diff(step[rows]) == 1
        result[rows[:-1][consecutive]] = values[rows[1:][consecutive]]
    return result


def _build_features(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    terminal = _terminal_reference(arrays)
    dt = np.asarray(arrays["dt"], dtype=np.float64).reshape(-1, 1)
    time_value = np.asarray(arrays["time"], dtype=np.float64).reshape(-1, 1)
    sim_time = np.asarray(arrays["sim_time"], dtype=np.float64).reshape(-1, 1)
    tau = time_value / np.maximum(sim_time, 1.0e-12)
    arctan_k = np.asarray(arrays["arctan_k"], dtype=np.float64).reshape(-1, 1)
    ramp = (2.0 / np.pi) * np.arctan(arctan_k * tau)
    u_scale = np.asarray(arrays["arctan_u_scale"], dtype=np.float64)
    feedforward = u_scale * ramp
    pdot_ref = np.asarray(arrays["pdot_ref"], dtype=np.float64)
    pdot_ref_next = _next_aligned(
        pdot_ref, arrays["trajectory_id"], arrays["step_index"]
    )
    reference_acceleration = (pdot_ref_next - pdot_ref) / np.maximum(dt, 1.0e-12)

    local = np.stack(
        (
            np.asarray(arrays["p_ref"], dtype=np.float64),
            pdot_ref,
            terminal,
            u_scale,
            feedforward,
            reference_acceleration,
        ),
        axis=2,
    )
    kp = np.asarray(arrays["pid_kp"], dtype=np.float64)[:, :1]
    kd = np.asarray(arrays["pid_kd"], dtype=np.float64)[:, :1]
    ki = np.asarray(arrays["pid_ki"], dtype=np.float64)[:, :1]
    sign = np.asarray(arrays["pid_error_sign"], dtype=np.float64)[:, :1]
    terminal_summary = np.column_stack(
        (
            terminal.mean(axis=1),
            terminal.std(axis=1),
            np.sqrt(np.mean(terminal**2, axis=1)),
        )
    )
    force_summary = np.column_stack(
        (
            u_scale.mean(axis=1),
            u_scale.std(axis=1),
            np.sqrt(np.mean(u_scale**2, axis=1)),
        )
    )
    global_features = np.hstack(
        (
            arctan_k,
            sim_time,
            dt,
            tau,
            ramp,
            kp,
            kd,
            ki,
            sign,
            np.log10(np.maximum(np.abs(kp), 1.0)),
            np.log10(np.maximum(np.abs(kd), 1.0)),
            terminal_summary,
            force_summary,
            np.asarray(arrays["static_context"], dtype=np.float64),
        )
    )
    local_layout = [
        "p_ref",
        "pdot_ref",
        "p_ref_terminal",
        "arctan_u_scale",
        "feedforward_force",
        "pddot_ref_fd",
    ]
    global_layout = [
        "arctan_k",
        "sim_time",
        "dt",
        "tau",
        "arctan_ramp",
        "pid_kp",
        "pid_kd",
        "pid_ki",
        "pid_error_sign",
        "log10_abs_pid_kp",
        "log10_abs_pid_kd",
        "terminal_mean",
        "terminal_std",
        "terminal_rms",
        "u_scale_mean",
        "u_scale_std",
        "u_scale_rms",
        *np.asarray(arrays["static_context_fields"], dtype=str).tolist(),
    ]
    return local, global_features, local_layout, global_layout


def _normalization(
    local: np.ndarray,
    global_features: np.ndarray,
    state_residual: np.ndarray,
    acceleration: np.ndarray,
    energy: np.ndarray,
    force_residual: np.ndarray,
    train: np.ndarray,
) -> dict[str, np.ndarray]:
    local_train = local[train].reshape(-1, local.shape[-1])
    state_train = state_residual[train].reshape(-1, state_residual.shape[-1])
    return {
        "local_mean": local_train.mean(axis=0).reshape(1, 1, -1),
        "local_std": _safe_std(local_train.std(axis=0)).reshape(1, 1, -1),
        "global_mean": global_features[train].mean(axis=0, keepdims=True),
        "global_std": _safe_std(global_features[train].std(axis=0, keepdims=True)),
        "state_mean": state_train.mean(axis=0).reshape(1, 1, -1),
        "state_std": _safe_std(state_train.std(axis=0)).reshape(1, 1, -1),
        "accel_mean": np.asarray([[acceleration[train].mean()]], dtype=np.float64),
        "accel_std": np.asarray(
            [[max(float(acceleration[train].std()), 1.0e-12)]], dtype=np.float64
        ),
        "energy_mean": energy[train].mean(axis=0, keepdims=True),
        "energy_std": _safe_std(energy[train].std(axis=0, keepdims=True)),
        "force_residual_mean": np.asarray(
            [[force_residual[train].mean()]], dtype=np.float64
        ),
        "force_residual_std": np.asarray(
            [[max(float(force_residual[train].std()), 1.0e-12)]],
            dtype=np.float64,
        ),
    }


def _analytic_force_torch(
    p: torch.Tensor,
    velocity: torch.Tensor,
    p_ref: torch.Tensor,
    pdot_ref: torch.Tensor,
    feedforward: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    sign: torch.Tensor,
    u_min: torch.Tensor,
    u_max: torch.Tensor,
) -> torch.Tensor:
    force = feedforward + sign * (kp * (p_ref - p) + kd * (pdot_ref - velocity))
    return torch.clamp(force, min=u_min, max=u_max)


def _analytic_force_numpy(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    p: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    time_value = np.asarray(arrays["time"][indices], dtype=np.float64).reshape(-1, 1)
    sim_time = np.asarray(arrays["sim_time"][indices], dtype=np.float64).reshape(-1, 1)
    arctan_k = np.asarray(arrays["arctan_k"][indices], dtype=np.float64).reshape(-1, 1)
    ramp = (2.0 / np.pi) * np.arctan(
        arctan_k * time_value / np.maximum(sim_time, 1.0e-12)
    )
    feedforward = np.asarray(arrays["arctan_u_scale"][indices], dtype=np.float64) * ramp
    force = feedforward + np.asarray(arrays["pid_error_sign"][indices]) * (
        np.asarray(arrays["pid_kp"][indices])
        * (np.asarray(arrays["p_ref"][indices]) - p)
        + np.asarray(arrays["pid_kd"][indices])
        * (np.asarray(arrays["pdot_ref"][indices]) - velocity)
    )
    return np.clip(
        force,
        np.asarray(arrays["u_min"][indices]),
        np.asarray(arrays["u_max"][indices]),
    )


def _predict(
    model: PhysicsConstrainedResponseOperator,
    local_norm: np.ndarray,
    global_norm: np.ndarray,
    norm: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    rail_mass: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    displacement_rows: list[np.ndarray] = []
    accel_rows: list[np.ndarray] = []
    energy_rows: list[np.ndarray] = []
    force_residual_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            rows = indices[start : start + batch_size]
            (
                displacement_norm,
                _velocity_prior_norm,
                accel_norm,
                energy_norm,
                force_residual_norm,
            ) = model(
                torch.as_tensor(local_norm[rows], dtype=torch.float32, device=device),
                torch.as_tensor(global_norm[rows], dtype=torch.float32, device=device),
            )
            displacement_rows.append(displacement_norm.cpu().numpy())
            accel_rows.append(accel_norm.cpu().numpy())
            energy_rows.append(energy_norm.cpu().numpy())
            force_residual_rows.append(force_residual_norm.cpu().numpy())
    displacement_residual = (
        np.concatenate(displacement_rows, axis=0) * norm["state_std"][:, :, 0]
        + norm["state_mean"][:, :, 0]
    )
    acceleration = (
        np.concatenate(accel_rows, axis=0) * norm["accel_std"] + norm["accel_mean"]
    )
    energy = (
        np.concatenate(energy_rows, axis=0) * norm["energy_std"] + norm["energy_mean"]
    )
    force_residual = (
        np.concatenate(force_residual_rows, axis=0) * norm["force_residual_std"]
        + norm["force_residual_mean"]
    )
    time_value = np.asarray(arrays["time"][indices], dtype=np.float64).reshape(-1, 1)
    sim_time = np.asarray(arrays["sim_time"][indices], dtype=np.float64).reshape(-1, 1)
    arctan_k = np.asarray(arrays["arctan_k"][indices], dtype=np.float64).reshape(-1, 1)
    ramp = (2.0 / np.pi) * np.arctan(
        arctan_k * time_value / np.maximum(sim_time, 1.0e-12)
    )
    feedforward = np.asarray(arrays["arctan_u_scale"][indices], dtype=np.float64) * ramp
    force = np.clip(
        feedforward + force_residual,
        np.asarray(arrays["u_min"][indices]),
        np.asarray(arrays["u_max"][indices]),
    )
    kp = np.asarray(arrays["pid_kp"][indices], dtype=np.float64)
    kd = np.asarray(arrays["pid_kd"][indices], dtype=np.float64)
    sign = np.asarray(arrays["pid_error_sign"][indices], dtype=np.float64)
    velocity_residual = (
        sign * (feedforward - force) - kp * displacement_residual
    ) / np.maximum(np.abs(kd), 1.0e-12)
    p = np.asarray(arrays["p_ref"][indices], dtype=np.float64) + displacement_residual
    velocity = (
        np.asarray(arrays["pdot_ref"][indices], dtype=np.float64) + velocity_residual
    )
    resistance = force - float(rail_mass) * acceleration
    return {
        "p": p,
        "pdot": velocity,
        "accel": acceleration,
        "energy": energy,
        "force": force,
        "resistance": resistance,
    }


def _evaluate(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    truth_state = np.hstack((arrays["p"][indices], arrays["pdot"][indices]))
    pred_state = np.hstack((predictions["p"], predictions["pdot"]))
    return {
        f"{prefix}_p_rel_rmse_pct": _metrics(arrays["p"][indices], predictions["p"])[
            "rel_rmse_pct"
        ],
        f"{prefix}_pdot_rel_rmse_pct": _metrics(
            arrays["pdot"][indices], predictions["pdot"]
        )["rel_rmse_pct"],
        f"{prefix}_state_rel_rmse_pct": _metrics(truth_state, pred_state)[
            "rel_rmse_pct"
        ],
        f"{prefix}_energy_rel_rmse_pct": _metrics(
            arrays["U_current"][indices], predictions["energy"]
        )["rel_rmse_pct"],
        f"{prefix}_resistance_rel_rmse_pct": _metrics(
            arrays["mechanical_resistance"][indices], predictions["resistance"]
        )["rel_rmse_pct"],
        f"{prefix}_force_rel_rmse_pct": _metrics(
            arrays["u"][indices], predictions["force"]
        )["rel_rmse_pct"],
    }


def _plot_history(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    epoch = np.asarray([row["epoch"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    ax.plot(epoch, [row["train_loss"] for row in rows], label="training objective")
    ax.plot(epoch, [row["val_score"] for row in rows], label="validation score")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / score (log scale)")
    ax.set_title("Physics-constrained response-operator optimization")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_history.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=160, sharey=True)
    for ax, split_name in zip(axes, ("train", "val")):
        for key, label in (
            ("p_rel_rmse_pct", "displacement"),
            ("pdot_rel_rmse_pct", "velocity"),
            ("energy_rel_rmse_pct", "strain energy"),
            ("resistance_rel_rmse_pct", "mechanical resistance"),
            ("force_rel_rmse_pct", "PD thrust"),
        ):
            values = [row[f"{split_name}_{key}"] for row in rows]
            ax.plot(epoch, values, marker="o", markersize=3, label=label)
        ax.axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="5% target")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(f"{split_name} full-time errors")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("relative RMSE / % (log scale)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_history.png")
    plt.close(fig)


def _per_trajectory_rows(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
) -> list[dict[str, Any]]:
    trajectory = np.asarray(arrays["trajectory_id"][indices], dtype=np.int64).reshape(
        -1
    )
    rows: list[dict[str, Any]] = []
    for tid in np.unique(trajectory):
        local = np.flatnonzero(trajectory == int(tid))
        global_rows = indices[local]
        metric = _evaluate(
            {key: value[local] for key, value in predictions.items()},
            arrays,
            global_rows,
            "test",
        )
        rows.append(
            {
                "trajectory_id": int(tid),
                "case_name": str(np.asarray(arrays["source_cases"])[int(tid)]),
                "rows": int(local.size),
                **metric,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_DIR / "data/processed/nominal.npz"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "outputs/nominal_training"
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--p-loss-weight", type=float, default=1.0)
    parser.add_argument("--pdot-loss-weight", type=float, default=2.0)
    parser.add_argument("--energy-loss-weight", type=float, default=2.0)
    parser.add_argument("--accel-loss-weight", type=float, default=0.1)
    parser.add_argument("--force-loss-weight", type=float, default=8.0)
    parser.add_argument("--resistance-loss-weight", type=float, default=4.0)
    parser.add_argument("--rail-mass", type=float, default=7.27)
    parser.add_argument("--train-split-id", type=int, default=0)
    parser.add_argument("--val-split-id", type=int, default=1)
    parser.add_argument("--test-split-id", type=int, default=2)
    parser.add_argument("--target-threshold-pct", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for a new training run")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Continue from a response-operator checkpoint in a new or existing output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(32)
    if int(args.eval_every) <= 0:
        raise ValueError("--eval-every must be positive.")
    if len({args.train_split_id, args.val_split_id, args.test_split_id}) != 3:
        raise ValueError("train/validation/test split ids must be distinct.")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    arrays = _load_dataset(args.dataset)
    _attach_pid_control_arrays(arrays)
    _attach_static_context_arrays(arrays)
    required = ("mechanical_resistance", "U_current", "accel")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(
            f"{args.dataset} is not an energy-conjugate V24 dataset; missing {missing}."
        )
    split = np.asarray(arrays["split"], dtype=np.int8).reshape(-1)
    train = np.flatnonzero(split == int(args.train_split_id)).astype(np.int64)
    val = np.flatnonzero(split == int(args.val_split_id)).astype(np.int64)
    test = np.flatnonzero(split == int(args.test_split_id)).astype(np.int64)
    if min(train.size, val.size, test.size) == 0:
        raise ValueError(
            f"Empty split: train={train.size}, validation={val.size}, test={test.size}."
        )

    local, global_features, local_layout, global_layout = _build_features(arrays)
    state_residual = np.stack(
        (
            np.asarray(arrays["p"], dtype=np.float64)
            - np.asarray(arrays["p_ref"], dtype=np.float64),
            np.asarray(arrays["pdot"], dtype=np.float64)
            - np.asarray(arrays["pdot_ref"], dtype=np.float64),
        ),
        axis=2,
    )
    acceleration = np.asarray(arrays["accel"], dtype=np.float64)
    energy = np.asarray(arrays["U_current"], dtype=np.float64).reshape(-1, 1)
    time_value = np.asarray(arrays["time"], dtype=np.float64).reshape(-1, 1)
    ramp = (2.0 / np.pi) * np.arctan(
        np.asarray(arrays["arctan_k"], dtype=np.float64).reshape(-1, 1)
        * time_value
        / np.maximum(
            np.asarray(arrays["sim_time"], dtype=np.float64).reshape(-1, 1),
            1.0e-12,
        )
    )
    feedforward = np.asarray(arrays["arctan_u_scale"], dtype=np.float64) * ramp
    force_residual = np.asarray(arrays["u"], dtype=np.float64) - feedforward
    norm = _normalization(
        local,
        global_features,
        state_residual,
        acceleration,
        energy,
        force_residual,
        train,
    )
    local_norm = ((local - norm["local_mean"]) / norm["local_std"]).astype(np.float32)
    global_norm = ((global_features - norm["global_mean"]) / norm["global_std"]).astype(
        np.float32
    )
    state_norm = ((state_residual - norm["state_mean"]) / norm["state_std"]).astype(
        np.float32
    )
    accel_norm = ((acceleration - norm["accel_mean"]) / norm["accel_std"]).astype(
        np.float32
    )
    energy_norm = ((energy - norm["energy_mean"]) / norm["energy_std"]).astype(
        np.float32
    )
    force_residual_norm = (
        (force_residual - norm["force_residual_mean"]) / norm["force_residual_std"]
    ).astype(np.float32)
    force_scale = max(float(np.sqrt(np.mean(np.asarray(arrays["u"])[train] ** 2))), 1.0)
    resistance_scale = max(
        float(
            np.sqrt(np.mean(np.asarray(arrays["mechanical_resistance"])[train] ** 2))
        ),
        1.0,
    )

    dataset = TensorDataset(
        torch.as_tensor(local_norm[train]),
        torch.as_tensor(global_norm[train]),
        torch.as_tensor(state_norm[train]),
        torch.as_tensor(accel_norm[train]),
        torch.as_tensor(energy_norm[train]),
        torch.as_tensor(force_residual_norm[train]),
        torch.as_tensor(np.asarray(arrays["p_ref"])[train], dtype=torch.float32),
        torch.as_tensor(np.asarray(arrays["pdot_ref"])[train], dtype=torch.float32),
        torch.as_tensor(feedforward[train], dtype=torch.float32),
        torch.as_tensor(np.asarray(arrays["pid_kp"])[train], dtype=torch.float32),
        torch.as_tensor(np.asarray(arrays["pid_kd"])[train], dtype=torch.float32),
        torch.as_tensor(
            np.asarray(arrays["pid_error_sign"])[train], dtype=torch.float32
        ),
        torch.as_tensor(np.asarray(arrays["u_min"])[train], dtype=torch.float32),
        torch.as_tensor(np.asarray(arrays["u_max"])[train], dtype=torch.float32),
        torch.as_tensor(np.asarray(arrays["u"])[train], dtype=torch.float32),
        torch.as_tensor(
            np.asarray(arrays["mechanical_resistance"])[train], dtype=torch.float32
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model = PhysicsConstrainedResponseOperator(
        local_dim=local.shape[2],
        global_dim=global_features.shape[1],
        hidden_dim=int(args.hidden_dim),
        num_blocks=int(args.num_blocks),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=3, min_lr=1.0e-6
    )

    state_mean_t = torch.as_tensor(
        norm["state_mean"], dtype=torch.float32, device=device
    )
    state_std_t = torch.as_tensor(norm["state_std"], dtype=torch.float32, device=device)
    accel_mean_t = torch.as_tensor(
        norm["accel_mean"], dtype=torch.float32, device=device
    )
    accel_std_t = torch.as_tensor(norm["accel_std"], dtype=torch.float32, device=device)
    force_residual_mean_t = torch.as_tensor(
        norm["force_residual_mean"], dtype=torch.float32, device=device
    )
    force_residual_std_t = torch.as_tensor(
        norm["force_residual_std"], dtype=torch.float32, device=device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    best_val: dict[str, Any] | None = None
    start_epoch = 1
    elapsed_offset = 0.0
    optimizer_state_restored = False
    scheduler_state_restored = False
    resume_checkpoint: dict[str, Any] | None = None

    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume_checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        resume_config = resume_checkpoint.get("model_config", {})
        expected_config = {
            "local_dim": int(local.shape[2]),
            "global_dim": int(global_features.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "num_blocks": int(args.num_blocks),
        }
        mismatched = {
            key: (resume_config.get(key), value)
            for key, value in expected_config.items()
            if resume_config.get(key) != value
        }
        if mismatched:
            raise ValueError(f"Resume checkpoint architecture mismatch: {mismatched}")
        checkpoint_norm = resume_checkpoint.get("normalization")
        if checkpoint_norm is None:
            raise KeyError("Resume checkpoint does not contain normalization data.")
        for key, value in norm.items():
            if key not in checkpoint_norm or not np.allclose(
                np.asarray(checkpoint_norm[key]),
                np.asarray(value),
                rtol=1.0e-7,
                atol=1.0e-10,
            ):
                raise ValueError(f"Resume checkpoint normalization mismatch for {key}.")

        model.load_state_dict(resume_checkpoint["state_dict"])
        resume_epoch = int(resume_checkpoint["epoch"])
        if int(args.epochs) <= resume_epoch:
            raise ValueError(
                f"--epochs must exceed checkpoint epoch {resume_epoch}; got {args.epochs}."
            )
        start_epoch = resume_epoch + 1
        source_history = _read_history(resume_path.parent / "history.csv")
        history = [row for row in source_history if int(row["epoch"]) <= resume_epoch]
        if not history:
            raise FileNotFoundError(
                f"No compatible history rows found beside resume checkpoint: {resume_path.parent / 'history.csv'}"
            )
        elapsed_offset = float(history[-1].get("elapsed_s", 0.0))
        best_row = min(history, key=lambda row: float(row["val_score"]))
        best_score = float(
            resume_checkpoint.get("validation_score", best_row["val_score"])
        )
        best_epoch = int(resume_checkpoint.get("epoch", best_row["epoch"]))
        best_val = dict(best_row)

        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            optimizer_state_restored = True
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            scheduler_state_restored = True
        else:
            for row in history:
                scheduler.step(float(row["val_score"]))
            restored_lr = float(history[-1]["lr"])
            for group in optimizer.param_groups:
                group["lr"] = restored_lr

        torch.save(
            resume_checkpoint,
            args.output_dir / "best_physics_response_operator.pt",
        )

    run_config = {
        **vars(args),
        "resume_epoch": int(start_epoch - 1) if args.resume_from is not None else None,
        "optimizer_state_restored": optimizer_state_restored,
        "scheduler_state_restored": scheduler_state_restored,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[phase10-v24] "
        f"rows={split.size} train={train.size} val={val.size} test={test.size} "
        f"local_dim={local.shape[2]} global_dim={global_features.shape[1]} "
        f"parameters={sum(p.numel() for p in model.parameters())} device={device} "
        "strict_split=true pd_manifold_projection=hard resistance_closure=hard",
        flush=True,
    )
    if args.resume_from is not None:
        print(
            "[phase10-v24-resume] "
            f"checkpoint={args.resume_from.resolve()} start_epoch={start_epoch} "
            f"target_epoch={args.epochs} best={best_score:.6g}@{best_epoch} "
            f"optimizer_state_restored={str(optimizer_state_restored).lower()} "
            f"scheduler_state_restored={str(scheduler_state_restored).lower()}",
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            (
                local_b,
                global_b,
                state_b,
                accel_b,
                energy_b,
                force_residual_b,
                p_ref_b,
                pdot_ref_b,
                feedforward_b,
                kp_b,
                kd_b,
                sign_b,
                u_min_b,
                u_max_b,
                force_truth_b,
                resistance_truth_b,
            ) = (value.to(device, non_blocking=True) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            (
                displacement_pred,
                velocity_prior_pred,
                accel_pred,
                energy_pred,
                force_residual_pred,
            ) = model(local_b, global_b)
            displacement_phys = (
                displacement_pred * state_std_t[:, :, 0] + state_mean_t[:, :, 0]
            )
            accel_phys = accel_pred * accel_std_t + accel_mean_t
            force_residual_phys = (
                force_residual_pred * force_residual_std_t + force_residual_mean_t
            )
            force_pred = torch.clamp(
                feedforward_b + force_residual_phys, min=u_min_b, max=u_max_b
            )
            velocity_residual = (
                sign_b * (feedforward_b - force_pred) - kp_b * displacement_phys
            ) / torch.clamp(torch.abs(kd_b), min=1.0e-12)
            p_pred = p_ref_b + displacement_phys
            pdot_pred = pdot_ref_b + velocity_residual
            resistance_pred = force_pred - float(args.rail_mass) * accel_phys
            loss = (
                float(args.p_loss_weight)
                * torch.mean((displacement_pred - state_b[:, :, 0]) ** 2)
                + float(args.pdot_loss_weight)
                * 0.5
                * (
                    torch.mean((velocity_prior_pred - state_b[:, :, 1]) ** 2)
                    + torch.mean(
                        (
                            (velocity_residual - state_mean_t[:, :, 1])
                            / state_std_t[:, :, 1]
                            - state_b[:, :, 1]
                        )
                        ** 2
                    )
                )
                + float(args.energy_loss_weight)
                * torch.mean((energy_pred - energy_b) ** 2)
                + float(args.accel_loss_weight)
                * torch.mean((accel_pred - accel_b) ** 2)
                + float(args.force_loss_weight)
                * torch.mean((force_residual_pred - force_residual_b) ** 2)
                + float(args.force_loss_weight)
                * torch.mean(((force_pred - force_truth_b) / force_scale) ** 2)
                + float(args.resistance_loss_weight)
                * torch.mean(
                    ((resistance_pred - resistance_truth_b) / resistance_scale) ** 2
                )
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * local_b.shape[0]
            total_rows += int(local_b.shape[0])

        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs):
            train_pred = _predict(
                model,
                local_norm,
                global_norm,
                norm,
                arrays,
                train,
                float(args.rail_mass),
                device,
                int(args.batch_size),
            )
            val_pred = _predict(
                model,
                local_norm,
                global_norm,
                norm,
                arrays,
                val,
                float(args.rail_mass),
                device,
                int(args.batch_size),
            )
            row: dict[str, Any] = {
                "epoch": int(epoch),
                "train_loss": total_loss / max(total_rows, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": float(elapsed_offset + time.perf_counter() - started),
            }
            row.update(_evaluate(train_pred, arrays, train, "train"))
            row.update(_evaluate(val_pred, arrays, val, "val"))
            selection_keys = (
                "val_p_rel_rmse_pct",
                "val_energy_rel_rmse_pct",
                "val_resistance_rel_rmse_pct",
                "val_force_rel_rmse_pct",
            )
            row["val_score"] = float(sum(float(row[key]) for key in selection_keys))
            history.append(row)
            scheduler.step(float(row["val_score"]))
            improved = float(row["val_score"]) < best_score
            if improved:
                best_score = float(row["val_score"])
                best_epoch = int(epoch)
                best_val = dict(row)
            checkpoint_payload = {
                "state_dict": model.state_dict(),
                "model_config": {
                    "local_dim": int(local.shape[2]),
                    "global_dim": int(global_features.shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "num_blocks": int(args.num_blocks),
                    "dropout": float(args.dropout),
                    "model_kind": "cyclic_physics_constrained_pd_response_operator",
                    "local_layout": local_layout,
                    "global_layout": global_layout,
                    "force_closure": "predicted thrust with velocity projected onto exact archived PD law",
                    "resistance_closure": "u - rail_mass * pddot",
                },
                "normalization": norm,
                "rail_mass": float(args.rail_mass),
                "epoch": int(epoch),
                "validation_score": float(row["val_score"]),
                "best_epoch": int(best_epoch),
                "best_validation_score": float(best_score),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(
                checkpoint_payload,
                args.output_dir / "last_physics_response_operator.pt",
            )
            if improved:
                torch.save(
                    checkpoint_payload,
                    args.output_dir / "best_physics_response_operator.pt",
                )
            _write_csv(args.output_dir / "history.csv", history)
            _plot_history(history, args.output_dir)
            print(
                "[phase10-v24-eval] "
                f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.5g} "
                f"train[p={row['train_p_rel_rmse_pct']:.3f}% "
                f"U={row['train_energy_rel_rmse_pct']:.3f}% "
                f"R={row['train_resistance_rel_rmse_pct']:.3f}% "
                f"F={row['train_force_rel_rmse_pct']:.3f}%] "
                f"val[p={row['val_p_rel_rmse_pct']:.3f}% "
                f"U={row['val_energy_rel_rmse_pct']:.3f}% "
                f"R={row['val_resistance_rel_rmse_pct']:.3f}% "
                f"F={row['val_force_rel_rmse_pct']:.3f}%] "
                f"score={row['val_score']:.3f} best={best_score:.3f}@{best_epoch}",
                flush=True,
            )

    checkpoint_path = args.output_dir / "best_physics_response_operator.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_pred = _predict(
        model,
        local_norm,
        global_norm,
        norm,
        arrays,
        test,
        float(args.rail_mass),
        device,
        int(args.batch_size),
    )
    test_metrics = _evaluate(test_pred, arrays, test, "test")
    checked = (
        "test_p_rel_rmse_pct",
        "test_energy_rel_rmse_pct",
        "test_resistance_rel_rmse_pct",
        "test_force_rel_rmse_pct",
    )
    target_passed = all(
        float(test_metrics[key]) < float(args.target_threshold_pct) for key in checked
    )
    per_trajectory = _per_trajectory_rows(test_pred, arrays, test)
    _write_csv(args.output_dir / "test_metrics_by_trajectory.csv", per_trajectory)
    np.savez_compressed(
        args.output_dir / "strict_test_predictions.npz",
        row_index=test,
        trajectory_id=np.asarray(arrays["trajectory_id"])[test],
        step_index=np.asarray(arrays["step_index"])[test],
        time=np.asarray(arrays["time"])[test],
        p_truth=np.asarray(arrays["p"])[test],
        p_pred=test_pred["p"],
        pdot_truth=np.asarray(arrays["pdot"])[test],
        pdot_pred=test_pred["pdot"],
        accel_truth=np.asarray(arrays["accel"])[test],
        accel_pred=test_pred["accel"],
        energy_truth=np.asarray(arrays["U_current"])[test],
        energy_pred=test_pred["energy"],
        resistance_truth=np.asarray(arrays["mechanical_resistance"])[test],
        resistance_pred=test_pred["resistance"],
        force_truth=np.asarray(arrays["u"])[test],
        force_pred=test_pred["force"],
    )
    manifest = {
        "status": "success",
        "model_kind": "cyclic_physics_constrained_pd_response_operator",
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_metrics": best_val,
        "strict_test_metrics": test_metrics,
        "target_threshold_pct": float(args.target_threshold_pct),
        "target_passed": bool(target_passed),
        "checked_test_metrics": list(checked),
        "split_isolation": {
            "strict": True,
            "train_rows": int(train.size),
            "validation_rows": int(val.size),
            "test_rows": int(test.size),
            "checkpoint_selected_on": "validation only",
            "test_usage": "one final evaluation after checkpoint selection",
        },
        "physics_closure": {
            "motor_force": "predicted thrust; velocity is analytically projected so the archived PD law is exact",
            "mechanical_resistance": "predicted thrust - rail_mass * predicted acceleration",
            "rail_mass_kg": float(args.rail_mass),
        },
        "training_elapsed_s": float(time.perf_counter() - started),
        "training_elapsed_s_total": float(
            elapsed_offset + time.perf_counter() - started
        ),
        "resumed_from": (
            str(args.resume_from.resolve()) if args.resume_from is not None else None
        ),
        "resume_epoch": int(start_epoch - 1) if args.resume_from is not None else None,
        "optimizer_state_restored": optimizer_state_restored,
        "scheduler_state_restored": scheduler_state_restored,
        "checkpoint_best": str(checkpoint_path),
        "history_csv": str(args.output_dir / "history.csv"),
        "loss_plot": str(args.output_dir / "loss_history.png"),
        "rmse_plot": str(args.output_dir / "rmse_history.png"),
        "test_predictions": str(args.output_dir / "strict_test_predictions.npz"),
        "test_by_trajectory": str(args.output_dir / "test_metrics_by_trajectory.csv"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        "[phase10-v24-test] "
        f"p={test_metrics['test_p_rel_rmse_pct']:.4f}% "
        f"pdot={test_metrics['test_pdot_rel_rmse_pct']:.4f}% "
        f"U={test_metrics['test_energy_rel_rmse_pct']:.4f}% "
        f"R={test_metrics['test_resistance_rel_rmse_pct']:.4f}% "
        f"F={test_metrics['test_force_rel_rmse_pct']:.4f}% "
        f"target_passed={str(target_passed).lower()}",
        flush=True,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
