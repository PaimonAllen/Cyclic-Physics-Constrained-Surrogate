#!/usr/bin/env python3
"""Train the full Phase-10/V25 operator or one controlled ablation.

The development data, optimizer, loss weights, evaluation cadence,
and validation-only checkpoint rule match the final V25 run.  Exactly one
architectural or physics component is removed in each named ablation.  The
formal Phase-5 audit trajectories and legacy-controller OOD trajectories are
evaluated separately after the validation-selected checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import secrets
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from response_data import _attach_static_context_arrays, _load_dataset, _metrics
from response_data import _attach_pid_control_arrays
from train_nominal import (
    N_RAILS,
    CircularResidualBlock,
    _analytic_force_numpy,
    _next_aligned,
    _build_features,
    _evaluate,
    _normalization,
    _plot_history,
    _write_csv,
)


from pinn_backbones import LiteraturePINN  # noqa: E402

SOFT_PINN_VARIANTS = ("soft_pinn", "soft_pinn_tanh", "soft_pinn_fourier")
VARIANTS = (
    "full",
    "no_cyclic_coupling",
    "no_invariant_energy_pool",
    "no_pd_projection",
    "no_force_balance",
    "no_hard_closures",
    *SOFT_PINN_VARIANTS,
)


class RailwiseResidualBlock(nn.Module):
    """Shared pointwise residual block with no communication between rails."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(3 * hidden_dim)
        # The 3x expansion matches the leading-order parameter count of two
        # width-preserving kernel-3 convolutions while removing rail coupling.
        self.linear1 = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.linear2 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.linear1(torch.nn.functional.silu(self.norm1(value)))
        value = self.dropout(value)
        value = self.linear2(torch.nn.functional.silu(self.norm2(value)))
        return residual + value


class OrderedRailPooling(nn.Module):
    """Parameter-matched ordered pooling that intentionally breaks cyclic invariance."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(N_RAILS, 3, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # (batch, rail, hidden) -> (batch, 3 * hidden)
        projected = self.projection(value.transpose(1, 2)).transpose(1, 2)
        return projected.reshape(projected.shape[0], -1)


class AblatedPhysicsResponseOperator(nn.Module):
    """V25 architecture with explicit switches for controlled ablations."""

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        variant: str,
        projection_mode: str = "legacy_velocity",
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported ablation variant: {variant}")
        if projection_mode not in {"legacy_velocity", "joint_scaled"}:
            raise ValueError(f"Unsupported projection mode: {projection_mode}")
        self.variant = variant
        self.projection_mode = projection_mode
        self.hard_pd_projection = variant not in {
            "no_pd_projection",
            "no_hard_closures",
            *SOFT_PINN_VARIANTS,
        }
        self.hard_force_balance = variant not in {
            "no_force_balance",
            "no_hard_closures",
            *SOFT_PINN_VARIANTS,
        }
        if variant in {"soft_pinn_tanh", "soft_pinn_fourier"}:
            self.pinn = LiteraturePINN(
                N_RAILS * local_dim + global_dim,
                hidden_dim,
                num_blocks,
                5 * N_RAILS + 1,
                N_RAILS * local_dim + 3,
                use_fourier=variant == "soft_pinn_fourier",
            )
            return
        if variant == "soft_pinn":
            layers: list[nn.Module] = []
            in_features = N_RAILS * local_dim + global_dim
            for _ in range(num_blocks):
                layers.extend(
                    (nn.Linear(in_features, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
                )
                in_features = hidden_dim
            layers.append(nn.Linear(in_features, 5 * N_RAILS + 1))
            self.pinn = nn.Sequential(*layers)
            return
        block_type: type[nn.Module]
        block_type = (
            RailwiseResidualBlock
            if variant == "no_cyclic_coupling"
            else CircularResidualBlock
        )

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
            block_type(hidden_dim, dropout) for _ in range(num_blocks)
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
            block_type(hidden_dim, dropout) for _ in range(max(2, num_blocks // 2))
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
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.resistance_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
            )
            if projection_mode == "joint_scaled" or not self.hard_force_balance
            else None
        )
        self.energy_state_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.ordered_energy_pool = (
            OrderedRailPooling() if variant == "no_invariant_energy_pool" else None
        )
        self.energy_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, local_features: torch.Tensor, global_features: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        if self.variant in SOFT_PINN_VARIANTS:
            response = self.pinn(
                torch.cat((local_features.flatten(1), global_features), dim=1)
            )
            displacement, velocity, acceleration, thrust, resistance = (
                response[:, :-1].reshape(-1, 5, N_RAILS).unbind(dim=1)
            )
            return {
                "displacement": displacement,
                "velocity_prior": velocity,
                "acceleration": acceleration,
                "force_residual": thrust,
                "resistance": resistance,
                "energy": response[:, -1:],
            }
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
        if self.ordered_energy_pool is None:
            rail_pool = torch.cat(
                (
                    energy_node.mean(dim=1),
                    energy_node.amax(dim=1),
                    torch.sqrt(torch.mean(energy_node**2, dim=1) + 1.0e-12),
                ),
                dim=1,
            )
        else:
            rail_pool = self.ordered_energy_pool(energy_node)
        energy = self.energy_head(torch.cat((rail_pool, energy_global), dim=1))
        resistance = (
            self.resistance_head(node).squeeze(-1) if self.resistance_head else None
        )
        return {
            "displacement": displacement,
            "velocity_prior": velocity_prior,
            "acceleration": acceleration,
            "energy": energy,
            "force_residual": force_residual,
            "resistance": resistance,
        }


def _relative_residual_pct(residual: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(np.asarray(reference) ** 2))), 1.0e-12)
    return 100.0 * float(np.sqrt(np.mean(np.asarray(residual) ** 2))) / denominator


def _project_constraints_torch(
    displacement: torch.Tensor,
    velocity: torch.Tensor,
    force_residual: torch.Tensor,
    acceleration: torch.Tensor,
    resistance: torch.Tensor,
    feedforward: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    sign: torch.Tensor,
    u_min: torch.Tensor,
    u_max: torch.Tensor,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rail_mass: float,
    hard_pd: bool,
    hard_balance: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted orthogonal projection onto the PD and rail-balance manifolds."""

    p_scale, v_scale, u_scale, a_scale, r_scale = scales
    p_var = p_scale.square()
    v_var = v_scale.square()
    u_var = u_scale.square()
    a_var = a_scale.square()
    r_var = r_scale.square()
    pd_p = sign * kp
    pd_v = sign * kd
    mass = float(rail_mass)

    if hard_pd and hard_balance:
        residual_pd = pd_p * displacement + pd_v * velocity + force_residual
        residual_balance = (
            -force_residual + mass * acceleration + resistance - feedforward
        )
        gram_pd = pd_p.square() * p_var + pd_v.square() * v_var + u_var
        gram_cross = -u_var
        gram_balance = u_var + mass * mass * a_var + r_var
        determinant = torch.clamp(
            gram_pd * gram_balance - gram_cross.square(), min=1.0e-20
        )
        lambda_pd = (
            residual_pd * gram_balance - residual_balance * gram_cross
        ) / determinant
        lambda_balance = (
            gram_pd * residual_balance - gram_cross * residual_pd
        ) / determinant
        displacement = displacement - p_var * pd_p * lambda_pd
        velocity = velocity - v_var * pd_v * lambda_pd
        force_residual = force_residual - u_var * (lambda_pd - lambda_balance)
        acceleration = acceleration - a_var * mass * lambda_balance
        resistance = resistance - r_var * lambda_balance
    elif hard_pd:
        residual_pd = pd_p * displacement + pd_v * velocity + force_residual
        gram_pd = torch.clamp(
            pd_p.square() * p_var + pd_v.square() * v_var + u_var, min=1.0e-20
        )
        lambda_pd = residual_pd / gram_pd
        displacement = displacement - p_var * pd_p * lambda_pd
        velocity = velocity - v_var * pd_v * lambda_pd
        force_residual = force_residual - u_var * lambda_pd
    elif hard_balance:
        residual_balance = (
            -force_residual + mass * acceleration + resistance - feedforward
        )
        gram_balance = torch.clamp(u_var + mass * mass * a_var + r_var, min=1.0e-20)
        lambda_balance = residual_balance / gram_balance
        force_residual = force_residual + u_var * lambda_balance
        acceleration = acceleration - a_var * mass * lambda_balance
        resistance = resistance - r_var * lambda_balance

    force = feedforward + force_residual
    bounded_force = torch.clamp(force, min=u_min, max=u_max)
    saturated = bounded_force != force
    if torch.any(saturated):
        bounded_residual = bounded_force - feedforward
        if hard_pd:
            residual_pd = pd_p * displacement + pd_v * velocity + bounded_residual
            gram_pd = torch.clamp(
                pd_p.square() * p_var + pd_v.square() * v_var, min=1.0e-20
            )
            lambda_pd = residual_pd / gram_pd
            displacement = torch.where(
                saturated, displacement - p_var * pd_p * lambda_pd, displacement
            )
            velocity = torch.where(
                saturated, velocity - v_var * pd_v * lambda_pd, velocity
            )
        if hard_balance:
            residual_balance = mass * acceleration + resistance - bounded_force
            gram_balance = torch.clamp(mass * mass * a_var + r_var, min=1.0e-20)
            lambda_balance = residual_balance / gram_balance
            acceleration = torch.where(
                saturated, acceleration - a_var * mass * lambda_balance, acceleration
            )
            resistance = torch.where(
                saturated, resistance - r_var * lambda_balance, resistance
            )
        force_residual = torch.where(saturated, bounded_residual, force_residual)
    return displacement, velocity, force_residual, acceleration, resistance


def _project_constraints_numpy(
    displacement: np.ndarray,
    velocity: np.ndarray,
    force_residual: np.ndarray,
    acceleration: np.ndarray,
    resistance: np.ndarray,
    feedforward: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    sign: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    scales: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rail_mass: float,
    hard_pd: bool,
    hard_balance: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy equivalent of :func:`_project_constraints_torch`."""

    p_scale, v_scale, u_scale, a_scale, r_scale = scales
    p_var, v_var, u_var = p_scale**2, v_scale**2, u_scale**2
    a_var, r_var = a_scale**2, r_scale**2
    pd_p = sign * kp
    pd_v = sign * kd
    mass = float(rail_mass)
    if hard_pd and hard_balance:
        residual_pd = pd_p * displacement + pd_v * velocity + force_residual
        residual_balance = (
            -force_residual + mass * acceleration + resistance - feedforward
        )
        gram_pd = pd_p**2 * p_var + pd_v**2 * v_var + u_var
        gram_cross = -u_var
        gram_balance = u_var + mass**2 * a_var + r_var
        determinant = np.maximum(gram_pd * gram_balance - gram_cross**2, 1.0e-20)
        lambda_pd = (
            residual_pd * gram_balance - residual_balance * gram_cross
        ) / determinant
        lambda_balance = (
            gram_pd * residual_balance - gram_cross * residual_pd
        ) / determinant
        displacement = displacement - p_var * pd_p * lambda_pd
        velocity = velocity - v_var * pd_v * lambda_pd
        force_residual = force_residual - u_var * (lambda_pd - lambda_balance)
        acceleration = acceleration - a_var * mass * lambda_balance
        resistance = resistance - r_var * lambda_balance
    elif hard_pd:
        residual_pd = pd_p * displacement + pd_v * velocity + force_residual
        gram_pd = np.maximum(pd_p**2 * p_var + pd_v**2 * v_var + u_var, 1.0e-20)
        lambda_pd = residual_pd / gram_pd
        displacement = displacement - p_var * pd_p * lambda_pd
        velocity = velocity - v_var * pd_v * lambda_pd
        force_residual = force_residual - u_var * lambda_pd
    elif hard_balance:
        residual_balance = (
            -force_residual + mass * acceleration + resistance - feedforward
        )
        gram_balance = np.maximum(u_var + mass**2 * a_var + r_var, 1.0e-20)
        lambda_balance = residual_balance / gram_balance
        force_residual = force_residual + u_var * lambda_balance
        acceleration = acceleration - a_var * mass * lambda_balance
        resistance = resistance - r_var * lambda_balance

    force = feedforward + force_residual
    bounded_force = np.clip(force, u_min, u_max)
    saturated = bounded_force != force
    if np.any(saturated):
        bounded_residual = bounded_force - feedforward
        if hard_pd:
            residual_pd = pd_p * displacement + pd_v * velocity + bounded_residual
            gram_pd = np.maximum(pd_p**2 * p_var + pd_v**2 * v_var, 1.0e-20)
            lambda_pd = residual_pd / gram_pd
            displacement = np.where(
                saturated, displacement - p_var * pd_p * lambda_pd, displacement
            )
            velocity = np.where(
                saturated, velocity - v_var * pd_v * lambda_pd, velocity
            )
        if hard_balance:
            residual_balance = mass * acceleration + resistance - bounded_force
            gram_balance = np.maximum(mass**2 * a_var + r_var, 1.0e-20)
            lambda_balance = residual_balance / gram_balance
            acceleration = np.where(
                saturated, acceleration - a_var * mass * lambda_balance, acceleration
            )
            resistance = np.where(
                saturated, resistance - r_var * lambda_balance, resistance
            )
        force_residual = np.where(saturated, bounded_residual, force_residual)
    return displacement, velocity, force_residual, acceleration, resistance


def _predict(
    model: AblatedPhysicsResponseOperator,
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
    chunks: dict[str, list[np.ndarray]] = {
        "displacement": [],
        "velocity_prior": [],
        "acceleration": [],
        "energy": [],
        "force_residual": [],
        "resistance": [],
    }
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            rows = indices[start : start + batch_size]
            output = model(
                torch.as_tensor(local_norm[rows], dtype=torch.float32, device=device),
                torch.as_tensor(global_norm[rows], dtype=torch.float32, device=device),
            )
            for key, value in output.items():
                if value is not None:
                    chunks[key].append(value.cpu().numpy())

    displacement_residual = (
        np.concatenate(chunks["displacement"]) * norm["state_std"][:, :, 0]
        + norm["state_mean"][:, :, 0]
    )
    velocity_prior = (
        np.concatenate(chunks["velocity_prior"]) * norm["state_std"][:, :, 1]
        + norm["state_mean"][:, :, 1]
    )
    acceleration = (
        np.concatenate(chunks["acceleration"]) * norm["accel_std"] + norm["accel_mean"]
    )
    energy = np.concatenate(chunks["energy"]) * norm["energy_std"] + norm["energy_mean"]
    force_residual = (
        np.concatenate(chunks["force_residual"]) * norm["force_residual_std"]
        + norm["force_residual_mean"]
    )
    if norm.get("force_gain") == "pid_kp":
        force_residual = force_residual * np.asarray(
            arrays["pid_kp"][indices], dtype=np.float64
        )
    time_value = np.asarray(arrays["time"][indices], dtype=np.float64).reshape(-1, 1)
    sim_time = np.asarray(arrays["sim_time"][indices], dtype=np.float64).reshape(-1, 1)
    arctan_k = np.asarray(arrays["arctan_k"][indices], dtype=np.float64).reshape(-1, 1)
    ramp = (2.0 / np.pi) * np.arctan(
        arctan_k * time_value / np.maximum(sim_time, 1.0e-12)
    )
    feedforward = np.asarray(arrays["arctan_u_scale"][indices], dtype=np.float64) * ramp
    if model.projection_mode == "joint_scaled":
        resistance = (
            np.concatenate(chunks["resistance"]) * norm["resistance_std"]
            + norm["resistance_mean"]
        )
        kp = np.asarray(arrays["pid_kp"][indices], dtype=np.float64)
        kd = np.asarray(arrays["pid_kd"][indices], dtype=np.float64)
        sign = np.asarray(arrays["pid_error_sign"][indices], dtype=np.float64)
        (
            displacement_residual,
            velocity_residual,
            force_residual,
            acceleration,
            resistance,
        ) = _project_constraints_numpy(
            displacement_residual,
            velocity_prior,
            force_residual,
            acceleration,
            resistance,
            feedforward,
            kp,
            kd,
            sign,
            np.asarray(arrays["u_min"][indices], dtype=np.float64),
            np.asarray(arrays["u_max"][indices], dtype=np.float64),
            (
                norm["state_std"][:, :, 0],
                norm["state_std"][:, :, 1],
                norm["force_residual_std"],
                norm["accel_std"],
                norm["resistance_std"],
            ),
            rail_mass,
            model.hard_pd_projection,
            model.hard_force_balance,
        )
        force = feedforward + force_residual
    else:
        force = np.clip(
            feedforward + force_residual,
            np.asarray(arrays["u_min"][indices]),
            np.asarray(arrays["u_max"][indices]),
        )
    if model.projection_mode == "legacy_velocity" and model.hard_pd_projection:
        kp = np.asarray(arrays["pid_kp"][indices], dtype=np.float64)
        kd = np.asarray(arrays["pid_kd"][indices], dtype=np.float64)
        sign = np.asarray(arrays["pid_error_sign"][indices], dtype=np.float64)
        velocity_residual = (
            sign * (feedforward - force) - kp * displacement_residual
        ) / np.maximum(np.abs(kd), 1.0e-12)
    elif model.projection_mode == "legacy_velocity":
        velocity_residual = velocity_prior
    p = np.asarray(arrays["p_ref"][indices], dtype=np.float64) + displacement_residual
    velocity = (
        np.asarray(arrays["pdot_ref"][indices], dtype=np.float64) + velocity_residual
    )
    if model.projection_mode == "legacy_velocity" and model.hard_force_balance:
        resistance = force - float(rail_mass) * acceleration
    elif model.projection_mode == "legacy_velocity":
        resistance = (
            np.concatenate(chunks["resistance"]) * norm["resistance_std"]
            + norm["resistance_mean"]
        )
    return {
        "p": p,
        "pdot": velocity,
        "accel": acceleration,
        "energy": energy,
        "force": force,
        "resistance": resistance,
    }


def _evaluate_with_closures(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    prefix: str,
    rail_mass: float,
) -> dict[str, float]:
    result = _evaluate(predictions, arrays, indices, prefix)
    pd_force = _analytic_force_numpy(
        arrays, indices, predictions["p"], predictions["pdot"]
    )
    balance_resistance = predictions["force"] - float(rail_mass) * predictions["accel"]
    result[f"{prefix}_pd_closure_rel_rmse_pct"] = _relative_residual_pct(
        predictions["force"] - pd_force, arrays["u"][indices]
    )
    result[f"{prefix}_balance_closure_rel_rmse_pct"] = _relative_residual_pct(
        predictions["resistance"] - balance_resistance,
        arrays["mechanical_resistance"][indices],
    )
    return result


def _test_groups(
    arrays: dict[str, np.ndarray], test: np.ndarray
) -> dict[str, np.ndarray]:
    trajectory_ids = np.asarray(arrays["trajectory_id"], dtype=np.int64).reshape(-1)
    source_paths = np.asarray(arrays["source_paths"], dtype=str)
    # Revision datasets explicitly seal entire cyclic-equivalence groups.
    # The legacy archive keeps its original Phase-5/OOD classification.
    formal_ids = (
        set(map(int, np.asarray(arrays["formal_test_ids"]).reshape(-1)))
        if "formal_test_ids" in arrays
        else {
            int(tid)
            for tid in np.unique(trajectory_ids[test])
            if "/phase5_full_nozzle_cases/" in source_paths[int(tid)]
        }
    )
    formal = test[np.isin(trajectory_ids[test], sorted(formal_ids))]
    ood = test[~np.isin(trajectory_ids[test], sorted(formal_ids))]
    if formal.size == 0 or ood.size == 0:
        raise ValueError(
            "Could not separate formal Phase-5 and historical OOD test trajectories."
        )
    return {"formal_test": formal, "ood_test": ood, "all_test": test}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_DIR / "data/processed/grouped.npz"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=3000)
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
    parser.add_argument(
        "--projection-mode",
        choices=("legacy_velocity", "joint_scaled"),
        default="legacy_velocity",
        help="Hard-closure reconstruction used by the full and structural variants.",
    )
    parser.add_argument(
        "--raw-aux-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight on pre-projection mechanical predictions.",
    )
    parser.add_argument("--train-split-id", type=int, default=0)
    parser.add_argument("--val-split-id", type=int, default=1)
    parser.add_argument("--test-split-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for a new training run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--gain-collocation-weight", type=float, default=0.0)
    parser.add_argument(
        "--pinn-kinematics-weight",
        type=float,
        default=0.0,
        help="Weight of each normalized interval kinematic residual for the soft PINN.",
    )
    parser.add_argument(
        "--force-gain-normalization", choices=("none", "pid_kp"), default="none"
    )
    parser.add_argument("--initialize-correlated-heads", action="store_true")
    parser.add_argument(
        "--selection-include-velocity",
        action="store_true",
        help="Include final velocity RMSE in validation checkpoint selection for every variant.",
    )
    return parser


def _device_training_batches(loader: DataLoader, device: torch.device):
    """Reuse GPU-resident training rows with the loader's shuffled batch order."""
    tensors = loader.dataset.tensors
    shapes = [tensor.shape[1:] for tensor in tensors]
    widths = [math.prod(shape) for shape in shapes]
    packed = torch.cat([tensor.flatten(1) for tensor in tensors], dim=1).to(device)

    def batches():
        # DataLoader normally consumes this base seed before sampling its batches.
        torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
        for indices in loader.batch_sampler:
            rows = torch.as_tensor(indices, dtype=torch.long, device=device)
            columns = packed.index_select(0, rows).split(widths, dim=1)
            yield tuple(
                column.reshape(len(indices), *shape)
                for column, shape in zip(columns, shapes)
            )

    return batches


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(32)
    use_pinn_kinematics = args.pinn_kinematics_weight > 0
    if (
        not math.isfinite(args.pinn_kinematics_weight)
        or args.pinn_kinematics_weight < 0
    ):
        raise ValueError(
            "The soft PINN kinematic weight must be finite and nonnegative."
        )
    if use_pinn_kinematics and args.variant not in SOFT_PINN_VARIANTS:
        raise ValueError("The kinematic penalty is available only for the soft PINN.")
    if args.variant in SOFT_PINN_VARIANTS and (
        args.projection_mode != "legacy_velocity"
        or args.initialize_correlated_heads
        or args.gain_collocation_weight != 0
    ):
        raise ValueError(
            "The soft PINN requires the legacy output interface without gain collocation."
        )
    if (
        args.force_gain_normalization != "none"
        and args.projection_mode != "legacy_velocity"
    ):
        raise ValueError("Force gain normalization uses the original projection.")
    if args.initialize_correlated_heads and args.force_gain_normalization != "pid_kp":
        raise ValueError(
            "Correlated initialization requires proportional gain normalization."
        )
    if int(args.eval_every) <= 0:
        raise ValueError("--eval-every must be positive.")
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite completed ablation output: {args.output_dir}"
        )
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
    split = np.asarray(arrays["split"], dtype=np.int8).reshape(-1)
    train = np.flatnonzero(split == int(args.train_split_id)).astype(np.int64)
    val = np.flatnonzero(split == int(args.val_split_id)).astype(np.int64)
    test = np.flatnonzero(split == int(args.test_split_id)).astype(np.int64)
    if min(train.size, val.size, test.size) == 0:
        raise ValueError(
            f"Empty split: train={train.size}, val={val.size}, test={test.size}."
        )

    local, global_features, local_layout, global_layout = _build_features(arrays)
    if args.variant == "soft_pinn_fourier" and global_layout[3] != "tau":
        raise ValueError(
            "The Fourier PINN requires normalized time in global feature column 3."
        )
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
    sim_time = np.asarray(arrays["sim_time"], dtype=np.float64).reshape(-1, 1)
    arctan_k = np.asarray(arrays["arctan_k"], dtype=np.float64).reshape(-1, 1)
    ramp = (2.0 / np.pi) * np.arctan(
        arctan_k * time_value / np.maximum(sim_time, 1.0e-12)
    )
    feedforward = np.asarray(arrays["arctan_u_scale"], dtype=np.float64) * ramp
    force_residual = np.asarray(arrays["u"], dtype=np.float64) - feedforward
    if args.force_gain_normalization == "pid_kp":
        if np.any(arrays["pid_kp"] <= 0):
            raise ValueError("Proportional gain normalization requires positive gains.")
        force_residual = force_residual / arrays["pid_kp"]
    resistance = np.asarray(arrays["mechanical_resistance"], dtype=np.float64)
    norm = _normalization(
        local,
        global_features,
        state_residual,
        acceleration,
        energy,
        force_residual,
        train,
    )
    if args.force_gain_normalization == "pid_kp":
        norm["force_gain"] = "pid_kp"
    norm["resistance_mean"] = resistance[train].mean(axis=0, keepdims=True)
    norm["resistance_std"] = np.maximum(
        resistance[train].std(axis=0, keepdims=True), 1.0e-12
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
    resistance_norm = (
        (resistance - norm["resistance_mean"]) / norm["resistance_std"]
    ).astype(np.float32)
    force_scale = max(float(np.sqrt(np.mean(np.asarray(arrays["u"])[train] ** 2))), 1.0)
    resistance_scale = max(float(np.sqrt(np.mean(resistance[train] ** 2))), 1.0)

    dataset = TensorDataset(
        torch.as_tensor(local_norm[train]),
        torch.as_tensor(global_norm[train]),
        torch.as_tensor(state_norm[train]),
        torch.as_tensor(accel_norm[train]),
        torch.as_tensor(energy_norm[train]),
        torch.as_tensor(force_residual_norm[train]),
        torch.as_tensor(resistance_norm[train]),
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
        torch.as_tensor(resistance[train], dtype=torch.float32),
    )
    if use_pinn_kinematics:
        next_row = _next_aligned(
            np.arange(split.size), arrays["trajectory_id"], arrays["step_index"]
        ).astype(np.int64)[train]
        query_times = np.asarray(arrays["time"]).reshape(-1)
        step = (query_times[next_row] - query_times[train])[:, None]
        valid = next_row != train
        if (
            not np.any(valid)
            or np.any(split[next_row] != args.train_split_id)
            or np.any(step[valid] <= 0)
        ):
            raise ValueError(
                "Kinematic pairs must be consecutive positive intervals in the training split."
            )
        reference_p = np.asarray(arrays["p_ref"])
        reference_v = np.asarray(arrays["pdot_ref"])
        velocity = np.asarray(arrays["pdot"])
        p_increment = (
            step[valid] * 0.5 * (velocity[train[valid]] + velocity[next_row[valid]])
        )
        v_increment = step[valid] * acceleration[train[valid]]
        displacement_increment_scale = max(
            float(np.sqrt(np.mean(p_increment**2))), 1.0e-12
        )
        velocity_increment_scale = max(float(np.sqrt(np.mean(v_increment**2))), 1.0e-12)
        paired = (
            local_norm[next_row],
            global_norm[next_row],
            reference_p[next_row],
            reference_v[next_row],
            step,
            valid[:, None],
        )
        dataset = TensorDataset(
            *dataset.tensors,
            *(torch.as_tensor(value, dtype=torch.float32) for value in paired),
        )
        print(
            f"[soft-pinn-kinematics] training intervals={int(valid.sum())} "
            f"displacement scale={displacement_increment_scale:.6g} m "
            f"velocity scale={velocity_increment_scale:.6g} m/s",
            flush=True,
        )
    loader_generator = torch.Generator().manual_seed(int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )
    training_batches = (
        _device_training_batches(loader, device)
        if args.variant in SOFT_PINN_VARIANTS and device.type == "cuda"
        else lambda: iter(loader)
    )
    model = AblatedPhysicsResponseOperator(
        local_dim=local.shape[2],
        global_dim=global_features.shape[1],
        hidden_dim=int(args.hidden_dim),
        num_blocks=int(args.num_blocks),
        dropout=float(args.dropout),
        variant=args.variant,
        projection_mode=args.projection_mode,
    ).to(device)
    if args.warm_start is not None:
        initial = torch.load(args.warm_start, map_location=device, weights_only=False)
        for key in (
            "local_mean",
            "local_std",
            "global_mean",
            "global_std",
            "state_mean",
            "state_std",
        ):
            if not np.allclose(
                initial["normalization"][key], norm[key], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(f"Warm-start normalization differs for {key}.")
        model.load_state_dict(initial["state_dict"], strict=True)
    if args.initialize_correlated_heads:
        if not np.all(arrays["pid_error_sign"][train] == 1):
            raise ValueError(
                "Correlated initialization requires positive control error sign."
            )
        model.force_residual_head.load_state_dict(model.displacement_head.state_dict())
        p_std = float(norm["state_std"][0, 0, 0])
        p_mean = float(norm["state_mean"][0, 0, 0])
        f_std = float(norm["force_residual_std"][0, 0])
        f_mean = float(norm["force_residual_mean"][0, 0])
        with torch.no_grad():
            model.force_residual_head[-1].weight.mul_(-p_std / f_std)
            model.force_residual_head[-1].bias.mul_(-p_std / f_std).add_(
                (-p_mean - f_mean) / f_std
            )
        for head in (model.displacement_head, model.force_residual_head):
            for parameter in head[0].parameters():
                parameter.requires_grad_(False)
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
    force_mean_t = torch.as_tensor(
        norm["force_residual_mean"], dtype=torch.float32, device=device
    )
    force_std_t = torch.as_tensor(
        norm["force_residual_std"], dtype=torch.float32, device=device
    )
    resistance_mean_t = torch.as_tensor(
        norm["resistance_mean"], dtype=torch.float32, device=device
    )
    resistance_std_t = torch.as_tensor(
        norm["resistance_std"], dtype=torch.float32, device=device
    )
    gain_ranges = {
        name: (float(np.min(arrays[name][train])), float(np.max(arrays[name][train])))
        for name in ("pid_kp", "pid_kd")
    }
    if (
        args.gain_collocation_weight > 0
        and min(gain_ranges["pid_kp"][0], gain_ranges["pid_kd"][0]) <= 0
    ):
        raise ValueError("Gain collocation requires positive training gains.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "[phase10-ablation] "
        f"variant={args.variant} rows={split.size} train={train.size} val={val.size} "
        f"test={test.size} parameters={parameter_count} device={device} "
        f"hard_pd={model.hard_pd_projection} hard_balance={model.hard_force_balance} "
        f"projection={model.projection_mode}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    best_val: dict[str, Any] | None = None
    checkpoint_path = args.output_dir / "best_ablation_response_operator.pt"
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in training_batches():
            (
                local_b,
                global_b,
                state_b,
                accel_b,
                energy_b,
                force_residual_b,
                resistance_b,
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
                *paired_b,
            ) = (value.to(device, non_blocking=True) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            if use_pinn_kinematics:
                next_local, next_global, next_p_ref, next_v_ref, step_b, valid_b = (
                    paired_b
                )
                pair_output = model(
                    torch.cat((local_b, next_local)), torch.cat((global_b, next_global))
                )
                batch_rows = local_b.shape[0]
                output = {key: value[:batch_rows] for key, value in pair_output.items()}
                next_displacement = (
                    pair_output["displacement"][batch_rows:] * state_std_t[:, :, 0]
                    + state_mean_t[:, :, 0]
                    + next_p_ref
                )
                next_velocity = (
                    pair_output["velocity_prior"][batch_rows:] * state_std_t[:, :, 1]
                    + state_mean_t[:, :, 1]
                    + next_v_ref
                )
            else:
                output = model(local_b, global_b)
            displacement_phys = (
                output["displacement"] * state_std_t[:, :, 0] + state_mean_t[:, :, 0]
            )
            velocity_prior_phys = (
                output["velocity_prior"] * state_std_t[:, :, 1] + state_mean_t[:, :, 1]
            )
            accel_phys = output["acceleration"] * accel_std_t + accel_mean_t
            force_residual_phys = output["force_residual"] * force_std_t + force_mean_t
            if args.force_gain_normalization == "pid_kp":
                force_residual_phys = force_residual_phys * kp_b
            if model.projection_mode == "joint_scaled":
                resistance_raw = (
                    output["resistance"] * resistance_std_t + resistance_mean_t
                )
                (
                    displacement_final,
                    velocity_final,
                    force_residual_final,
                    accel_final,
                    resistance_final,
                ) = _project_constraints_torch(
                    displacement_phys,
                    velocity_prior_phys,
                    force_residual_phys,
                    accel_phys,
                    resistance_raw,
                    feedforward_b,
                    kp_b,
                    kd_b,
                    sign_b,
                    u_min_b,
                    u_max_b,
                    (
                        state_std_t[:, :, 0],
                        state_std_t[:, :, 1],
                        force_std_t,
                        accel_std_t,
                        resistance_std_t,
                    ),
                    float(args.rail_mass),
                    model.hard_pd_projection,
                    model.hard_force_balance,
                )
                force_pred = feedforward_b + force_residual_final
                displacement_final_norm = (
                    displacement_final - state_mean_t[:, :, 0]
                ) / state_std_t[:, :, 0]
                velocity_final_norm = (
                    velocity_final - state_mean_t[:, :, 1]
                ) / state_std_t[:, :, 1]
                accel_final_norm = (accel_final - accel_mean_t) / accel_std_t
                force_final_norm = (force_residual_final - force_mean_t) / force_std_t
                resistance_final_norm = (
                    resistance_final - resistance_mean_t
                ) / resistance_std_t
                projected_loss = (
                    float(args.p_loss_weight)
                    * torch.mean((displacement_final_norm - state_b[:, :, 0]) ** 2)
                    + float(args.pdot_loss_weight)
                    * torch.mean((velocity_final_norm - state_b[:, :, 1]) ** 2)
                    + float(args.energy_loss_weight)
                    * torch.mean((output["energy"] - energy_b) ** 2)
                    + float(args.accel_loss_weight)
                    * torch.mean((accel_final_norm - accel_b) ** 2)
                    + float(args.force_loss_weight)
                    * torch.mean((force_final_norm - force_residual_b) ** 2)
                    + float(args.force_loss_weight)
                    * torch.mean(((force_pred - force_truth_b) / force_scale) ** 2)
                    + float(args.resistance_loss_weight)
                    * torch.mean((resistance_final_norm - resistance_b) ** 2)
                )
                raw_auxiliary_loss = (
                    float(args.p_loss_weight)
                    * torch.mean((output["displacement"] - state_b[:, :, 0]) ** 2)
                    + float(args.pdot_loss_weight)
                    * torch.mean((output["velocity_prior"] - state_b[:, :, 1]) ** 2)
                    + float(args.accel_loss_weight)
                    * torch.mean((output["acceleration"] - accel_b) ** 2)
                    + float(args.force_loss_weight)
                    * torch.mean((output["force_residual"] - force_residual_b) ** 2)
                    + float(args.resistance_loss_weight)
                    * torch.mean((output["resistance"] - resistance_b) ** 2)
                )
                loss = (
                    projected_loss
                    + float(args.raw_aux_loss_weight) * raw_auxiliary_loss
                )
            else:
                force_pred = torch.clamp(
                    feedforward_b + force_residual_phys, min=u_min_b, max=u_max_b
                )
                if model.hard_pd_projection:
                    velocity_residual = (
                        sign_b * (feedforward_b - force_pred) - kp_b * displacement_phys
                    ) / torch.clamp(torch.abs(kd_b), min=1.0e-12)
                    velocity_loss = 0.5 * (
                        torch.mean((output["velocity_prior"] - state_b[:, :, 1]) ** 2)
                        + torch.mean(
                            (
                                (velocity_residual - state_mean_t[:, :, 1])
                                / state_std_t[:, :, 1]
                                - state_b[:, :, 1]
                            )
                            ** 2
                        )
                    )
                else:
                    velocity_loss = torch.mean(
                        (output["velocity_prior"] - state_b[:, :, 1]) ** 2
                    )
                if model.hard_force_balance:
                    resistance_pred = force_pred - float(args.rail_mass) * accel_phys
                else:
                    resistance_pred = (
                        output["resistance"] * resistance_std_t + resistance_mean_t
                    )
                loss = (
                    float(args.p_loss_weight)
                    * torch.mean((output["displacement"] - state_b[:, :, 0]) ** 2)
                    + float(args.pdot_loss_weight) * velocity_loss
                    + float(args.energy_loss_weight)
                    * torch.mean((output["energy"] - energy_b) ** 2)
                    + float(args.accel_loss_weight)
                    * torch.mean((output["acceleration"] - accel_b) ** 2)
                    + float(args.force_loss_weight)
                    * torch.mean((output["force_residual"] - force_residual_b) ** 2)
                    + float(args.force_loss_weight)
                    * torch.mean(((force_pred - force_truth_b) / force_scale) ** 2)
                    + float(args.resistance_loss_weight)
                    * torch.mean(
                        ((resistance_pred - resistance_truth_b) / resistance_scale) ** 2
                    )
                )
                if args.variant in SOFT_PINN_VARIANTS:
                    pd_force = torch.clamp(
                        feedforward_b
                        - sign_b
                        * (kp_b * displacement_phys + kd_b * velocity_prior_phys),
                        min=u_min_b,
                        max=u_max_b,
                    )
                    pd_residual = (force_pred - pd_force) / force_scale
                    balance_residual = (
                        resistance_pred
                        - force_pred
                        + float(args.rail_mass) * accel_phys
                    ) / resistance_scale
                    loss = (
                        loss
                        + pd_residual.square().mean()
                        + balance_residual.square().mean()
                    )
                    if use_pinn_kinematics:
                        position = p_ref_b + displacement_phys
                        velocity = pdot_ref_b + velocity_prior_phys
                        displacement_residual = (
                            next_displacement
                            - position
                            - step_b * 0.5 * (velocity + next_velocity)
                        ) / displacement_increment_scale
                        velocity_residual = (
                            next_velocity - velocity - step_b * accel_phys
                        ) / velocity_increment_scale
                        interval_loss = (
                            (
                                displacement_residual.square()
                                + velocity_residual.square()
                            )
                            * valid_b
                        ).sum() / (valid_b.sum().clamp_min(1) * N_RAILS)
                        loss = loss + args.pinn_kinematics_weight * interval_loss
                if args.gain_collocation_weight > 0 and model.hard_pd_projection:
                    context_collocation = global_b.clone()
                    gains = {}
                    for name in ("pid_kp", "pid_kd"):
                        lower, upper = gain_ranges[name]
                        gain = torch.exp(
                            math.log(lower)
                            + torch.rand_like(kp_b) * math.log(upper / lower)
                        )
                        gains[name] = gain
                        for feature, value in (
                            (name, gain),
                            ("log10_abs_" + name, torch.log10(gain)),
                        ):
                            column = global_layout.index(feature)
                            context_collocation[:, column] = (
                                value[:, 0] - float(norm["global_mean"][0, column])
                            ) / float(norm["global_std"][0, column])
                    collocation = model(local_b, context_collocation)
                    p_collocation = (
                        collocation["displacement"] * state_std_t[:, :, 0]
                        + state_mean_t[:, :, 0]
                    )
                    v_collocation = (
                        collocation["velocity_prior"] * state_std_t[:, :, 1]
                        + state_mean_t[:, :, 1]
                    )
                    residual_collocation = (
                        collocation["force_residual"] * force_std_t + force_mean_t
                    )
                    if args.force_gain_normalization == "pid_kp":
                        residual_collocation = residual_collocation * gains["pid_kp"]
                    unbounded_force = feedforward_b + residual_collocation
                    f_collocation = torch.clamp(
                        unbounded_force,
                        min=u_min_b,
                        max=u_max_b,
                    )
                    v_projected = (
                        sign_b * (feedforward_b - f_collocation)
                        - gains["pid_kp"] * p_collocation
                    ) / gains["pid_kd"]
                    unsaturated = (unbounded_force > u_min_b) & (
                        unbounded_force < u_max_b
                    )
                    squared_residual = (
                        (v_projected - v_collocation) / state_std_t[:, :, 1]
                    ) ** 2
                    consistency = (
                        squared_residual * unsaturated
                    ).sum() / unsaturated.sum().clamp_min(1)
                    loss = loss + float(args.gain_collocation_weight) * consistency
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
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
                "elapsed_s": float(time.perf_counter() - started),
            }
            row.update(
                _evaluate_with_closures(
                    train_pred, arrays, train, "train", float(args.rail_mass)
                )
            )
            row.update(
                _evaluate_with_closures(
                    val_pred, arrays, val, "val", float(args.rail_mass)
                )
            )
            selection_keys = [
                "val_p_rel_rmse_pct",
                "val_energy_rel_rmse_pct",
                "val_resistance_rel_rmse_pct",
                "val_force_rel_rmse_pct",
            ]
            if (
                args.selection_include_velocity
                or model.projection_mode == "joint_scaled"
            ):
                selection_keys.insert(1, "val_pdot_rel_rmse_pct")
            row["val_score"] = float(sum(float(row[key]) for key in selection_keys))
            history.append(row)
            scheduler.step(float(row["val_score"]))
            if float(row["val_score"]) < best_score:
                best_score = float(row["val_score"])
                best_epoch = int(epoch)
                best_val = dict(row)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_config": {
                            "local_dim": int(local.shape[2]),
                            "global_dim": int(global_features.shape[1]),
                            "hidden_dim": int(args.hidden_dim),
                            "num_blocks": int(args.num_blocks),
                            "dropout": float(args.dropout),
                            "variant": args.variant,
                            "projection_mode": model.projection_mode,
                            "local_layout": local_layout,
                            "global_layout": global_layout,
                            "hard_pd_projection": model.hard_pd_projection,
                            "hard_force_balance": model.hard_force_balance,
                        },
                        "normalization": norm,
                        "rail_mass": float(args.rail_mass),
                        "epoch": int(epoch),
                        "validation_score": best_score,
                    },
                    checkpoint_path,
                )
            _write_csv(args.output_dir / "history.csv", history)
            _plot_history(history, args.output_dir)
            print(
                "[phase10-ablation-eval] "
                f"variant={args.variant} epoch={epoch}/{args.epochs} "
                f"loss={row['train_loss']:.5g} "
                f"val[p={row['val_p_rel_rmse_pct']:.3f}% "
                f"U={row['val_energy_rel_rmse_pct']:.3f}% "
                f"R={row['val_resistance_rel_rmse_pct']:.3f}% "
                f"F={row['val_force_rel_rmse_pct']:.3f}% "
                f"PDres={row['val_pd_closure_rel_rmse_pct']:.3e}% "
                f"FBres={row['val_balance_closure_rel_rmse_pct']:.3e}%] "
                f"score={row['val_score']:.3f} best={best_score:.3f}@{best_epoch}",
                flush=True,
            )

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
    groups = _test_groups(arrays, test)
    test_metrics: dict[str, dict[str, float]] = {}
    for group_name, group_rows in groups.items():
        local_rows = np.flatnonzero(np.isin(test, group_rows))
        group_pred = {key: value[local_rows] for key, value in test_pred.items()}
        test_metrics[group_name] = _evaluate_with_closures(
            group_pred, arrays, group_rows, group_name, float(args.rail_mass)
        )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
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
        "experiment": "phase10_v25_one_factor_ablation",
        "variant": args.variant,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_metrics": best_val,
        "test_metrics": test_metrics,
        "parameter_count": int(parameter_count),
        "hard_pd_projection": bool(model.hard_pd_projection),
        "hard_force_balance": bool(model.hard_force_balance),
        "projection_mode": model.projection_mode,
        "validation_score_components": selection_keys,
        "training_elapsed_s": float(time.perf_counter() - started),
        "split_isolation": {
            "train_rows": int(train.size),
            "validation_rows": int(val.size),
            "test_rows": int(test.size),
            "checkpoint_selected_on": "validation only",
            "test_usage": "one final evaluation after checkpoint selection",
        },
        "checkpoint_best": str(checkpoint_path),
        "history_csv": str(args.output_dir / "history.csv"),
        "loss_plot": str(args.output_dir / "loss_history.png"),
        "rmse_plot": str(args.output_dir / "rmse_history.png"),
        "test_predictions": str(args.output_dir / "test_predictions.npz"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    formal = test_metrics["formal_test"]
    print(
        "[phase10-ablation-test] "
        f"variant={args.variant} p={formal['formal_test_p_rel_rmse_pct']:.4f}% "
        f"U={formal['formal_test_energy_rel_rmse_pct']:.4f}% "
        f"R={formal['formal_test_resistance_rel_rmse_pct']:.4f}% "
        f"F={formal['formal_test_force_rel_rmse_pct']:.4f}% "
        f"PDres={formal['formal_test_pd_closure_rel_rmse_pct']:.4e}% "
        f"FBres={formal['formal_test_balance_closure_rel_rmse_pct']:.4e}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
