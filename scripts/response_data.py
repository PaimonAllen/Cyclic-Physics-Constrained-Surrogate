"""Portable recorded-response data helpers. No structural solver is imported."""

from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if "source_paths" in arrays:
        arrays["source_paths"] = np.asarray(
            [
                (
                    str((ROOT / str(p)).resolve())
                    if not Path(str(p)).is_absolute()
                    else str(p)
                )
                for p in arrays["source_paths"]
            ]
        )
    return arrays


def _attach_static_context_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Attach geometry/material context to each transition.

    The augmented trajectories vary these parameters.  Omitting them makes
    the reduced transition non-unique even when p, pdot, and u are identical.
    """

    if "static_context" in arrays:
        return
    source_paths = np.asarray(arrays["source_paths"]).reshape(-1)
    trajectory_id = np.asarray(arrays["trajectory_id"], dtype=np.int64).reshape(-1)
    context = np.zeros((trajectory_id.size, 12), dtype=np.float64)
    fields = (
        ("geometry", "Rb"),
        ("geometry", "Rd"),
        ("geometry", "Ru"),
        ("geometry", "Hb"),
        ("geometry", "Hd"),
        ("geometry", "Hu"),
        ("geometry", "thickness"),
        ("material", "E"),
        ("material", "nu"),
        ("material", "rho"),
        ("material", "rho_b"),
        ("material", "damping_c"),
    )
    for tid in np.unique(trajectory_id):
        tid_int = int(tid)
        config_path = Path(str(source_paths[tid_int])).with_name("case_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        values = [float(config[group][name]) for group, name in fields]
        context[trajectory_id == tid_int] = np.asarray(values, dtype=np.float64)
    arrays["static_context"] = context
    arrays["static_context_fields"] = np.asarray(
        [f"{group}.{name}" for group, name in fields]
    )


def _safe_std(x: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=np.float64), eps)


def _metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(truth) & np.isfinite(pred)
    if not mask.any():
        return {"rmse": math.nan, "rel_rmse_pct": math.nan, "max_abs": math.nan}
    err = pred[mask] - truth[mask]
    denom = max(float(np.sqrt(np.mean(truth[mask] ** 2))), 1.0e-12)
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "rmse": rmse,
        "rel_rmse_pct": 100.0 * rmse / denom,
        "max_abs": float(np.max(np.abs(err))),
    }


def _as_control_vector(
    value, n_rails: int, default: float | None = None
) -> np.ndarray | None:
    if value is None:
        if default is None:
            return None
        return np.full((int(n_rails),), float(default), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full((int(n_rails),), float(arr[0]), dtype=np.float64)
    if arr.size == 12 and int(n_rails) != 12:
        return arr[: int(n_rails)].astype(np.float64, copy=True)
    if arr.size != int(n_rails):
        raise ValueError(
            f"Expected scalar or {int(n_rails)} control values, got {arr.size}."
        )
    return arr.astype(np.float64, copy=True)


def _direction_weights(pattern: str, peak_deg: float, n_rails: int) -> np.ndarray:
    normalized = str(pattern).strip().lower()
    theta = np.arange(int(n_rails), dtype=np.float64) * (2.0 * np.pi / 12.0)
    if normalized == "uniform":
        values = np.ones(int(n_rails), dtype=np.float64)
    elif normalized in {"peak_opposite_min", "cosine", "cosine_shifted"}:
        peak = np.deg2rad(float(peak_deg))
        values = (np.cos(theta - peak) + 1.0) / 2.0
    elif normalized == "peak_120deg_decay":
        peak = np.deg2rad(120.0)
        values = (np.cos(theta - peak) + 1.0) / 2.0
    else:
        raise ValueError(f"Unknown arctan direction pattern: {pattern!r}")
    return values.reshape(-1)


def _load_case_control_from_source(
    source_path: str | Path, n_rails: int
) -> dict[str, np.ndarray | float]:
    case_config_path = Path(str(source_path)).parent / "case_config.json"
    if not case_config_path.exists():
        raise FileNotFoundError(f"Missing case_config.json next to {source_path}")
    payload = json.loads(case_config_path.read_text(encoding="utf-8"))
    control = dict(payload.get("control", {}))
    sim_time = max(float(dict(payload.get("time", {})).get("sim_time", 1.0)), 1.0e-12)
    u_scale = np.asarray(control.get("arctan_u_scale", 0.0), dtype=np.float64).reshape(
        -1
    )
    if u_scale.size == 1:
        u_scale = float(u_scale[0]) * _direction_weights(
            str(control.get("direction_pattern", "uniform")),
            float(control.get("peak_deg", 0.0)),
            int(n_rails),
        )
    else:
        u_scale = _as_control_vector(u_scale, int(n_rails), default=0.0)
        assert u_scale is not None
    u_min = _as_control_vector(control.get("u_min"), int(n_rails), default=-1.0e30)
    u_max = _as_control_vector(control.get("u_max"), int(n_rails), default=1.0e30)
    return {
        "sim_time": float(sim_time),
        "arctan_u_scale": np.asarray(u_scale, dtype=np.float64).reshape(-1),
        "arctan_k": float(control.get("arctan_k", 0.0)),
        "pid_kp": float(control.get("pid_kp", 0.0)),
        "pid_ki": float(control.get("pid_ki", 0.0)),
        "pid_kd": float(control.get("pid_kd", 0.0)),
        "pid_error_sign": float(control.get("pid_error_sign", 1.0)),
        "u_min": np.asarray(u_min, dtype=np.float64).reshape(-1),
        "u_max": np.asarray(u_max, dtype=np.float64).reshape(-1),
    }


def _attach_pid_control_arrays(arrays: dict[str, np.ndarray]) -> None:
    if all(
        k in arrays
        for k in (
            "pid_kp",
            "pid_kd",
            "pid_ki",
            "pid_error_sign",
            "arctan_k",
            "arctan_u_scale",
            "sim_time",
            "u_min",
            "u_max",
        )
    ):
        return
    if (
        "source_paths" not in arrays
        or "trajectory_id" not in arrays
        or "p" not in arrays
    ):
        return
    n_rails = int(np.asarray(arrays["p"]).shape[1])
    source_paths = np.asarray(arrays["source_paths"]).reshape(-1)
    trajectory_id = np.asarray(arrays["trajectory_id"], dtype=np.int64).reshape(-1)
    n = trajectory_id.size
    scalar_defaults = {
        "pid_kp": np.zeros((n, 1), dtype=np.float64),
        "pid_ki": np.zeros((n, 1), dtype=np.float64),
        "pid_kd": np.zeros((n, 1), dtype=np.float64),
        "pid_error_sign": np.ones((n, 1), dtype=np.float64),
        "arctan_k": np.zeros((n, 1), dtype=np.float64),
        "sim_time": np.ones((n, 1), dtype=np.float64),
    }
    vector_defaults = {
        "arctan_u_scale": np.zeros((n, n_rails), dtype=np.float64),
        "u_min": np.full((n, n_rails), -1.0e30, dtype=np.float64),
        "u_max": np.full((n, n_rails), 1.0e30, dtype=np.float64),
    }
    cache: dict[int, dict[str, np.ndarray | float]] = {}
    for tid in np.unique(trajectory_id):
        tid_int = int(tid)
        if tid_int < 0 or tid_int >= source_paths.size:
            continue
        cache[tid_int] = _load_case_control_from_source(
            str(source_paths[tid_int]), n_rails
        )
    for tid, cfg in cache.items():
        mask = trajectory_id == int(tid)
        for key in scalar_defaults:
            scalar_defaults[key][mask, 0] = float(cfg[key])
        for key in vector_defaults:
            vector_defaults[key][mask, :] = np.asarray(
                cfg[key], dtype=np.float64
            ).reshape(1, -1)
    arrays.update(scalar_defaults)
    arrays.update(vector_defaults)


def _terminal_reference(arrays: dict[str, np.ndarray]) -> np.ndarray:
    trajectory_id = np.asarray(arrays["trajectory_id"], dtype=np.int64).reshape(-1)
    step_index = np.asarray(arrays["step_index"], dtype=np.int64).reshape(-1)
    terminal = np.zeros_like(np.asarray(arrays["p_ref"], dtype=np.float64))
    for tid in np.unique(trajectory_id):
        rows = np.flatnonzero(trajectory_id == int(tid))
        last = rows[np.argmax(step_index[rows])]
        terminal[rows] = np.asarray(arrays["p_ref"][last], dtype=np.float64)
    return terminal


def _json_scalar(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in data.files:
        return {}
    value = data[key]
    text = str(value.item() if value.shape == () else value.reshape(-1)[0])
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _text_scalar(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    return str(value.item() if value.shape == () else value.reshape(-1)[0])


def _load_audit_trajectory(
    path: Path, trajectory_id: int
) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as data:
        required = (
            "P_actual",
            "P_dot_actual",
            "U_control",
            "P_ref",
            "P_dot_ref",
            "T",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing {missing}.")
        p_all = np.asarray(data["P_actual"], dtype=np.float64)
        v_all = np.asarray(data["P_dot_actual"], dtype=np.float64)
        u_all = np.asarray(data["U_control"], dtype=np.float64)
        pref_all = np.asarray(data["P_ref"], dtype=np.float64)
        vref_all = np.asarray(data["P_dot_ref"], dtype=np.float64)
        time_all = np.asarray(data["T"], dtype=np.float64).reshape(-1)
        if "U" in data.files:
            energy_all = np.asarray(data["U"], dtype=np.float64).reshape(-1)
        elif "U_vec" in data.files:
            energy_all = np.asarray(data["U_vec"], dtype=np.float64).reshape(-1)
        else:
            raise KeyError(f"{path} has no U/U_vec strain-energy history.")
        config = _json_scalar(data, "case_config_json")
        mode = _text_scalar(
            data, "simulation_mode", str(config.get("simulation_mode", ""))
        )

    n_steps = min(
        p_all.shape[0],
        v_all.shape[0],
        u_all.shape[0],
        pref_all.shape[0],
        vref_all.shape[0],
        time_all.size,
        energy_all.size,
    )
    if n_steps < 3 or p_all.shape[1] != 12:
        raise ValueError(f"{path} is not a complete 12-rail trajectory.")
    p_all, v_all, u_all = p_all[:n_steps], v_all[:n_steps], u_all[:n_steps]
    pref_all, vref_all = pref_all[:n_steps], vref_all[:n_steps]
    time_all, energy_all = time_all[:n_steps], energy_all[:n_steps]
    rows = np.arange(n_steps - 1, dtype=np.int32)
    dt = time_all[1:] - time_all[:-1]
    valid = np.isfinite(dt) & (dt > 0.0)
    rows, dt = rows[valid], dt[valid]
    p = p_all[rows]
    velocity = v_all[rows]
    control = u_all[rows]
    p_ref = pref_all[rows]
    v_ref = vref_all[rows]
    next_state = np.hstack((p_all[rows + 1], v_all[rows + 1]))
    acceleration = (v_all[rows + 1] - velocity) / dt.reshape(-1, 1)
    time_value = time_all[rows].reshape(-1, 1)
    time_next = time_all[rows + 1].reshape(-1, 1)
    tau = time_value / max(float(time_all[n_steps - 1]), 1.0e-12)
    features = np.hstack(
        (
            p,
            velocity,
            control,
            p_ref,
            v_ref,
            p_ref - p,
            v_ref - velocity,
            dt.reshape(-1, 1),
            tau,
        )
    )
    case_name = str(config.get("case_name", path.parent.parent.name))
    return {
        "X": features,
        "p": p,
        "pdot": velocity,
        "u": control,
        "p_ref": p_ref,
        "pdot_ref": v_ref,
        "next_state": next_state,
        "dt": dt.reshape(-1, 1),
        "time": time_value,
        "time_next": time_next,
        "U_current": energy_all[rows].reshape(-1, 1),
        "U_next": energy_all[rows + 1].reshape(-1, 1),
        "accel": acceleration,
        "split": np.full(rows.size, 2, dtype=np.int8),
        "source_row": np.full(rows.size, -1, dtype=np.int64),
        "trajectory_id": np.full(rows.size, trajectory_id, dtype=np.int32),
        "step_index": rows,
        "source_path": str(path.resolve()),
        "source_case": case_name,
        "simulation_mode": mode,
    }
