"""Rebuild learning labels from released rail histories and frozen row selection."""

import argparse, json
from pathlib import Path
import numpy as np
from response_data import (
    _load_dataset,
    _attach_pid_control_arrays,
    _attach_static_context_arrays,
)
from train_nominal import _build_features, _normalization

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=["nominal", "grouped"], default="grouped")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/preprocessed")
    args = p.parse_args()
    a = _load_dataset(ROOT / f"data/processed/{args.protocol}.npz")
    rebuilt = {
        k: np.empty_like(a[k])
        for k in (
            "p",
            "pdot",
            "u",
            "p_ref",
            "pdot_ref",
            "U_current",
            "U_next",
            "next_state",
            "accel",
            "mechanical_resistance",
        )
    }
    ids = a["trajectory_id"].reshape(-1)
    for tid in np.unique(ids):
        rows = np.flatnonzero(ids == tid)
        path = Path(a["source_paths"][int(tid)])
        with np.load(path, allow_pickle=False) as z:
            t = z["T"].reshape(-1)
            now = np.searchsorted(t, a["time"][rows, 0])
            nxt = np.searchsorted(t, a["time_next"][rows, 0])
            np.testing.assert_allclose(t[now], a["time"][rows, 0], atol=1e-12)
            np.testing.assert_allclose(t[nxt], a["time_next"][rows, 0], atol=1e-12)
            dt = (t[nxt] - t[now])[:, None]
            np.testing.assert_allclose(dt, a["dt"][rows], atol=1e-12)
            for target, key in [
                ("p", "P_actual"),
                ("pdot", "P_dot_actual"),
                ("u", "U_control"),
                ("p_ref", "P_ref"),
                ("pdot_ref", "P_dot_ref"),
            ]:
                rebuilt[target][rows] = z[key][now]
            energy = z["U" if "U" in z else "U_vec"].reshape(-1)
            rebuilt["U_current"][rows, 0] = energy[now]
            rebuilt["U_next"][rows, 0] = energy[nxt]
            rebuilt["next_state"][rows] = np.hstack(
                [z["P_actual"][nxt], z["P_dot_actual"][nxt]]
            )
            rebuilt["accel"][rows] = (
                z["P_dot_actual"][nxt] - z["P_dot_actual"][now]
            ) / dt
            rebuilt["mechanical_resistance"][rows] = (
                z["U_control"][now] - 7.27 * rebuilt["accel"][rows]
            )
    deltas = {k: float(np.max(np.abs(v - a[k]))) for k, v in rebuilt.items()}
    for k, v in rebuilt.items():
        np.testing.assert_allclose(v, a[k], rtol=1e-10, atol=1e-7)
    a.update(rebuilt)
    for key in (
        "static_context",
        "static_context_fields",
        "pid_kp",
        "pid_kd",
        "pid_ki",
        "pid_error_sign",
        "arctan_k",
        "arctan_u_scale",
        "sim_time",
        "u_min",
        "u_max",
    ):
        a.pop(key, None)
    _attach_pid_control_arrays(a)
    _attach_static_context_arrays(a)
    local, global_, ll, gl = _build_features(a)
    train = np.flatnonzero(a["split"] == 0)
    residual = np.stack([a["p"] - a["p_ref"], a["pdot"] - a["pdot_ref"]], axis=-1)
    ff = (
        a["arctan_u_scale"]
        * (2 / np.pi)
        * np.arctan(a["arctan_k"] * a["time"] / a["sim_time"])
    )
    force = a["u"] - ff
    if args.protocol == "grouped":
        force = force / a["pid_kp"]
    norm = _normalization(
        local, global_, residual, a["accel"], a["U_current"], force, train
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Store portable paths in the regenerated snapshot.
    a["source_paths"] = np.asarray(
        [Path(x).relative_to(ROOT).as_posix() for x in a["source_paths"]]
    )
    np.savez_compressed(args.output_dir / f"{args.protocol}.npz", **a)
    np.savez_compressed(
        args.output_dir / f"{args.protocol}_features.npz",
        local=local,
        global_context=global_,
        **norm,
    )
    report = {
        "protocol": args.protocol,
        "rows": len(ids),
        "local_features": ll,
        "global_features": gl,
        "raw_label_max_absolute_differences": deltas,
        "normalization_split": 0,
        "unused_legacy_arrays": "X and auxiliary energy-gradient labels remain archival fields; the response network uses the documented 6 local and 29 global features.",
    }
    (args.output_dir / f"{args.protocol}_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
