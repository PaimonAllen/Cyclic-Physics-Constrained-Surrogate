"""Check algebraic closures, time derivatives, rail work and cyclic equivariance."""

import argparse, json
import numpy as np
import torch
from response_data import _load_dataset
from evaluate import ROOT, registry, load_model, infer


def rel(error, truth):
    return float(100 * np.linalg.norm(error) / max(np.linalg.norm(truth), 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="full")
    p.add_argument("--run", default="run_01", help="Public run ID")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    run = next(
        r
        for r in registry()
        if r["protocol"] == "grouped"
        and r["variant"] == args.model
        and r["run_id"] == args.run
    )
    a = _load_dataset(ROOT / "data/processed/grouped.npz")
    formal = json.loads((ROOT / "results/reported/case_groups.json").read_text())[
        "case_to_cyclic_group"
    ]
    idx = np.flatnonzero(np.isin(a["trajectory_id"], list(map(int, formal))))
    m, c, f = load_model(ROOT / run["checkpoint"], args.device)
    pred = infer(m, c, f, a, idx, args.device)
    ff = (
        a["arctan_u_scale"][idx]
        * (2 / np.pi)
        * np.arctan(a["arctan_k"][idx] * a["time"][idx] / a["sim_time"][idx])
    )
    raw_pd = ff + a["pid_error_sign"][idx] * (
        a["pid_kp"][idx] * (a["p_ref"][idx] - pred["p"])
        + a["pid_kd"][idx] * (a["pdot_ref"][idx] - pred["pdot"])
    )
    report = {
        "model": args.model,
        "run_id": args.run,
        "pd_equality_residual_pct": rel(pred["force"] - raw_pd, a["u"][idx]),
        "rail_balance_residual_pct": rel(
            pred["force"] - c["rail_mass"] * pred["accel"] - pred["resistance"],
            a["mechanical_resistance"][idx],
        ),
        "per_simulation": [],
    }
    for tid in np.unique(a["trajectory_id"][idx]):
        rows = np.flatnonzero(a["trajectory_id"][idx] == tid)
        t = a["time"][idx[rows], 0]
        dt = a["dt"][idx[rows], 0]
        v = pred["pdot"][rows]
        pos = pred["p"][rows]
        acc = pred["accel"][rows]
        item = {
            "simulation_id": int(tid),
            "dpdt_vs_velocity_pct": rel(
                np.gradient(pos, t, axis=0, edge_order=2)[1:-1] - v[1:-1],
                a["pdot"][idx[rows]][1:-1],
            ),
            "dvdt_vs_accel_pct": rel(
                np.diff(v, axis=0) / np.diff(t)[:, None] - acc[:-1],
                a["accel"][idx[rows]][:-1],
            ),
            "work_pred_J": np.sum(
                pred["force"][rows] * v * dt[:, None], axis=0
            ).tolist(),
            "work_truth_J": np.sum(
                a["u"][idx[rows]] * a["pdot"][idx[rows]] * dt[:, None], axis=0
            ).tolist(),
        }
        report["per_simulation"].append(item)
    # This shift preserves the global summaries used by the network.
    shifted = dict(a)
    for k in (
        "p",
        "pdot",
        "u",
        "p_ref",
        "pdot_ref",
        "arctan_u_scale",
        "u_min",
        "u_max",
    ):
        shifted[k] = np.roll(a[k], 1, axis=1)
    rotated = infer(m, c, f, shifted, idx, args.device)
    report["cyclic_max_absolute_errors"] = {
        k: float(np.max(np.abs(rotated[k] - np.roll(pred[k], 1, axis=1))))
        for k in ("p", "pdot", "accel", "force", "resistance")
    }
    report["energy_invariance_max_absolute_J"] = float(
        np.max(np.abs(rotated["energy"] - pred["energy"]))
    )
    out = ROOT / "outputs/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.model}_{args.run}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "per_simulation"}, indent=2)
    )


if __name__ == "__main__":
    main()
