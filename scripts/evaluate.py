"""Evaluate released checkpoints without a structural solver."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from response_data import _load_dataset
from train_nominal import (
    PhysicsConstrainedResponseOperator,
    _build_features,
    _predict as strict_predict,
)
from train_response import AblatedPhysicsResponseOperator, _predict as ablation_predict

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("p", "pdot", "accel", "energy", "resistance", "force")
TRUTH = {
    "p": "p",
    "pdot": "pdot",
    "accel": "accel",
    "energy": "U_current",
    "resistance": "mechanical_resistance",
    "force": "u",
}


def registry():
    return json.loads((ROOT / "configs/experiments.json").read_text())["runs"]


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    keys = ["local_dim", "global_dim", "hidden_dim", "num_blocks", "dropout"]
    if "variant" in config:
        model = AblatedPhysicsResponseOperator(
            **{k: config[k] for k in keys},
            variant=config["variant"],
            projection_mode=config.get("projection_mode", "legacy_velocity"),
        )
        predict = ablation_predict
    else:
        model = PhysicsConstrainedResponseOperator(**{k: config[k] for k in keys})
        predict = strict_predict
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint, predict


def infer(model, checkpoint, predict, arrays, indices=None, device="cpu"):
    indices = np.arange(len(arrays["time"])) if indices is None else np.asarray(indices)
    norm = checkpoint["normalization"]
    local, global_, _, _ = _build_features(arrays)
    local = ((local - norm["local_mean"]) / norm["local_std"]).astype(np.float32)
    global_ = ((global_ - norm["global_mean"]) / norm["global_std"]).astype(np.float32)
    return predict(
        model,
        local,
        global_,
        norm,
        arrays,
        indices,
        checkpoint["rail_mass"],
        torch.device(device),
        512,
    )


def errors(pred, arrays, indices):
    result = {}
    for field in FIELDS:
        truth = np.asarray(arrays[TRUTH[field]])[indices]
        value = np.asarray(pred[field])
        mask = np.isfinite(truth) & np.isfinite(value)
        result[field] = float(
            100
            * np.linalg.norm(value[mask] - truth[mask])
            / max(np.linalg.norm(truth[mask]), 1e-12)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=["grouped", "nominal", "all"], default="grouped"
    )
    parser.add_argument("--model", default="full", help="Model name or all")
    parser.add_argument("--run", nargs="*", help="Public run IDs to evaluate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument(
        "--check-archived",
        action="store_true",
        help="Compare fresh predictions with released prediction records",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    selected = [
        r
        for r in registry()
        if (args.protocol == "all" or r["protocol"] == args.protocol)
        and (args.model == "all" or r["variant"] == args.model)
        and (not args.run or r["run_id"] in args.run)
    ]
    if not selected:
        raise ValueError("No matching checkpoints")
    groups = json.loads((ROOT / "results/reported/case_groups.json").read_text())
    mapping = {int(k): v for k, v in groups["case_to_cyclic_group"].items()}
    formal = set(mapping)
    original = set(groups["original_cases"])
    rows = []
    cache = {}
    for run in selected:
        protocol = run["protocol"]
        if protocol not in cache:
            cache[protocol] = _load_dataset(ROOT / f"data/processed/{protocol}.npz")
        arrays = cache[protocol]
        idx = np.flatnonzero(arrays["split"] == 2)
        model, ck, predict = load_model(ROOT / run["checkpoint"], args.device)
        pred = infer(model, ck, predict, arrays, idx, args.device)
        parity = None
        if args.check_archived:
            with np.load(ROOT / run["predictions"]) as z:
                np.testing.assert_array_equal(
                    arrays["trajectory_id"][idx], z["trajectory_id"].reshape(-1)
                )
                differences = {
                    k: float(
                        100
                        * np.linalg.norm(pred[k] - z[k + "_pred"])
                        / max(np.linalg.norm(z[k + "_pred"]), 1e-12)
                    )
                    for k in FIELDS
                }
                parity = max(differences.values())
                # Float32 convolution and amplified PD inversion differ across CPU/GPU backends.
                # Compare a dimensionless RMS delta, avoiding unit-dependent tolerances near zero.
                if not np.isfinite(parity) or parity > 0.05:
                    raise AssertionError(
                        f"Prediction RMS differences exceed 0.05 percent: {differences}"
                    )
        ids = arrays["trajectory_id"][idx].reshape(-1)
        for scope, cases in (
            [("original_3", original), ("expanded_25", formal)]
            if protocol == "grouped"
            else [("original_3", original)]
        ):
            per_group = []
            for group in sorted({mapping[c] for c in cases}):
                mask = np.array(
                    [int(i) in cases and mapping.get(int(i)) == group for i in ids]
                )
                subset = {k: v[mask] for k, v in pred.items()}
                per_group.append(errors(subset, arrays, idx[mask]))
            if protocol == "nominal":
                mask = np.isin(ids, list(cases))
                result = errors(
                    {k: v[mask] for k, v in pred.items()}, arrays, idx[mask]
                )
            else:
                result = {k: float(np.mean([g[k] for g in per_group])) for k in FIELDS}
            row = {
                "protocol": protocol,
                "model": run["variant"],
                "run_id": run["run_id"],
                "scope": scope,
                **result,
                "E_test": float(np.mean(list(result.values()))),
                "selected_epoch": ck["epoch"],
                "E_val": ck["validation_score"],
                "max_archived_prediction_RMS_difference_pct": parity,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_run_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for key in sorted({(r["protocol"], r["model"], r["scope"]) for r in rows}):
        sub = [r for r in rows if (r["protocol"], r["model"], r["scope"]) == key]
        s = dict(zip(["protocol", "model", "scope"], key))
        s["runs"] = len(sub)
        for field in (*FIELDS, "E_test", "E_val"):
            values = [r[field] for r in sub]
            s[field + "_mean"] = float(np.mean(values))
            s[field + "_sample_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        summary.append(s)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
