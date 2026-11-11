"""Frozen-checkpoint evaluation of the released additional deployment commands."""

import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from response_data import (
    _load_audit_trajectory,
    _attach_pid_control_arrays,
    _attach_static_context_arrays,
)
from evaluate import ROOT, registry, load_model, infer, errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="full")
    p.add_argument("--run", default="run_01", help="Public run ID")
    p.add_argument("--case", default="all")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/supplementary")
    args = p.parse_args()
    torch.set_num_threads(2)
    run = next(
        r
        for r in registry()
        if r["protocol"] == "grouped"
        and r["variant"] == args.model
        and r["run_id"] == args.run
    )
    model, ck, predict = load_model(ROOT / run["checkpoint"], args.device)
    rows = []
    for case in json.loads((ROOT / "data/supplementary/index.json").read_text()):
        if args.case != "all" and args.case != case["case"]:
            continue
        path = ROOT / case["history"]
        raw = _load_audit_trajectory(path, 0)
        a = {k: np.asarray(v) for k, v in raw.items() if not isinstance(v, str)}
        a["source_paths"] = np.array([str(path)])
        _attach_pid_control_arrays(a)
        _attach_static_context_arrays(a)
        a["mechanical_resistance"] = a["u"] - ck["rail_mass"] * a["accel"]
        idx = np.arange(len(a["time"]))
        pred = infer(model, ck, predict, a, idx, args.device)
        e = errors(pred, a, idx)
        sat = (a["u"] <= a["u_min"] + 1e-6) | (a["u"] >= a["u_max"] - 1e-6)
        row = {
            "case": case["case"],
            "model": args.model,
            "run_id": args.run,
            "records": len(idx),
            **e,
            "saturated_rail_record_fraction": float(sat.mean()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not rows:
        raise ValueError("No matching supplementary case")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.model}_{args.run}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
