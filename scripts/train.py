"""Retrain a released model using its archived experiment configuration."""

import argparse, json, subprocess, sys, secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=["nominal", "grouped"], default="grouped")
    p.add_argument("--model", default="full")
    p.add_argument("--run", help="Public run ID used as the configuration template")
    p.add_argument("--seed", type=int, help="Optional initialization seed for a new local training run")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument(
        "--batch-size",
        type=int,
        help="Override the archived batch size for available memory",
    )
    p.add_argument("--dataset", type=Path, help="Optional alternate prepared dataset")
    p.add_argument(
        "--resume-from", type=Path, help="Explicit nominal checkpoint continuation"
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    runs = json.loads((ROOT / "configs/experiments.json").read_text())["runs"]
    matching = [
        r for r in runs if r["protocol"] == args.protocol and r["variant"] == args.model
    ]
    if not matching:
        raise ValueError("Unsupported protocol/model combination")
    template = next((r for r in matching if args.run is None or r["run_id"] == args.run), None)
    if template is None:
        raise ValueError("No configuration matches the requested run ID")
    cfg = json.loads((ROOT / template["config"]).read_text())
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    cfg["resume_from"] = str(args.resume_from) if args.resume_from else None
    if template["kind"] == "strict":
        import train_nominal as trainer
    else:
        import train_response as trainer
    out = (
        args.output_dir
        or ROOT / f"outputs/training/{args.protocol}/{args.model}/{template['run_id']}"
    )
    cfg.update(
        dataset=str(args.dataset or ROOT / f"data/processed/{args.protocol}.npz"),
        output_dir=str(out),
        seed=seed,
        epochs=args.epochs,
        eval_every=args.eval_every,
        device=args.device,
    )
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    command = [sys.executable, "-s", str(Path(trainer.__file__))]
    for action in trainer.build_parser()._actions:
        key = action.dest
        if (
            key in ("help", "resume", "warm_start")
            or key not in cfg
            or cfg[key] is None
        ):
            continue
        flag = next((x for x in action.option_strings if x.startswith("--")), None)
        if flag is None:
            continue
        value = cfg[key]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                command.append(flag)
        else:
            command.extend(
                [flag, *map(str, value if isinstance(value, list) else [value])]
            )
    print(f"Training {args.protocol}/{args.model} with template {template['run_id']}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
