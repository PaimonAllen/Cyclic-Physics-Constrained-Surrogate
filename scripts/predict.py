"""Predict physical rail responses from command/context features only."""

import argparse
from pathlib import Path
import numpy as np
from evaluate import ROOT, load_model, infer
from response_data import _load_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / "examples/command_queries.npz")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/grouped/full/run_01/model.pt",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output", type=Path, default=ROOT / "outputs/predicted_responses.npz"
    )
    args = p.parse_args()
    arrays = _load_dataset(args.inputs)
    model, checkpoint, predict = load_model(args.checkpoint, args.device)
    result = infer(model, checkpoint, predict, arrays, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, time=arrays["time"], **result)
    print(f'Saved {len(arrays["time"])} query records to {args.output}')


if __name__ == "__main__":
    main()
