"""Verify standalone data, reported grouped metrics, integrity and release size."""

import argparse, csv, hashlib, json, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-checksums", action="store_true")
    args = p.parse_args()
    files = set(
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .strip("\0")
        .split("\0")
    )
    files.discard("")
    files.discard("SHA256SUMS")
    files = [ROOT / f for f in sorted(files) if (ROOT / f).is_file()]
    sizes = [f.stat().st_size for f in files]
    if max(sizes) > 90 * 2**20:
        raise ValueError("A file exceeds this project's 90 MiB file budget")
    if sum(sizes) > 900 * 2**20:
        raise ValueError("Release exceeds this project's 900 MiB checkout budget")
    forbidden = [
        str(f.relative_to(ROOT))
        for f in files
        if f.suffix.lower() in {".tex", ".bib", ".mat"}
        or any(x in f.name.lower() for x in ("integrator.py", "shell.py", "abaqus.py"))
    ]
    if forbidden:
        raise ValueError(forbidden)
    runs = json.loads((ROOT / "configs/experiments.json").read_text())["runs"]
    for run in runs:
        if "seed" in run or not run.get("run_id", "").startswith("run_"):
            raise ValueError("Released runs must use anonymous run IDs")
        if "seed_" in run["directory"]:
            raise ValueError("A run directory exposes a numeric seed")
    groups = json.loads((ROOT / "results/reported/case_groups.json").read_text())
    mapping = {int(k): v for k, v in groups["case_to_cyclic_group"].items()}
    per = {}
    fields = ("p", "pdot", "accel", "energy", "resistance", "force")
    for r in runs:
        if r["protocol"] != "grouped":
            continue
        with np.load(ROOT / r["predictions"]) as z:
            ids = z["trajectory_id"].reshape(-1)
            assert np.isin(ids, list(mapping)).sum() == 6126
            for scope, cases in [
                ("original_3", set(groups["original_cases"])),
                ("expanded_25", set(mapping)),
            ]:
                vals = []
                for g in sorted({mapping[i] for i in cases}):
                    mask = np.isin(ids, [i for i in cases if mapping[i] == g])
                    v = []
                    for k in fields:
                        t = z[k + "_truth"][mask].astype(np.float64)
                        pred = z[k + "_pred"][mask].astype(np.float64)
                        v.append(
                            float(100 * np.linalg.norm(pred - t) / np.linalg.norm(t))
                        )
                    vals.append(v)
                v = np.mean(vals, axis=0)
                per.setdefault((r["variant"], scope), []).append(np.r_[v, v.mean()])
    expected = list(
        csv.DictReader(
            (ROOT / "results/reported/ablation_three_pinn_grouped/summary.csv").open()
        )
    )
    for row in expected:
        if row["scope"] not in ("original_3", "expanded_25"):
            continue
        model = "soft_pinn" if row["model"] == "soft_pinn_w2" else row["model"]
        v = np.asarray(per[model, row["scope"]])[
            :, (*fields, "total").index(row["metric"])
        ]
        np.testing.assert_allclose(
            [v.mean(), v.std(ddof=1)],
            [float(row["mean_percent"]), float(row["sample_sd_percent"])],
            rtol=1e-12,
            atol=1e-12,
        )
    m = json.loads((ROOT / "data/splits/cyclic_groups.json").read_text())
    with np.load(ROOT / "data/processed/grouped.npz") as a:
        for group in m["groups"].values():
            selection = np.isin(a["trajectory_id"], group["trajectory_ids"])
            assert np.all(a["split"][selection] == group["split"])
    if args.write_checksums:
        entries = []
        for f in files:
            h = hashlib.sha256()
            with f.open("rb") as stream:
                for block in iter(lambda: stream.read(2**20), b""):
                    h.update(block)
            entries.append(h.hexdigest() + "  " + f.relative_to(ROOT).as_posix())
        (ROOT / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
    elif (ROOT / "SHA256SUMS").exists():
        for line in (ROOT / "SHA256SUMS").read_text().splitlines():
            expected_hash, name = line.split("  ", 1)
            h = hashlib.sha256()
            with (ROOT / name).open("rb") as stream:
                for block in iter(lambda: stream.read(2**20), b""):
                    h.update(block)
            assert h.hexdigest() == expected_hash, name
    print(
        json.dumps(
            {
                "files": len(files),
                "checkout_MiB": sum(sizes) / 2**20,
                "largest_file_MiB": max(sizes) / 2**20,
                "trained_checkpoints": len(runs),
                "grouped_reported_metrics": "exactly reproduced from archived predictions",
                "group_isolation": "verified",
                "size_budget": "90 MiB per file and 900 MiB total",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
