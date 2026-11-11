# Reproduction guide

Install the requirements using the README instructions, then run preprocessing and checkpoint evaluation. Generated outputs go to `outputs/`. Released archival results remain under `models/` and `results/reported/`.

```bash
python scripts/prepare_data.py --protocol nominal
python scripts/prepare_data.py --protocol grouped
python scripts/evaluate.py --protocol all --model all --check-archived
python scripts/diagnostics.py --model full --run run_01
python scripts/evaluate_supplementary.py --model full --run run_01
```

| Model identifier | Component / backbone |
|---|---|
| `full` | Complete cyclic model |
| `no_pd_projection` | Independent learned velocity |
| `no_cyclic_coupling` | Parameter-matched railwise residual blocks |
| `no_invariant_energy_pool` | Ordered trainable rail pooling |
| `no_force_balance` | Independent resistance head |
| `no_hard_closures` | Independent velocity and resistance |
| `soft_pinn` | SiLU dense model with soft constraints and kinematic weight 2 |
| `soft_pinn_tanh` | Six-layer tanh dense model, width 924 |
| `soft_pinn_fourier` | Six-layer tanh dense model, width 915, 96 temporal Fourier features |

Fourier scales are 0.5, 2 and 8, each with 16 frequencies and sine/cosine pairs. Every released matched-seed run uses 3,000 epochs and batch size 512. The saved run configuration is authoritative for model-specific optimizer and loss settings.

`--check-archived` compares inference from each released checkpoint against its saved prediction arrays. The dimensionless RMS difference must stay within 0.05% of each archived prediction field to accommodate float32 CPU/GPU backend differences and PD-error amplification. Each run records its actual maximum RMS difference. Weights remain fixed. `diagnostics.py` reports algebraic PD and force-balance residuals, native-grid time-derivative diagnostics, rail work and cyclic relabeling error. Derivative metrics depend on the query grid and differentiation convention. `evaluate_supplementary.py` uses the stored native query intervals and fixed learned weights, reporting all six response errors and the saturated rail-record fraction.

The loss and relative-RMSE plots are regenerated at every evaluation interval during training. Use the default `--eval-every 100` to preserve the archived reporting cadence. A new one-epoch run is suitable only as an execution check.

## Integrity

`SHA256SUMS` contains SHA-256 hashes for the distributed files. `release_manifest.json` maps copied scientific artifacts to relative paths in the development project. Checkpoints and archived predictions are byte-identical copies. Trajectory exports retain the recorded numeric rail/energy arrays and portable case metadata. Provenance paths use the same public run labels. The code adaptations change import paths and default data/output locations to support standalone use.
