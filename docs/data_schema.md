# Data and preprocessing

Arrays use SI units. Positions are in m, velocities in m/s, accelerations in m/s², thrust and resistance in N, and shell strain energy in J. Each source folder includes `case_config.json` with control, geometry and material context. `rail_history.npz` preserves time, twelve-rail responses, reference commands, thrust and energy. Full shell degrees of freedom and solver implementations belong to the development project.

`scripts/prepare_data.py` uses the frozen row selection to rebuild position, velocity, thrust, energy, next state, interval acceleration and `R = u − 7.27a` from the released histories. Partition assignments are fixed by `data/splits/records.csv` and `cyclic_groups.json`. Normalizers are fitted on training records only. Actual response values supply supervised targets, while command/control/static features supply network inputs.

The stored `X` array with 86 columns and auxiliary energy-gradient labels are legacy archival fields. `_build_features` constructs the actual 6 local and 29 global response-network features. The earlier 48-level, 17-channel history representation describes the data-generation background; the released network uses the explicit feature interface listed here.

The nominal table has 16,605/2,254/669 train/validation/test records from 118/20/6 simulations. Nominal formal metrics use the 347 records belonging to three representative test cases. The grouped table has 9,428/3,652/6,448 records from 84/32/28 simulations. Its formal test uses 6,126 records from 25 simulations, with three historical OOD cases reported separately. Both tables contain the same 19,528 response records; their partitions and training-only statistics differ.

## Split identifiers

`0 = train`, `1 = validation`, `2 = test`.

`trajectory_id` indexes `data/trajectories/case_XXX/`. `step_index`, `time`, `time_next` and `dt` identify recorded intervals. `data/splits/records.csv` exposes both partition labels for every row.

## Local inputs

| Field | Meaning |
|---|---|
| `p_ref` | Reference position |
| `pdot_ref` | Reference velocity |
| `p_ref_terminal` | Terminal reference displacement |
| `arctan_u_scale` | Feedforward force amplitude |
| `feedforward_force` | Feedforward force at the query time |
| `pddot_ref_fd` | Interval difference of reference velocity |

## Global context

The 29 global features are ordered as follows

```text
arctan_k, sim_time, dt, tau, arctan_ramp,
pid_kp, pid_kd, pid_ki, pid_error_sign,
log10_abs_pid_kp, log10_abs_pid_kd,
terminal_mean, terminal_std, terminal_rms,
u_scale_mean, u_scale_std, u_scale_rms,
geometry.Rb, geometry.Rd, geometry.Ru,
geometry.Hb, geometry.Hd, geometry.Hu, geometry.thickness,
material.E, material.nu, material.rho, material.rho_b, material.damping_c
```

Archived response arrays retain float64 values, normalized network inputs use float32, and reported error aggregation uses float64.
