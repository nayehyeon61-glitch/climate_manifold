# pinn_statistical_flow_off_20260928_125742

Pairwise raw vs E→model→D comparison with PINN + statistical constraints, Flow terms off.
Commit `b6e9a3a`. 5 models × {raw, PINN+W2, PINN+KL entropy} × seeds 7, 19, 43 = 45 runs.

- Settings: `EPOCHS=20`, `BATCH_SIZE=16`, `WINDOW_STRIDE=4`, `HISTORY_STEPS=6`, `HISTORY_STRIDE=4`,
  `HORIZON_STEPS=20`, `CONSTRAINT_DECODER=separate_surface_and_information`, `STATIC_WEIGHT=0`,
  PINN levels 500/850, all Flow weights 0.
- Data: `experiments/a64-b512-pinn-20260922/data/{surface,information-pinn}.npz`.
- Evaluation: every checkpoint scored once on `validation` and once on `test` (2,102 origins each).
- `comparison*.{json,csv}` are validation; `comparison.test*` are the same comparator on test.
- `report/{validation,test}/` are produced by `scripts/climode_report.py`; ClimODE-protocol
  scores cross-checked against Aalto-QuML/ClimODE `evaluation_rmsd_mm` / `evaluation_acc_mm`.

## Test normalized RMSE (seed mean ± std; persistence 0.975)

| Model | raw | PINN+W2 | PINN+KL |
|---|---|---|---|
| convlstm | **0.6737 ± 0.0062** | 0.7041 ± 0.0004 | 0.7031 ± 0.0036 |
| simvp | 0.7239 ± 0.0041 | 0.7389 ± 0.0037 | 0.7367 ± 0.0013 |
| mlp | 0.7231 ± 0.0074 | 0.7324 ± 0.0026 | 0.7339 ± 0.0024 |
| neural_ode | 0.7263 ± 0.0078 | 0.7327 ± 0.0041 | 0.7319 ± 0.0017 |
| climode | 0.7756 ± 0.0022 | 0.7234 ± 0.0086 | 0.7226 ± 0.0106 |

## Caveat: training origins cover only 00 UTC

`WINDOW_STRIDE=4` on the 6-hourly archive keeps every fourth start, so all 2,872 training
origins are at 00 UTC while validation and test origins cover 00/06/12/18 UTC equally.
Models never see three of the four diurnal phases during training. t2m skill against
persistence drops to about 0.10–0.31 (versus about 0.43 in the stride-1
`full_joint_bs16_20260927_042919` sweep). Treat t2m conclusions as unreliable and do not
compare absolute scores with stride-1 experiments. `climode` here is the repository's
matched transport ODE (`raw_backend=matched`), not the vendored upstream ClimODE model.

Checkpoints (`*.pt`) and forecast arrays (`*.npz`) are not committed (`.gitignore`).
