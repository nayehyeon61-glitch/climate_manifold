# full_joint_bs16_20260927_042919

Joint three-arm comparison (raw / E-F-D forecast-only / E-F-D + Climate Manifold)
for `neural_ode` and `climode`, seeds 7, 19, 43 (18 runs), commit `6bec5a7`.

- Training: `scripts/run_model_comparison.sh` with `BATCH_SIZE=16`, `EPOCHS=20`,
  `WINDOW_STRIDE=1`, `HISTORY_STEPS=6`, `HISTORY_STRIDE=4`, `HORIZON_STEPS=20`, PINN levels 500/850.
- Data: `experiments/a64-b512-pinn-20260922/data/{surface,information-pinn}.npz`.
- Evaluation: every checkpoint scored once on `validation` (during the sweep) and once on
  `test` (after the sweep) with `scripts/post_sweep_test_report.sh`.
- Report: `scripts/climode_report.py`; ClimODE-protocol scores cross-checked against
  Aalto-QuML/ClimODE `evaluation_rmsd_mm` / `evaluation_acc_mm` (commit e729d23).

| Path | Contents |
|---|---|
| `REPORT.html` | Tables and interpretation (validation vs test) |
| `per_run/` | Per-run `*.validation.json`, `*.test.json`, metadata, learning curves |
| `report/{validation,test}/` | CSV summaries, lead-time RMSE/ACC, maps, GIFs, cross-check |
| `logs/` | Sweep log and post-sweep test/report log |

Checkpoints (`*.pt`) and forecast arrays (`*.npz`) are not committed (`.gitignore`);
they remain in `runs/full_joint_bs16_20260927_042919/` on the training machine.
