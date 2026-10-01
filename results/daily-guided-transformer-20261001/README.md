# Daily guided Transformer comparison (2026-10-01)

Results of `scripts/run_daily_guided_transformer.sh` on branch
`feature/yehyeon-daily-transformer` at `9c3c021`, launched with
[`launch.sh`](launch.sh). See [`docs/daily_transformer_training.md`](../../docs/daily_transformer_training.md)
for the data route.

- Data: daily ERA5 1979-01-01–2025-12-31, pooled to 16×32; targets msl, t2m, u10, v10
- Forecast: 6 daily history steps → leads 24/48/72/96/120 h
- Model: `raw_latent_guide_transformer_v1` (hidden 128, depth 4, 4 heads, patch 2), 20 epochs, batch 16
- Arms: raw, latent, guided (learned guide), guided (zero guide); seeds 7, 19, 43
- Run finished 2026-10-02 04:08 KST, exit 0. Checkpoints (`*.pt`) and forecasts (`*.npz`) are not committed.

## Normalized RMSE (mean ± sample std over 3 seeds)

| Arm | Validation | Test |
|---|---|---|
| **Guided (learned)** | **0.5914 ± 0.0021** | **0.5926 ± 0.0023** |
| Raw | 0.5970 ± 0.0062 | 0.5991 ± 0.0067 |
| Latent | 0.6073 ± 0.0013 | 0.6086 ± 0.0013 |
| Guided (zero) | 0.6149 ± 0.0047 | 0.6160 ± 0.0044 |

Persistence: ≈ 0.927 (test).

On test, Guided (learned) reduces normalized RMSE by 1.08 % vs. Raw, 2.62 % vs. Latent
and 3.80 % vs. Guided (zero), and is better than each in all three paired seeds.
The margin over Raw is about one Raw seed standard deviation, so three seeds do not
establish it firmly.

By variable and lead (test, `fig2`): against Raw, Guided is 0.4–1.7 % worse at 24 h
for all four variables and 0.3–2.2 % better from 48 h on (except t2m at 48 h). Against
Guided (zero) it is better in every cell (up to 8.6 % for msl at 48 h), so the gain
comes from the learned guide input rather than the extra capacity. Selection MSE for
Guided is still decreasing at epoch 20, while Latent plateaus around epoch 10 (`fig4`).

## Contents

- `metrics/` — `comparison*.json/csv` (validation: no suffix; test: `.test`), per-variable/lead
  scores (`*.climode.csv`), paired effects (`*-effects.csv`), per-model metrics, manifests and
  validation/test evaluations.
- `figures/` — PNG (300 dpi) and PDF for test and validation:
  `fig1` overall nRMSE, `fig2` RMSE-reduction heatmaps (bold = all three seeds agree in sign),
  `fig3` RMSE by lead (line = seed mean, band = seed range), `fig4` training curves.
- `logs/run.log` — full run log.

Regenerate the figures (needs matplotlib and pandas):

```
python scripts/plot_daily_guided_comparison.py results/daily-guided-transformer-20261001/metrics <out dir>
```
