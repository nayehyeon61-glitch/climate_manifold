# Source provenance

- Source: https://github.com/nayehyeon61-glitch/climate_diffusion
- Branch: `feature/a64-b512-expanded`
- Pinned commit: `928f6392f78d2b632c1b0bcea47950e95c2f12c8`
- Extraction: 2026-09-23
- Target: https://github.com/nayehyeon61-glitch/climate_manifold

## File mapping

| Standalone file | Source |
|---|---|
| `architecture.py` | `manifold_moe.py`: PhysicsManifoldAE, decode/Jacobian; A-only config/core |
| `nn.py` | `moe.py`: OrthoDCT, FieldDCT, mlp only |
| `model.py` | `information_process.py`: A branches only |
| `train.py` | `train_information_process.py`: A losses, schedule, selection, sealing |
| `forecast.py` | `information_forecast.py`: auxiliary A and drift evaluation; routing removed |
| `audit.py` | `scripts/audit_information_process.py` |
| `archive.py`, `data.py` | `moe_data.py`, necessary I/O helpers from `data.py` |
| Physical/data modules | `manifold_physics.py`, `hybrid_pinn.py`, `temporal_supervision.py`, `fixed_step_data.py`, `physical_information.py`, `information_shards.py` |
| ERA5 scripts | `prepare_era5_extra.py`, `stream_era5_extra.py`, `prepare_temporal_120h.py` |
| Tests | Original PINN/ERA5/shard tests, with standalone A tests and smoke runner |

All module paths above are relative to `src/climate_manifold` or the original
`src/climate_diffusion`, unless explicitly prefixed by `scripts/`.

Hydra experts, router/gate, specialist history encoder, frozen reference copies,
projected B residual fields, B/C optimizers and checkpoint transitions are removed.
The A auxiliary raw-z sampler is preserved because it participates in A's process losses.
This changes parameter initialization RNG consumption; a fresh run with the same seed
is not promised to reproduce a complete original training run bit for bit.

Original fixed-step, physical-information and information-shard format IDs are
retained so existing prepared datasets remain reusable. The new checkpoint format
is `climate_manifold.a.v1`; old A/B/C checkpoints require explicit migration.
The five original temporal partitions are retained for experiment comparability.

`prepare_temporal_120h.py --checkpoint` now checks the actual standalone checkpoint
layout. The standalone data CLI defaults to 6h. A-only training defaults match the
expanded original runner (64 coordinates, hidden width 512, 60 epochs, curriculum
interval 4). There is no Hydra expert latent-dimension setting in this repository.

Only code, documentation and synthetic test generators are included. ERA5 arrays,
trained weights, credentials and generated experiment artifacts are not published.
No new license grant has been added; source authorship is retained.

## Verification on 2026-09-23

- `python -m pytest -q`: **65 passed** (22.54 seconds).
- Installed the package in an isolated environment and checked the train/evaluate/audit entry points.
- Expanded synthetic run: manifold 64, hidden 512, 7 A epochs including PINN warm-up;
  checkpoint reload, 20-step auxiliary and drift forecasts, four-pair geometry audit completed.
- Preflight verified the trained checkpoint's split, normalization, statistics and time contract.
- Compared against the untouched pinned source with identical retained weights, batch and RNG streams:
  both 120h forecast modes, all losses at warm-up/early/full curriculum epochs (1/2/7), and all
  retained parameter gradients were bitwise equal in the tested CPU environment.
- This comparison used a synthetic 4-variable 4×8 grid. Original instantiated parameters:
  5,302,116; standalone: 1,389,120. This reduction is specific to that fixture, not a full ERA5 benchmark.
- Environment: Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.5.3, pandas 3.0.6, xarray 2026.7.0.
  Upstream NetCDF/NumPy deprecation and ABI-size warnings were emitted; the exercised I/O checks passed.

No real ERA5 training or GPU validation was performed. These checks establish extraction fidelity
and software integration, not forecast skill, physical conservation or probabilistic calibration.
