"""Sample observed-pair conditional flows of normalized spatial marginals.

The auxiliary CFM head is independent of the forecast predictor. Its outputs
are marginal quantile scenarios, not spatial weather fields or calibrated
forecast ensembles. Future targets are never passed to its inference path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..physical_information import digest
from ..train import data_contract
from .conditional_flow import sample_observed_conditional_flow
from .train import load_predictor, windows


OBSERVED_INPUT_KEYS = frozenset((
    'constraint_states', 'constraint_information', 'constraint_dt_hours',
    'origin', 'information', 'origin_time_ns',
))


def sample_conditional_flow(checkpoint, archive, output, *, information=None,
                            split='validation', max_cases=0, origin_stride=1,
                            members=8, steps=32, seed=7, device='cpu'):
    """Write [case, member, lead, variable, quantile] scenarios and provenance.

    Every case gets its own seed ``seed + case_index``. Increasing max_cases
    preserves an existing prefix without relying on a particular batch size.
    """
    if split not in ('calibration', 'validation', 'test'):
        raise ValueError('Use a held-out downstream split')
    for name, value, minimum in (('max_cases', max_cases, 0), ('origin_stride', origin_stride, 1),
                                 ('members', members, 1), ('steps', steps, 1), ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if seed >= 2**63 - 1:
        raise ValueError('seed must be below 2**63 - 1')
    output = Path(output)
    if output.suffix != '.npz':
        raise ValueError('Conditional flow output must be a .npz path')
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError('Choose new conditional flow sample/report paths')

    pipeline, payload = load_predictor(checkpoint, device)
    if getattr(pipeline, 'conditional_flow', None) is None:
        raise ValueError('Checkpoint has no trained observed conditional flow head')
    metadata = payload['a_metadata']
    data = data_contract(archive, information, metadata['mode'], pipeline.a_config, metadata)
    dataset = windows(data, pipeline.a_config, split, origin_stride, max_cases,
                      reconstruction_constraints=True)
    if seed + len(dataset) - 1 >= 2**63 - 1:
        raise ValueError('Per-origin seeds exceed the supported seed range')
    leads = torch.tensor(payload['lead_hours'], device=device, dtype=torch.float32)
    scenarios, origins, seeds = [], [], []
    variable_names = quantile_levels = None
    for index in range(len(dataset)):
        # The dataset also contains supervised forecasting labels. Whitelist
        # observed inputs before sampling, so neither targets nor future
        # information/dt/history can accidentally reach the flow head.
        row = dataset[index]
        batch = {key: value[None].to(device) for key, value in row.items()
                 if key in OBSERVED_INPUT_KEYS}
        case_seed = seed + index
        result = sample_observed_conditional_flow(
            pipeline, batch, leads, members=members, steps=steps, seed=case_seed)
        values = result['quantiles'].detach().cpu().numpy()
        names = list(result['variable_names'])
        levels = result['quantile_levels'].detach().cpu().numpy()
        if (values.ndim != 5 or values.shape[:3] != (1, members, len(leads))
                or values.shape[3:] != (len(names), len(levels))
                or not np.isfinite(values).all()):
            raise FloatingPointError('Invalid conditional flow quantile samples')
        if not np.array_equal(result['lead_hours'].detach().cpu().numpy(), leads.cpu().numpy()):
            raise ValueError('Conditional flow returned incompatible lead hours')
        if variable_names is not None and (names != variable_names or not np.array_equal(levels, quantile_levels)):
            raise ValueError('Conditional flow output contract changed between origins')
        variable_names, quantile_levels = names, levels
        scenarios.append(values[0])
        origins.append(int(batch['origin_time_ns'][0].cpu()))
        seeds.append(case_seed)

    report = {
        'format': 'climate_manifold.observed_conditional_flow_samples.v1',
        'checkpoint_sha256': digest(checkpoint),
        'archive_sha256': metadata['archive_sha256'],
        'information_sha256': metadata['information_sha256'],
        'conditional_flow_config': payload['conditional_flow_config'],
        'constraint_decoder': payload['constraint_decoder'],
        'split': split, 'origin_stride': origin_stride,
        'cases': len(scenarios), 'members': members, 'steps': steps,
        'seed': seed, 'origin_seeds': seeds,
        'seed_contract': 'one local generator per origin, seeded with seed + zero-based case index',
        'noise_contract': 'independent Gaussian entries across member, lead, variable and quantile coordinates; the learned velocity couples leads',
        'flow_integrator': 'explicit midpoint in tau from 0 to 1',
        'quantile_rearrangement': 'sort each variable at each lead after the final ODE step; the integrated path is not constrained to be monotone',
        'lead_hours': payload['lead_hours'], 'variable_names': variable_names,
        'quantile_levels': quantile_levels.tolist(),
        'array_axes': ['case', 'member', 'lead', 'variable', 'quantile'],
        'origin_times': [str(np.datetime64(origin, 'ns')) + 'Z' for origin in origins],
        'sampler_inputs': sorted(OBSERVED_INPUT_KEYS),
        'forecast_predictor_used': False,
        'normalization': {
            'output_units': 'normalized variable values, not physical Pa/K/m/s',
            'surface': 'grid-cell-specific training mean and scale',
            'information': 'training information mean and scale',
            'arrays_in_npz': ['surface_mean', 'surface_scale', 'information_mean', 'information_scale'],
            'surface_shape': list(pipeline.a_config.grid),
            'information_shape': metadata['information_metadata']['shape'],
            'physical_conversion': 'A normalized surface marginal cannot be converted to a physical marginal by one scalar inverse transform.',
        },
        'schema': metadata['schema'],
        'information_metadata': metadata['information_metadata'],
        'limits': 'Marginal distribution scenarios do not encode geographic placement and are not validated as calibrated ensembles.',
    }
    arrays = {
        'normalized_quantiles': np.stack(scenarios),
        'quantile_levels': quantile_levels,
        'variable_names': np.asarray(variable_names, dtype=str),
        'lead_hours': leads.detach().cpu().numpy(),
        'origin_time_ns': np.asarray(origins, dtype=np.int64),
        'origin_seeds': np.asarray(seeds, dtype=np.int64),
        'surface_mean': np.asarray(metadata['mean']),
        'surface_scale': np.asarray(metadata['scale']),
        'information_mean': np.asarray(metadata['information_mean']),
        'information_scale': np.asarray(metadata['information_scale']),
        'metadata_json': np.asarray(json.dumps(report, ensure_ascii=False)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects existing files if another process wrote
    # to these paths while sampling was running.
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with sidecar.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'archive', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--information')
    parser.add_argument('--split', choices=('calibration', 'validation', 'test'), default='validation')
    parser.add_argument('--max-cases', type=int, default=0)
    parser.add_argument('--origin-stride', type=int, default=1)
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cpu')
    result = sample_conditional_flow(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ('cases', 'members', 'lead_hours', 'variable_names')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
