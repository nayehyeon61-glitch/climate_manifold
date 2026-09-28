"""Causal, reproducible sampling of a saved observed-pair CFM head."""
import importlib
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_spatial_training import _capture
from test_split_training import _split_args
from climate_manifold.downstream.conditional_flow import sample_observed_conditional_flow
from climate_manifold.downstream.sample_conditional_flow import (
    OBSERVED_INPUT_KEYS, main, sample_conditional_flow,
)
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train, windows


def test_saved_flow_sampling_is_causal_reproducible_and_prefix_stable(
        pinn_prepared, tmp_path, monkeypatch):
    _, _, _, archive = pinn_prepared
    information = tmp_path/'pinn-information.npz'
    args = _split_args(archive, tmp_path, 'pinn_statistical',
                       '--conditional-flow-weight', '.1', '--conditional-flow-quantiles', '8',
                       '--conditional-flow-hidden-dim', '8')
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    original = captured[0][0]
    restored, payload = load_predictor(checkpoint)
    _, _, data = initialize_manifold(args)
    dataset = windows(data, restored.a_config, 'validation', max_windows=1,
                      reconstruction_constraints=True)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    batch = {key: value for key, value in batch.items() if key in OBSERVED_INPUT_KEYS}
    leads = torch.tensor(payload['lead_hours'])
    original.train()
    torch_state = torch.random.get_rng_state().clone()
    expected = sample_observed_conditional_flow(original, batch, leads, members=3, steps=2, seed=17)
    assert original.training
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    actual = sample_observed_conditional_flow(restored, batch, leads, members=3, steps=2, seed=17)
    assert not restored.training
    torch.testing.assert_close(expected['quantiles'], actual['quantiles'], rtol=0, atol=0)
    assert not torch.equal(actual['quantiles'][:, 0], actual['quantiles'][:, 1])

    def forbidden_forecast(*args, **kwargs):
        raise AssertionError('Sampling must not call F or the forecast decoder')

    monkeypatch.setattr(restored.predictor, 'forward', forbidden_forecast)
    monkeypatch.setattr(restored.bridge.manifold.core.manifold.decoder, 'forward', forbidden_forecast)
    module = importlib.import_module('climate_manifold.downstream.sample_conditional_flow')
    monkeypatch.setattr(module, 'load_predictor', lambda *args, **kwargs: (restored, payload))
    original_windows = module.windows
    observed_batches = []

    class PoisonedTargets:
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            row = self.dataset[index]
            for key in ('targets', 'history', 'dt_hours'):
                row[key].fill_(float('nan'))
            row['statistical_flow_information_targets'] = torch.full((2, 3), float('nan'))
            return row

    def observed_windows(*args, **kwargs):
        assert kwargs['reconstruction_constraints'] is True
        assert not kwargs.get('statistical_flow_targets', False)
        assert not kwargs.get('information_targets', False)
        return PoisonedTargets(original_windows(*args, **kwargs))

    def observed_sampler(pipeline, batch, leads, **kwargs):
        assert set(batch) <= OBSERVED_INPUT_KEYS
        assert {'constraint_states', 'constraint_information', 'constraint_dt_hours'} <= set(batch)
        observed_batches.append(set(batch))
        return sample_observed_conditional_flow(pipeline, batch, leads, **kwargs)

    monkeypatch.setattr(module, 'windows', observed_windows)
    monkeypatch.setattr(module, 'sample_observed_conditional_flow', observed_sampler)
    first = tmp_path/'samples-one.npz'
    assert main(['--checkpoint', str(checkpoint), '--archive', str(archive),
                 '--information', str(information), '--output', str(first),
                 '--max-cases', '1', '--members', '3', '--steps', '2', '--seed', '17']) == 0
    second = tmp_path/'samples-two.npz'
    report = sample_conditional_flow(checkpoint, archive, second, information=information,
                                    max_cases=2, members=3, steps=2, seed=17)
    third = tmp_path/'samples-new-seed.npz'
    sample_conditional_flow(checkpoint, archive, third, information=information,
                            max_cases=1, members=3, steps=2, seed=18)
    assert observed_batches
    with np.load(first, allow_pickle=False) as one, np.load(second, allow_pickle=False) as two, \
            np.load(third, allow_pickle=False) as new:
        assert two['normalized_quantiles'].shape == (2, 3, 2, len(two['variable_names']), 8)
        np.testing.assert_array_equal(one['normalized_quantiles'][0], two['normalized_quantiles'][0])
        # Reloaded JSON normalization metadata uses float64 intermediates;
        # compare to the original float32 training data within roundoff.
        np.testing.assert_allclose(one['normalized_quantiles'][0], actual['quantiles'][0].numpy(),
                                   rtol=1e-6, atol=1e-6)
        assert not np.array_equal(one['normalized_quantiles'], new['normalized_quantiles'])
        assert np.all(np.diff(two['normalized_quantiles'], axis=-1) >= 0)
        np.testing.assert_array_equal(two['origin_seeds'], [17, 18])
        np.testing.assert_allclose(two['quantile_levels'], (np.arange(8) + .5)/8)
        assert all(str(name).startswith(('surface:', 'information:')) for name in two['variable_names'])
        assert not any('terrain' in str(name) for name in two['variable_names'])
        np.testing.assert_array_equal(two['surface_mean'], payload['a_metadata']['mean'])
        np.testing.assert_array_equal(two['information_scale'], payload['a_metadata']['information_scale'])
        assert json.loads(str(two['metadata_json'])) == report
    assert json.loads(second.with_suffix('.json').read_text()) == report
    assert report['conditional_flow_config'] == payload['conditional_flow_config']
    assert report['forecast_predictor_used'] is False
    assert report['normalization']['output_units'].startswith('normalized')
    with pytest.raises(FileExistsError):
        sample_conditional_flow(checkpoint, archive, first, information=information)
    sidecar_only = tmp_path/'sidecar-only.json'
    sidecar_only.write_text('preserve existing report')
    with pytest.raises(FileExistsError):
        sample_conditional_flow(checkpoint, archive, sidecar_only.with_suffix('.npz'), information=information)
    assert sidecar_only.read_text() == 'preserve existing report'


@pytest.mark.parametrize('kwargs', [
    {'members': 0}, {'steps': -1}, {'seed': -1}, {'seed': 2**63},
    {'max_cases': -1}, {'origin_stride': 0}, {'split': 'train'},
])
def test_sampler_rejects_invalid_controls_before_loading(tmp_path, kwargs):
    with pytest.raises(ValueError):
        sample_conditional_flow(tmp_path/'absent.pt', tmp_path/'absent.npz',
                                tmp_path/'samples.npz', **kwargs)


def test_sampling_requires_npz_and_a_trained_flow_head(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='npz'):
        sample_conditional_flow('absent.pt', 'absent.npz', tmp_path/'samples.json')
    module = importlib.import_module('climate_manifold.downstream.sample_conditional_flow')
    monkeypatch.setattr(module, 'load_predictor', lambda *args, **kwargs: (object(), {}))
    with pytest.raises(ValueError, match='no trained observed conditional flow'):
        sample_conditional_flow('absent.pt', 'absent.npz', tmp_path/'samples.npz')
