"""Train-only observed-transition measures: leakage, multimodality and gradients."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from climate_manifold.architecture import ManifoldConfig
from climate_manifold.downstream.dynamics_guide import DynamicsGuide
from climate_manifold.downstream.transition_measure import (
    TransitionBank, make_transition_config, transition_config_from_payload,
    transition_measure_losses, validate_transition_config, validate_transition_bank_manifest,
)
from climate_manifold.temporal_supervision import TemporalWindowDataset, area_weights


def predictor(version='v2', **extra):
    return SimpleNamespace(dynamics_version=version, dynamics_encoder='conv_gru',
                           dynamics_mode='learned', **extra)


def objective(**extra):
    options = {'transition_neighbors': 4, 'transition_bank_size': 16,
               'transition_pool_lat': 2, 'transition_pool_lon': 3, 'horizon_steps': 3}
    options.update({'transition_'+key: value for key, value in extra.items()})
    if 'forecast_steps' in extra:
        options['horizon_steps'] = extra['forecast_steps']
        options.pop('transition_forecast_steps')
    return make_transition_config(SimpleNamespace(**options), predictor())


def example(*, count=200, max_start=120):
    config = ManifoldConfig(state_dim=48, grid=(2, 4, 6), history_steps=3,
                            history_stride=2, horizon_steps=3, step_hours=24,
                            representation_kind='spatial', latent_channels=2)
    rng = np.random.default_rng(19)
    states = rng.normal(size=(count, config.state_dim)).astype(np.float32)
    times = np.datetime64('2001-01-01')+np.arange(count).astype('timedelta64[D]')
    schema = {'variables': [
        {'name': n, 'coords': {'lat': [-60., -20., 20., 60.],
                              'lon': [0., 60., 120., 180., 240., 300.]}}
        for n in ('msl', 't2m')], 'state_dim': 48, 'forecast_step_hours': 24}
    prefix = max_start+config.history_span_steps+config.horizon_steps
    data = dict(states=states, times=times, schema=schema,
                split={'train': list(range(max_start+1)), 'validation': [max_start+10]},
                mean=states[:prefix].mean(0), scale=states[:prefix].std(0), train_end=prefix)
    starts = list(range(0, max_start+1, 12))
    return data, config, starts


def batch_for(data, config, starts):
    dataset = TemporalWindowDataset(data['states'], data['times'], config, starts,
                                    data['mean'], data['scale'], data['schema'])
    return next(iter(DataLoader(dataset, batch_size=len(starts))))


def pipeline_for(config, kind='conv_gru'):
    torch.manual_seed(8)
    return SimpleNamespace(a_config=config, dynamics_encoder=DynamicsGuide(
        config.grid, (2, 2, 3), config.history_steps, hidden=4, kind=kind,
        history_dt_hours=config.step_hours*config.history_stride),
        predictor=nn.Linear(config.state_dim, config.state_dim))


def checkpoint_for(data, config, bank):
    return {'dynamics_transition_config': bank.config, 'dynamics_transition_bank': bank.manifest,
            'a_metadata': {'config': asdict(config), 'split': data['split']},
            'lead_hours': [i*config.step_hours for i in range(1, bank.config['forecast_steps']+1)],
            'training_contract': {'train_starts_sha256': bank.manifest['training_starts_sha256']}}


def test_v1_is_zero_cost_compatible_and_rejects_silent_v2_options():
    assert make_transition_config(SimpleNamespace(), predictor('v1')) is None
    assert transition_config_from_payload({}, predictor('v1')) is None
    with pytest.raises(ValueError, match='v2'):
        make_transition_config(SimpleNamespace(transition_weight=.1), predictor('v1'))
    with pytest.raises(ValueError, match='v1'):
        transition_config_from_payload({'dynamics_transition_config': objective()}, predictor('v1'))


@pytest.mark.parametrize('changes', [
    {'weight': 0}, {'weight': float('nan')}, {'neighbors': 1}, {'neighbors': True},
    {'bank_size': 2}, {'pool_lat': 0}, {'pool_lon': 1.5}, {'exclusion_steps': -1},
    {'endpoint_temperature': 0}, {'relation_temperature': -1},
    {'latent_temperature': float('inf')}, {'geometry_weight': -1},
    {'history_weight': -1}, {'history_weight': 0, 'future_weight': 0},
])
def test_invalid_objectives_rejected(changes):
    with pytest.raises(ValueError):
        objective(**changes)


def test_config_and_bank_persisted_contract_rejects_tampering():
    data, config, starts = example()
    bank = TransitionBank(data, config, starts, objective())
    payload = checkpoint_for(data, config, bank)
    assert transition_config_from_payload(payload, predictor()) == bank.config
    for key in ('dynamics_transition_config', 'dynamics_transition_bank'):
        changed = deepcopy(payload)
        changed.pop(key)
        with pytest.raises(ValueError):
            transition_config_from_payload(changed, predictor())
    changes = ({'train_only': False}, {'starts_sha256': '0'*64}, {'count': 50},
               {'train_prefix_end': 150}, {'train_last_time_ns': 0},
               {'training_starts_sha256': '0'*64}, {'transition_config_sha256': '0'*64})
    for mutation in changes:
        changed = deepcopy(payload)
        changed['dynamics_transition_bank'].update(mutation)
        with pytest.raises(ValueError):
            transition_config_from_payload(changed, predictor())
    changed = {**bank.config, 'surprise': 1}
    with pytest.raises(ValueError):
        validate_transition_config(changed)


def test_bank_only_retains_training_prefix_and_actual_loader_subset():
    data, config, starts = example()
    cfg = objective(bank_size=6)
    bank = TransitionBank(data, config, starts[:9], cfg)
    assert len(bank.states) == data['train_end'] and len(bank.times) == data['train_end']
    assert set(bank.starts.tolist()) <= set(starts[:9])
    assert bank.manifest['count'] == 6
    assert bank.manifest['training_starts_count'] == 9
    assert bank.manifest['training_starts_sha256'] == hashlib.sha256(json.dumps(starts[:9]).encode()).hexdigest()
    poisoned = deepcopy(data)
    poisoned['states'][data['train_end']:] = np.nan
    again = TransitionBank(poisoned, config, starts[:9], cfg)
    assert again.manifest == bank.manifest
    torch.testing.assert_close(again.states, bank.states, rtol=0, atol=0)
    with pytest.raises(ValueError, match='training split'):
        TransitionBank(data, config, starts+[data['split']['validation'][0]], cfg)
    with pytest.raises(ValueError, match='unique'):
        TransitionBank(data, config, [0, 0], cfg)


def test_current_only_retrieval_is_future_label_invariant_and_uses_nonoverlapping_windows():
    data, config, starts = example()
    bank = TransitionBank(data, config, list(range(90)), objective(exclusion_steps=3))
    batch = batch_for(data, config, [0, 32, 70])
    original = bank.retrieve(batch)
    # Retrieval must not inspect even a poisoned future label tensor.
    batch['targets'].fill_(float('nan'))
    changed = bank.retrieve(batch)
    for key in original:
        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
    gap = config.history_span_steps+config.horizon_steps-1+3
    assert torch.all((original['indices']-original['query_starts'][:, None]).abs() > gap)
    assert torch.allclose(original['prior_weights'].sum(-1), torch.ones(3))
    with pytest.raises(ValueError, match='actual training'):
        bank.retrieve(batch_for(data, config, [96]))
    with pytest.raises(ValueError, match='training timestamps'):
        bank.retrieve(batch_for(data, config, [140]))


def test_per_query_candidate_count_is_batch_invariant_and_too_small_fails():
    data, config, _ = example()
    bank = TransitionBank(data, config, [0, 10, 20, 25], objective(neighbors=4))
    query = bank.retrieve(batch_for(data, config, [25, 0]))
    assert query['indices'].shape == (2, 3)
    assert query['valid_mask'].sum(-1).tolist() == [2, 3]
    alone = bank.retrieve(batch_for(data, config, [25]))
    torch.testing.assert_close(query['indices'][0, query['valid_mask'][0]], alone['indices'][0])
    torch.testing.assert_close(query['prior_weights'][0, query['valid_mask'][0]], alone['prior_weights'][0])
    model = pipeline_for(config)
    batch = batch_for(data, config, [25, 0])
    scores = transition_measure_losses(model, {'dynamics_history_latent': model.dynamics_encoder(batch['history'])}, batch, bank, bank.config)
    scores['transition_regularization'].backward()
    assert all(torch.isfinite(value) for value in scores.values())
    with pytest.raises(ValueError, match='at least two nonoverlapping'):
        TransitionBank(data, config, [0, 1, 2], objective(neighbors=3))


def test_coarse_descriptor_uses_area_weights_and_preserves_position():
    data, config, starts = example()
    bank = TransitionBank(data, config, starts, objective(pool_lat=4, pool_lon=6))
    area = torch.tensor(area_weights(data['schema']), dtype=torch.float32)
    field = torch.zeros(2, config.state_dim)
    field[0, 0] = 1
    field[1, 5] = 1
    descriptor = bank.describe(field)
    assert not torch.equal(descriptor[0], descriptor[1])
    torch.testing.assert_close(descriptor[0].square().sum(), area[0, 0]/2)
    # Constant fields are unchanged by area-weighted pooling in any resolution.
    coarse = TransitionBank(data, config, starts, objective(pool_lat=2, pool_lon=3))
    torch.testing.assert_close(coarse.describe(torch.ones(1, config.state_dim)).square().sum(), torch.tensor(1.))


def test_chunked_topk_matches_brute_force_and_never_uses_future_for_retrieval():
    data, config, _ = example(count=1400, max_start=1300)
    bank = TransitionBank(data, config, list(range(1200)), objective(bank_size=1100))
    batch = batch_for(data, config, [0, 511, 999])
    result = bank.retrieve(batch)
    current = bank.describe(batch['history'][:, -1])
    distances = (current[:, None]-bank.current_descriptors[None]).square().sum(-1)
    allowed = (result['query_starts'][:, None]-bank.starts[None]).abs() >= bank.window_length
    distances[~allowed] = float('inf')
    indices = torch.argsort(distances, stable=True)[:, :4]
    torch.testing.assert_close(result['indices'], bank.starts[indices], rtol=0, atol=0)
    torch.testing.assert_close(result['distances'], distances.gather(1, indices), rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['conv_gru', 'temporal_conv'])
def test_auxiliary_gradient_reaches_shared_encoder_only_and_support_is_online(kind):
    data, config, starts = example()
    cfg = objective()
    bank = TransitionBank(data, config, starts, cfg)
    batch = batch_for(data, config, [0, 60])
    batch['history'].requires_grad_()
    batch['targets'].requires_grad_()
    model = pipeline_for(config, kind)
    prediction = {'dynamics_history_latent': model.dynamics_encoder(batch['history']),
                  'mean': model.predictor(batch['history'][:, -1])}
    scores = transition_measure_losses(model, prediction, batch, bank, cfg)
    scores['transition_regularization'].backward()
    gradients = [p.grad for p in model.dynamics_encoder.parameters() if p.grad is not None]
    assert gradients and sum(g.abs().sum() for g in gradients) > 0
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(p.grad is None for p in model.predictor.parameters())
    assert batch['targets'].grad is None
    assert batch['history'].grad is not None
    assert scores['transition_neighbors'] >= 2
    assert all(torch.isfinite(x) for x in scores.values())
    before = bank.empirical_measure(model, batch)['support_latent']
    assert not before.requires_grad
    with torch.no_grad():
        model.dynamics_encoder.readout.bias.add_(2)
    after = bank.empirical_measure(model, batch)['support_latent']
    torch.testing.assert_close(after-before, torch.full_like(before, 2), atol=1e-6, rtol=1e-6)


def test_identical_current_allows_distinct_past_modes_and_future_relational_labels():
    data, config, _ = example()
    starts = [0, 20, 40, 60, 80, 100, 120]
    data['states'].fill(0)
    data['mean'].fill(0)
    data['scale'].fill(1)
    for index, start in enumerate(starts):
        sign = 1. if index % 2 == 0 else -1.
        origin = start+config.history_span_steps-1
        data['states'][start:origin] = sign
        data['states'][origin+1:origin+config.horizon_steps+1] = sign
    cfg = objective(neighbors=6)
    bank = TransitionBank(data, config, starts, cfg)
    model = pipeline_for(config)
    batch = batch_for(data, config, [0])
    measure = bank.empirical_measure(model, batch)
    torch.testing.assert_close(measure['prior_weights'], torch.full((1, 6), 1/6))
    assert measure['support_latent'].var(1).mean() > 0
    assert set(measure['indices'][0].tolist()) == set(starts[1:])
    distances = measure['history_distance']+measure['future_distance']
    target = torch.softmax(measure['log_prior_weights']-distances, -1)
    # Same-current predecessors retain separate support atoms; labels favor
    # histories with the query's past and subsequent evolution, not one mean.
    positive = torch.tensor([starts.index(i) % 2 == 0 for i in measure['indices'][0]])
    assert target[0, positive].mean() > target[0, ~positive].mean()
    batch['targets'].neg_()
    altered = bank.empirical_measure(model, batch)
    torch.testing.assert_close(altered['indices'], measure['indices'])
    torch.testing.assert_close(altered['support_latent'], measure['support_latent'])
    assert not torch.equal(altered['future_distance'], measure['future_distance'])


def test_forecast_window_bounds_all_future_labels_and_manifest():
    data, config, starts = example()
    cfg = objective(forecast_steps=1)
    bank = TransitionBank(data, config, starts, cfg)
    assert bank.future_descriptors.shape[1] == 1
    assert bank.window_length == config.history_span_steps+config.horizon_steps
    assert bank.manifest['future_steps'] == 1
    batch = batch_for(data, config, [0, 60])
    model = pipeline_for(config)
    prediction = {'dynamics_history_latent': model.dynamics_encoder(batch['history'])}
    original = transition_measure_losses(model, prediction, batch, bank, cfg)
    batch['targets'][:, 1:] = float('nan')
    changed = transition_measure_losses(model, prediction, batch, bank, cfg)
    for key in original:
        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
    payload = checkpoint_for(data, config, bank)
    payload['lead_hours'] = [24, 48]
    with pytest.raises(ValueError, match='main forecast'):
        transition_config_from_payload(payload, predictor())


def test_zero_forecast_guide_still_accepts_transition_supervision():
    control = SimpleNamespace(dynamics_version='v2', dynamics_encoder='conv_gru', dynamics_mode='zero')
    cfg = make_transition_config(SimpleNamespace(horizon_steps=3), control)
    data, config, starts = example()
    bank = TransitionBank(data, config, starts, cfg)
    assert transition_config_from_payload(checkpoint_for(data, config, bank), control) == cfg


def information_example():
    data, config, starts = example()
    names = ['z850', 'other', 'terrain_height', 'z500']
    meta = {'shape': [4, *config.grid[1:]],
            'grid': data['schema']['variables'][0]['coords'],
            'variables': [{'name': name, 'unit': 'm',
                           'kind': 'static' if name == 'terrain_height' else 'dynamic'} for name in names]}
    cells = config.grid[1]*config.grid[2]
    info = np.random.default_rng(25).normal(size=(len(data['states']), 4*cells)).astype(np.float32)
    terrain = np.linspace(-2, 3, cells, dtype=np.float32)
    info[:, 2*cells:3*cells] = terrain
    data.update(information=info, information_metadata=meta)
    return data, config, starts


def information_objective(**extra):
    cfg = objective(**extra)
    cfg.update(information_mode='geopotential_terrain', information_fields=['z500', 'z850', 'terrain_height'])
    return validate_transition_config(cfg)


def information_batch(data, config, starts):
    from climate_manifold.downstream.dynamics_information import select_dynamics_information
    result = batch_for(data, config, starts)
    origin = np.asarray(starts)+config.history_span_steps-1
    histories = np.asarray(starts)[:, None]+np.arange(config.history_steps)*config.history_stride
    futures = origin[:, None]+np.arange(1, config.horizon_steps+1)
    select = lambda array: torch.from_numpy(select_dynamics_information(
        array, data['information_metadata'], 'geopotential_terrain', raw_grid=config.grid))
    result['dynamics_information_history'] = select(data['information'][histories])
    result['dynamics_information_targets'] = select(data['information'][futures])
    return result


class InformationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Linear(config.state_dim+3*config.grid[1]*config.grid[2], 12)
        self.last_information = None

    def forward(self, history, information):
        self.last_information = information.detach().clone()
        return self.layer(torch.cat((history, information), -1))


def information_pipeline(config):
    torch.manual_seed(18)
    return SimpleNamespace(a_config=config, dynamics_encoder=InformationEncoder(config))


def test_information_bank_train_only_contiguous_reads_and_metadata_checksum():
    data, config, starts = information_example()
    original = data['information']
    reads = []

    class SliceOnly:
        def __getitem__(self, index):
            assert isinstance(index, slice) and index.step is None
            assert index.start >= 0 and index.stop <= data['train_end']
            reads.append(index)
            return original[index]

    data['information'] = SliceOnly()
    cfg = information_objective()
    bank = TransitionBank(data, config, starts, cfg)
    assert reads
    assert bank.information.shape == (data['train_end'], 3*24)
    assert bank.manifest['information_fields'] == ['z500', 'z850', 'terrain_height']
    assert bank.manifest['train_information_sha256']
    data['information'] = original.copy()
    data['information'][data['train_end']:] = np.nan
    poisoned = TransitionBank(data, config, starts, cfg)
    assert poisoned.manifest == bank.manifest
    invalid = deepcopy(data)
    invalid['information'][5, 2*24] += 1
    with pytest.raises(ValueError, match='constant'):
        TransitionBank(invalid, config, starts, cfg)


def test_current_geopotential_controls_retrieval_but_future_geopotential_does_not():
    data, config, starts = information_example()
    bank = TransitionBank(data, config, starts, information_objective())
    batch = information_batch(data, config, [0, 60])
    original = bank.retrieve(batch)
    batch['dynamics_information_targets'].fill_(float('nan'))
    repeated = bank.retrieve(batch)
    for key in original:
        torch.testing.assert_close(repeated[key], original[key])
    batch['dynamics_information_history'][:, -1, :24] += 10
    changed = bank.retrieve(batch)
    assert not torch.equal(changed['distances'], original['distances'])


def test_future_geopotential_supervises_relations_and_terrain_never_becomes_a_future_label():
    data, config, starts = information_example()
    bank = TransitionBank(data, config, starts, information_objective(forecast_steps=1))
    model = information_pipeline(config)
    batch = information_batch(data, config, [0, 60])
    original = bank.empirical_measure(model, batch)
    batch['dynamics_information_targets'][:, 0, :24] += 3
    changed = bank.empirical_measure(model, batch)
    torch.testing.assert_close(changed['indices'], original['indices'])
    torch.testing.assert_close(changed['history_distance'], original['history_distance'])
    assert not torch.equal(changed['future_distance'], original['future_distance'])
    # Neither terrain at any lead nor geopotential outside the trained forecast
    # window participates in temporal relation labels.
    batch['dynamics_information_targets'][:, :, 48:] = float('nan')
    batch['dynamics_information_targets'][:, 1:] = float('nan')
    bounded = bank.empirical_measure(model, batch)
    torch.testing.assert_close(bounded['future_distance'], changed['future_distance'])
    encoded = model.dynamics_encoder(batch['history'], batch['dynamics_information_history'])
    losses = transition_measure_losses(model, {'dynamics_history_latent': encoded}, batch, bank, bank.config)
    losses['transition_regularization'].backward()
    assert model.dynamics_encoder.layer.weight.grad.abs().sum() > 0


def test_terrain_spatial_pattern_conditions_support_without_fake_temporal_distances():
    data, config, starts = information_example()
    cfg = information_objective()
    bank = TransitionBank(data, config, starts, cfg)
    model = information_pipeline(config)
    batch = information_batch(data, config, [0, 60])
    original = bank.empirical_measure(model, batch)
    assert model.dynamics_encoder.last_information[..., 48:].std(-1).min() > 0
    shifted = deepcopy(data)
    shifted['information'][:, 48:72] += np.linspace(-2, 2, 24)
    second = TransitionBank(shifted, config, starts, cfg)
    changed = second.empirical_measure(model, information_batch(shifted, config, [0, 60]))
    for key in ('indices', 'prior_weights', 'history_distance', 'future_distance'):
        torch.testing.assert_close(changed[key], original[key], rtol=0, atol=0)
    assert not torch.equal(changed['support_latent'], original['support_latent'])
    assert second.manifest['train_information_sha256'] != bank.manifest['train_information_sha256']
