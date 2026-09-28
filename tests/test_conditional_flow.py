"""Observed-history CFM learns a conditional vector field, independently of F."""
from types import SimpleNamespace

import pytest
import torch

from test_pinn_training import pinn_prepared
from test_reconstruction_objective import split_prepared
from climate_manifold.downstream.conditional_flow import (
    ObservedConditionalFlow, conditional_flow_config_from_args,
    conditional_flow_config_from_payload, make_conditional_flow_config,
    observed_conditional_flow_losses, sample_observed_conditional_flow,
    validate_conditional_flow_config,
)
from climate_manifold.downstream.constraint_protocol import make_constraint_contract


def payload_with_contract(config):
    return {'constraint_pair': 'pinn_statistical', 'conditional_flow_config': config,
            'constraint_contract': make_constraint_contract(
                'pinn_statistical', ['pinn', 'statistical'], 'separate_surface_and_information')}


def prepared(split_prepared, mode='separate_surface_and_information'):
    pipe, previous = split_prepared
    config = make_conditional_flow_config(.2, quantiles=5, hidden_dim=12, noise_scale=.3)
    pipe.conditional_flow = ObservedConditionalFlow(pipe.bridge.manifold, config, mode)
    batch = {**previous, 'targets': previous['targets'].clone().detach().requires_grad_(),
             'statistical_flow_information_targets': previous['information_targets'].clone().detach().requires_grad_()}
    return pipe, batch, torch.tensor([6., 12.]), config


def generator(seed=73):
    return torch.Generator().manual_seed(seed)


def test_disabled_loss_does_not_require_pipeline_or_targets():
    losses = observed_conditional_flow_losses(None, {}, [], None)
    assert all(value == 0 and not value.requires_grad for value in losses.values())
    assert conditional_flow_config_from_payload({}) is None
    assert conditional_flow_config_from_args(SimpleNamespace(), None) is None


@pytest.mark.parametrize('kind', ('w2', 'kl_entropy'))
def test_config_is_orthogonal_to_reconstruction_metric(kind):
    args = SimpleNamespace(conditional_flow_weight=.1, statistical_loss=kind,
                           constraint_decoder=None)
    config = conditional_flow_config_from_args(args, 'pinn_statistical')
    assert config == make_conditional_flow_config(.1)
    payload = {**payload_with_contract(config), 'statistical_loss': kind, 'conditional_flow_weight': .1}
    assert conditional_flow_config_from_payload(payload) == config


@pytest.mark.parametrize('changes', ({'weight': -1.}, {'weight': True}, {'weight': float('nan')},
    {'weight': float('inf')}, {'quantiles': 0}, {'quantiles': 513}, {'quantiles': True},
    {'hidden_dim': 0}, {'hidden_dim': 4097}, {'hidden_dim': 1.5},
    {'noise_scale': 0.}, {'noise_scale': -1.}, {'noise_scale': float('nan')}))
def test_invalid_config_values_are_rejected(changes):
    with pytest.raises(ValueError):
        make_conditional_flow_config(**{'weight': .1, **changes})


@pytest.mark.parametrize('changes,pair', (
    ({}, 'pinn_static'), ({'bridge': 'raw'}, 'pinn_statistical'),
    ({'training_mode': 'frozen'}, 'statistical_static'),
    ({'anchor': 'origin'}, 'pinn_statistical'),
    ({'constraint_decoder': 'surface_and_information'}, 'pinn_statistical'),
    ({'statistical_flow_weight': .1}, 'pinn_statistical'),
))
def test_incompatible_routes_are_rejected(changes, pair):
    with pytest.raises(ValueError):
        conditional_flow_config_from_args(SimpleNamespace(conditional_flow_weight=.1, **changes), pair)


def test_configuration_integrity_and_inactive_tuning_options():
    config = make_conditional_flow_config(.2)
    for key, replacement in (('path', 'forecast_latent'), ('sampling_rearrangement', 'none'),
                              ('weight', 0.), ('noise_scale', 0.)):
        with pytest.raises(ValueError):
            validate_conditional_flow_config({**config, key: replacement})
    with pytest.raises(ValueError, match='Missing'):
        conditional_flow_config_from_payload({'conditional_flow_weight': .2})
    with pytest.raises(ValueError, match='disagrees'):
        conditional_flow_config_from_payload({**payload_with_contract(config), 'conditional_flow_weight': .3})
    with pytest.raises(ValueError, match='positive'):
        conditional_flow_config_from_args(SimpleNamespace(conditional_flow_quantiles=32), 'pinn_statistical')
    with pytest.raises(ValueError, match='mutually exclusive'):
        conditional_flow_config_from_payload({**payload_with_contract(config),
                                              'statistical_flow_config': {'weight': .1}})


def test_legacy_shared_decoder_metadata_cannot_claim_conditional_flow():
    payload = payload_with_contract(make_conditional_flow_config(.1))
    payload['constraint_contract'] = {'version': 'climate_manifold.reconstruction_constraints.v1',
                                      'pair': 'pinn_statistical'}
    with pytest.raises(ValueError, match='independent of forecast D'):
        conditional_flow_config_from_payload(payload)
    del payload['constraint_contract']
    with pytest.raises(ValueError, match='constraint_contract'):
        conditional_flow_config_from_payload(payload)


@pytest.mark.parametrize('mode', ('separate_surface_and_information', 'information_only'))
def test_only_observed_encoder_and_auxiliary_decoders_receive_flow_gradients(split_prepared, monkeypatch, mode):
    pipe, batch, leads, config = prepared(split_prepared, mode)
    manifold = pipe.bridge.manifold
    encoded = []
    original = manifold.raw_encode

    def encode(states, info):
        encoded.append((states.detach().clone(), info.detach().clone()))
        return original(states, info)

    def forbidden(*args, **kwargs):
        raise AssertionError('No forecast F, forecast D, or PINN enters observed CFM')

    monkeypatch.setattr(manifold, 'raw_encode', encode)
    monkeypatch.setattr(pipe.predictor, 'forward', forbidden)
    monkeypatch.setattr(manifold.core.manifold.decoder, 'forward', forbidden)
    monkeypatch.setattr(manifold.pinn, 'forward', forbidden)
    if mode == 'information_only':
        monkeypatch.setattr(pipe.reconstruction_decoder, 'forward', forbidden)
    losses = observed_conditional_flow_losses(pipe, batch, leads, config, generator=generator())
    losses['conditional_flow_regularization'].backward()
    torch.testing.assert_close(losses['conditional_flow_regularization'], .2*losses['conditional_flow'])
    assert len(encoded) == 1
    torch.testing.assert_close(encoded[0][0], batch['constraint_states'])
    torch.testing.assert_close(encoded[0][1], batch['constraint_information'])
    groups = [manifold.core.manifold.encoder, manifold.information, manifold.info_head, pipe.conditional_flow]
    if mode != 'information_only':
        groups.append(pipe.reconstruction_decoder)
    for group in groups:
        gradients = [p.grad for p in group.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    for group in (pipe.predictor, manifold.core.manifold.decoder, manifold.pinn):
        assert all(p.grad is None for p in group.parameters())
    for name in ('targets', 'statistical_flow_information_targets', 'constraint_states', 'constraint_information'):
        assert batch[name].grad is None


def test_head_does_not_register_the_manifold_or_forecast_parameters(split_prepared):
    pipe, _, _, _ = prepared(split_prepared)
    flow = pipe.conditional_flow
    existing = {id(p) for p in pipe.bridge.manifold.parameters()} | {id(p) for p in pipe.predictor.parameters()}
    assert not (existing & {id(p) for p in flow.parameters()})
    assert all(not key.startswith(('core.', 'manifold.', 'info_head.')) for key in flow.state_dict())


def test_future_labels_only_change_detached_interpolant_not_observed_condition(split_prepared):
    pipe, batch, leads, config = prepared(split_prepared)
    inputs = []

    def capture(_module, args):
        inputs.append(tuple(value.detach().clone() if isinstance(value, torch.Tensor) else value for value in args))
        assert not args[0].requires_grad and args[3].requires_grad

    hook = pipe.conditional_flow.register_forward_pre_hook(capture)
    try:
        original = observed_conditional_flow_losses(pipe, batch, leads, config, generator=generator())
        altered = {**batch, 'targets': batch['targets']+7.,
                   'statistical_flow_information_targets': batch['statistical_flow_information_targets']+7.}
        changed = observed_conditional_flow_losses(pipe, altered, leads, config, generator=generator())
    finally:
        hook.remove()
    assert len(inputs) == 2
    assert not torch.equal(inputs[0][0], inputs[1][0])
    torch.testing.assert_close(inputs[0][3], inputs[1][3], rtol=0, atol=0)
    # One time parameter is shared by every lead, variable and quantile.
    assert inputs[0][1].shape == (2, 1, 1, 1)
    assert original['conditional_flow'] != changed['conditional_flow']


def test_static_future_information_is_excluded(split_prepared):
    pipe, batch, leads, config = prepared(split_prepared)
    original = observed_conditional_flow_losses(pipe, batch, leads, config, generator=generator())
    poison = batch['statistical_flow_information_targets'].detach().clone()
    head = pipe.conditional_flow
    fields = poison.reshape(*poison.shape[:2], *head.info_shape)
    for index in range(head.info_shape[0]):
        if index not in head.dynamic_indices:
            fields[:, :, index] = float('nan')
    changed = observed_conditional_flow_losses(pipe, {**batch, 'statistical_flow_information_targets': poison},
                                               leads, config, generator=generator())
    for key in original:
        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
    assert all('terrain' not in name for name in head.variable_names)


@pytest.mark.parametrize('dt', (None, torch.ones(2, 2), torch.tensor([[6., 12.], [6., 12.]]),
                               torch.tensor([[6., float('nan')], [6., 6.]])))
def test_future_time_alignment_is_checked_before_encoding(split_prepared, monkeypatch, dt):
    pipe, batch, leads, config = prepared(split_prepared)

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid temporal alignment must fail before encoding')

    monkeypatch.setattr(pipe.bridge.manifold, 'raw_encode', forbidden)
    with pytest.raises(ValueError, match='dt_hours'):
        observed_conditional_flow_losses(pipe, {**batch, 'dt_hours': dt}, leads, config)


def test_temporal_head_couples_adjacent_future_leads(split_prepared):
    pipe, _, leads, _ = prepared(split_prepared)
    head = pipe.conditional_flow
    state = torch.zeros(2, 2, head.variable_count, head.config['quantiles'])
    context = torch.zeros(2, head.config['hidden_dim'])
    original = head(state, .5, leads, context)
    state[:, 1] += 1.
    changed = head(state, .5, leads, context)
    assert not torch.equal(original[:, 0], changed[:, 0])


def test_sampling_uses_only_observed_pair_repeats_seed_and_preserves_modes(split_prepared, monkeypatch):
    pipe, batch, leads, _ = prepared(split_prepared)
    observed = {key: batch[key] for key in ('constraint_states', 'constraint_information', 'constraint_dt_hours')}

    def forbidden(*args, **kwargs):
        raise AssertionError('Forecast modules must not enter observed CFM sampling')

    monkeypatch.setattr(pipe.predictor, 'forward', forbidden)
    monkeypatch.setattr(pipe.bridge.manifold.core.manifold.decoder, 'forward', forbidden)
    pipe.train()
    pipe.reconstruction_decoder.eval()
    original_modes = {module: module.training for module in pipe.modules()}
    rng = torch.random.get_rng_state().clone()
    first = sample_observed_conditional_flow(pipe, observed, leads, members=3, steps=3, seed=12)
    second = sample_observed_conditional_flow(pipe, observed, leads, members=3, steps=3, seed=12)
    changed = sample_observed_conditional_flow(pipe, observed, leads, members=3, steps=3, seed=13)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(module.training == mode for module, mode in original_modes.items())
    quantiles = first['quantiles']
    assert quantiles.shape == (2, 3, 2, pipe.conditional_flow.variable_count, 5)
    assert torch.isfinite(quantiles).all() and not quantiles.requires_grad
    assert (quantiles.diff(dim=-1) >= 0).all()
    torch.testing.assert_close(quantiles, second['quantiles'], rtol=0, atol=0)
    assert not torch.equal(quantiles, changed['quantiles'])
    assert not torch.equal(quantiles[:, 0], quantiles[:, 1])
    torch.testing.assert_close(first['lead_hours'], leads)
    torch.testing.assert_close(first['quantile_levels'], torch.tensor([.1, .3, .5, .7, .9]))


def test_midpoint_sampler_exactly_integrates_constant_velocity_before_final_sort(split_prepared, monkeypatch):
    pipe, batch, leads, _ = prepared(split_prepared)
    head = pipe.conditional_flow
    monkeypatch.setattr(head, 'forward', lambda state, tau, leads, context: torch.full_like(state, 2.))
    first = sample_observed_conditional_flow(pipe, batch, leads, members=2, steps=1, seed=5)
    second = sample_observed_conditional_flow(pipe, batch, leads, members=2, steps=8, seed=5)
    torch.testing.assert_close(first['quantiles'], second['quantiles'], rtol=1e-6, atol=1e-6)
    monkeypatch.setattr(head, 'forward', lambda state, tau, leads, context: torch.zeros_like(state))
    initial = sample_observed_conditional_flow(pipe, batch, leads, members=2, steps=1, seed=5)
    torch.testing.assert_close(first['quantiles'], initial['quantiles']+2., rtol=1e-6, atol=1e-6)


def test_sampling_fails_explicitly_for_nonfinite_trajectory_and_restores_modes(split_prepared, monkeypatch):
    pipe, batch, leads, _ = prepared(split_prepared)
    pipe.train()
    monkeypatch.setattr(pipe.conditional_flow, 'forward',
                        lambda state, *args: torch.full_like(state, float('nan')))
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        sample_observed_conditional_flow(pipe, batch, leads, members=1, steps=1)
    assert pipe.training
