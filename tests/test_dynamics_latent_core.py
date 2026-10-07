"""State-transition and target-coordinate gates for field/latent supervision."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from climate_manifold.downstream.dynamics_guide import DynamicsGuide
from climate_manifold.downstream.dynamics_objective import (
    dynamics_auxiliary_config_from_args, dynamics_auxiliary_config_from_payload,
    dynamics_auxiliary_losses, dynamics_contract, dynamics_latent_diagnostics,
    future_dynamics_targets, validate_dynamics_auxiliary_config,
)
from climate_manifold.downstream.dynamics_rollout import DynamicsLatentRolloutHead
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.statistical_objective import make_statistical_config


def _config(mode='field', kind='signed_measure'):
    return {
        'version': 'climate_manifold.dynamics_auxiliary.v2',
        'statistical_loss_config': make_statistical_config(kind),
        'reconstruction_weight': .1, 'statistical_weight': .1, 'tendency_weight': .1,
        'supervision': mode, 'latent_weight': .1 if mode == 'field_latent' else 0.,
        'target_encoder': 'online_stop_gradient',
        'target_semantics': 'same_length_strided_truth_window',
    }


def _predictor_config(architecture='latent_rollout'):
    return PredictorConfig(
        training_mode='joint', latent_layout='spatial', dynamics_encoder='conv_gru',
        dynamics_hidden_dim=4, dynamics_channels=2,
        dynamics_auxiliary_architecture=architecture)


def _toy(kind='conv_gru', stride=2):
    torch.manual_seed(19)
    raw_grid, latent_grid, history_steps, step_hours = (2, 4, 6), (2, 2, 3), 3, 6.
    pipe = nn.Module()
    pipe.config = _predictor_config()
    pipe.a_config = SimpleNamespace(grid=raw_grid, history_steps=history_steps,
                                    history_stride=stride, step_hours=step_hours)
    pipe.dynamics_encoder = DynamicsGuide(
        raw_grid, latent_grid, history_steps, hidden=4, kind=kind,
        history_dt_hours=stride*step_hours)
    pipe.dynamics_auxiliary_head = DynamicsLatentRolloutHead(
        latent_grid, raw_grid, hidden=4, step_hours=step_hours)
    temporal = SimpleNamespace(metric=torch.full((48,), 1/48),
                               area=torch.full((4, 6), 1/24),
                               scale=torch.ones(48), tendency_scale=torch.ones(48))
    pipe.bridge = SimpleNamespace(manifold=SimpleNamespace(temporal=temporal))
    origin = (history_steps-1)*stride
    sequence = torch.randn(2, origin+9, 48)
    batch = {
        'history': sequence[:, :origin+1:stride].clone(),
        'origin': sequence[:, origin].clone(),
        'targets': sequence[:, origin+1:].clone(),
        'dynamics_target_sequence': sequence.clone(),
    }
    return pipe, batch, torch.tensor([6., 18., 48.])


def _gradient(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0


def test_rollout_advances_every_archive_step_in_physical_days_then_decodes():
    class ConstantRate(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, state_dt):
            self.calls += 1
            torch.testing.assert_close(state_dt[:, -1], torch.full_like(state_dt[:, -1], .25))
            return torch.ones_like(state_dt[:, :-1])*2

    head = DynamicsLatentRolloutHead((2, 2, 3), (2, 2, 3), hidden=4, step_hours=6.)
    rate = ConstantRate()
    head.transition, head.decoder = rate, nn.Identity()
    history = torch.randn(2, 3, 12)
    output = head.rollout(history, torch.tensor([6., 18., 48.]))
    assert rate.calls == 8  # No separate restart per lead or missing intermediate steps.
    expected = history[:, -1, None] + torch.tensor([.5, 1.5, 4.])[None, :, None]
    torch.testing.assert_close(output['latent'], expected)
    torch.testing.assert_close(output['fields'], output['latent'])


def test_rollout_uses_own_predicted_state_and_generalizes_requested_horizon():
    class StateRate(nn.Module):
        def forward(self, state_dt):
            return state_dt[:, :-1]

    head = DynamicsLatentRolloutHead((1, 2, 2), (1, 4, 4), hidden=4, step_hours=12.)
    head.transition = StateRate()
    history = torch.ones(2, 3, 4, requires_grad=True)
    result = head.rollout(history, torch.tensor([12., 36., 72.]))
    torch.testing.assert_close(result['latent'],
                               torch.tensor([1.5, 1.5**3, 1.5**6])[None, :, None].expand(2, 3, 4))
    torch.testing.assert_close(result['fields'], head.decode(result['latent']), rtol=0, atol=0)
    result['latent'][:, -1].sum().backward()
    torch.testing.assert_close(history.grad[:, -1], torch.full((2, 4), 1.5**6))
    assert torch.count_nonzero(history.grad[:, :-1]) == 0
    with torch.no_grad():
        longer = head.rollout(history, torch.tensor([12., 72., 120.]))
    torch.testing.assert_close(longer['latent'][:, 1], result['latent'][:, -1])


@pytest.mark.parametrize('leads', [[0.], [-6.], [7.], [12., 6.], [6., 6.], [float('nan')]])
def test_rollout_rejects_invalid_or_unaligned_leads(leads):
    head = DynamicsLatentRolloutHead((1, 2, 2), (1, 2, 2), hidden=2, step_hours=6.)
    with pytest.raises(ValueError, match='lead|Lead'):
        head.rollout(torch.zeros(1, 2, 4), torch.tensor(leads))


@pytest.mark.parametrize('kind', ['conv_gru', 'temporal_conv'])
@pytest.mark.parametrize('stride', [1, 3])
def test_teacher_reencodes_same_length_strided_truth_windows_without_gradient(kind, stride):
    pipe, batch, leads = _toy(kind, stride)
    batch['dynamics_target_sequence'].requires_grad_(True)
    teacher = future_dynamics_targets(pipe, batch, leads)
    expected = []
    indices = torch.arange(3)*stride
    for step in (1, 3, 8):
        expected.append(pipe.dynamics_encoder(batch['dynamics_target_sequence'][:, indices+step])[:, -1])
    torch.testing.assert_close(teacher, torch.stack(expected, 1), rtol=0, atol=0)
    assert not teacher.requires_grad and teacher.grad_fn is None
    assert batch['dynamics_target_sequence'].grad is None
    assert all(p.grad is None for p in pipe.dynamics_encoder.parameters())


def test_teacher_rejects_misaligned_label_spans():
    pipe, batch, leads = _toy()
    for index, pattern in ((0, 'prefix'), (-1, 'suffix')):
        broken = {**batch, 'dynamics_target_sequence': batch['dynamics_target_sequence'].clone()}
        broken['dynamics_target_sequence'][:, index] += 1
        with pytest.raises(ValueError, match=pattern):
            future_dynamics_targets(pipe, broken, leads)
    with pytest.raises(ValueError, match='span'):
        future_dynamics_targets(pipe, {**batch, 'dynamics_target_sequence': batch['dynamics_target_sequence'][:, :-1]}, leads)


@pytest.mark.parametrize('kind', ['w2', 'kl_entropy', 'signed_measure'])
def test_hybrid_loss_reaches_encoder_transition_decoder_but_not_target_inputs(kind):
    pipe, batch, leads = _toy()
    batch['history'].requires_grad_(True)
    batch['targets'].requires_grad_(True)
    batch['dynamics_target_sequence'].requires_grad_(True)
    prediction = {'dynamics_history_latent': pipe.dynamics_encoder(batch['history'])}
    losses = dynamics_auxiliary_losses(pipe, prediction, batch, leads, _config('field_latent', kind))
    assert losses['dynamics_latent'] > 0
    assert losses['dynamics_target_latent_variance'] > 0
    losses['dynamics_regularization'].backward()
    for component in (pipe.dynamics_encoder, pipe.dynamics_auxiliary_head.transition,
                      pipe.dynamics_auxiliary_head.decoder):
        _gradient(component)
    assert batch['history'].grad.abs().sum() > 0
    assert batch['targets'].grad is batch['dynamics_target_sequence'].grad is None


def test_field_arm_never_reads_target_encoder_labels_and_hybrid_adds_exact_latent_term():
    pipe, batch, leads = _toy()
    prediction = {'dynamics_history_latent': pipe.dynamics_encoder(batch['history'])}
    field = dynamics_auxiliary_losses(pipe, prediction, batch, leads, _config())
    poison = {**batch, 'dynamics_target_sequence': torch.full_like(batch['dynamics_target_sequence'], float('nan'))}
    independent = dynamics_auxiliary_losses(pipe, prediction, poison, leads, _config())
    for name in field:
        torch.testing.assert_close(independent[name], field[name], rtol=0, atol=0)
    hybrid = dynamics_auxiliary_losses(pipe, prediction, batch, leads, _config('field_latent'))
    torch.testing.assert_close(hybrid['dynamics_regularization'],
                               field['dynamics_regularization']+.1*hybrid['dynamics_latent'])
    for name in ('dynamics_reconstruction', 'dynamics_statistical', 'dynamics_tendency'):
        torch.testing.assert_close(hybrid[name], field[name], rtol=0, atol=0)


def test_collapse_diagnostic_uses_across_states_variance_not_spatial_pattern_variance():
    same_state = torch.arange(12.).expand(2, 3, 12)
    scores = dynamics_latent_diagnostics(same_state.clone().requires_grad_(), same_state)
    assert scores['dynamics_target_latent_variance'] == 0
    assert scores['dynamics_target_latent_rms'] > 0
    assert scores['dynamics_latent'] == 0
    assert all(not score.requires_grad for name, score in scores.items() if name != 'dynamics_latent')


def test_v1_config_and_contract_remain_exact_and_v2_requires_grounded_field_losses():
    direct = _predictor_config('direct')
    old = dynamics_auxiliary_config_from_args(SimpleNamespace(), direct)
    assert set(old) == {'version', 'statistical_loss_config', 'reconstruction_weight',
                        'statistical_weight', 'tendency_weight'}
    assert old['version'] == 'climate_manifold.dynamics_auxiliary.v1'
    assert dynamics_contract(direct)['version'] == 'climate_manifold.dual_dynamics.v1'
    rollout = _predictor_config()
    field = dynamics_auxiliary_config_from_args(SimpleNamespace(), rollout)
    hybrid = dynamics_auxiliary_config_from_args(SimpleNamespace(dynamics_supervision='field_latent'), rollout)
    assert field['latent_weight'] == 0 and hybrid['latent_weight'] == .1
    for config in (field, hybrid):
        for weight in ('reconstruction_weight', 'statistical_weight'):
            with pytest.raises(ValueError, match='positive field'):
                validate_dynamics_auxiliary_config({**config, weight: 0.})
    for args in (SimpleNamespace(dynamics_supervision='field_latent'),
                 SimpleNamespace(dynamics_latent_weight=0.)):
        with pytest.raises(ValueError, match='latent_rollout'):
            dynamics_auxiliary_config_from_args(args, direct)
    with pytest.raises(ValueError, match='only valid'):
        dynamics_auxiliary_config_from_args(SimpleNamespace(dynamics_latent_weight=.1), rollout)
    with pytest.raises(ValueError, match='version disagrees'):
        dynamics_auxiliary_config_from_payload({'dynamics_contract': dynamics_contract(rollout),
                                                'dynamics_auxiliary_config': old}, rollout)
    with pytest.raises(ValueError, match='enabled dynamics'):
        PredictorConfig(dynamics_auxiliary_architecture='latent_rollout')


def test_supervision_options_do_not_change_network_initialization_or_rng():
    states = []
    for supervision in ('field', 'field_latent'):
        torch.manual_seed(31)
        config = _predictor_config()
        dynamics_auxiliary_config_from_args(SimpleNamespace(dynamics_supervision=supervision), config)
        encoder = DynamicsGuide((2, 4, 6), (2, 2, 3), 3, hidden=4)
        head = DynamicsLatentRolloutHead((2, 2, 3), (2, 4, 6), hidden=4)
        states.append((deepcopy(encoder.state_dict()), deepcopy(head.state_dict()), torch.random.get_rng_state()))
    for first, second in zip(states[0][:2], states[1][:2]):
        assert first.keys() == second.keys()
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    torch.testing.assert_close(states[0][2], states[1][2], rtol=0, atol=0)
