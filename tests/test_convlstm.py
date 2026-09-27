"""Contracts for the time-conditioned, grid-preserving ConvLSTM predictor."""
import math

import pytest
import torch
from torch import nn

from climate_manifold.downstream.convlstm import ConvLSTMPredictor


@pytest.fixture(autouse=True)
def deterministic_small_cpu_test():
    torch.manual_seed(71)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(grid=(2, 3, 5), history_steps=3, batch=2):
    return (torch.randn(batch, history_steps, math.prod(grid)),
            torch.tensor([6., 12., 24.]),
            torch.tensor([1_577_836_800_000_000_000] * batch))


@pytest.mark.parametrize('grid,periodic', [((2, 9, 18), False), ((2, 8, 16), True),
                                        ((1, 1, 1), True), ((1, 1, 3), False)])
def test_grid_shape_and_gradient_reach_all_history_and_parameters(grid, periodic):
    model = ConvLSTMPredictor(grid, 3, hidden=4, periodic_lon=periodic)
    history, leads, origin = sample(grid)
    history.requires_grad_()
    output, std = model(history, leads, origin)
    assert output.shape == (2, 3, math.prod(grid)) and std is None
    assert not any(isinstance(module, nn.Linear) for module in model.modules())
    output.square().mean().backward()
    assert torch.isfinite(history.grad).all()
    assert (history.grad.abs().sum(dim=(0, 2)) > 0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0 for p in model.parameters())


def test_rollout_prefix_train_eval_consistency_and_no_teacher_forcing():
    model = ConvLSTMPredictor((2, 3, 5), 3, hidden=4)
    history, leads, origin = sample()
    model.train()
    full = model(history, leads, origin)[0]
    prefix = model(history, leads[:2], origin)[0]
    torch.testing.assert_close(full[:, :2], prefix, rtol=0, atol=0)
    model.eval()
    torch.testing.assert_close(model(history, leads, origin)[0], full, rtol=0, atol=0)


def test_history_is_consumed_once_then_predictions_are_fed_back_with_actual_times():
    model = ConvLSTMPredictor((2, 3, 5), 3, hidden=4, history_dt_hours=24.)
    history, leads, origin = sample()
    inputs = []
    handle = model.cell.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach().clone()))
    result = model(history, leads, origin)[0]
    handle.remove()
    assert len(inputs) == 3 + len(leads) - 1
    for index in range(3):
        torch.testing.assert_close(inputs[index][:, :2].flatten(1), history[:, index])
    for index in range(len(leads) - 1):
        torch.testing.assert_close(inputs[3 + index][:, :2].flatten(1), result[:, index])
    # Channel order: state C=2, four calendar fields, relative days, dt days.
    assert [entry[0, 6, 0, 0].item() for entry in inputs] == [-2., -1., 0., .25, .5]
    assert [entry[0, 7, 0, 0].item() for entry in inputs] == [0., 1., 1., .25, .25]


def test_elapsed_hours_scale_forecast_rates_instead_of_counting_recurrent_steps():
    model = ConvLSTMPredictor((2, 3, 5), 3, hidden=4)
    class UnitRate(nn.Module):
        def forward(self, features):
            return torch.ones_like(features[:, :2])
    model.readout = UnitRate()
    history, _, origin = sample()
    leads = torch.tensor([3., 9., 27.])
    result = model(history, leads, origin)[0]
    expected = history[:, -1, None] + leads[None, :, None] / 24.
    torch.testing.assert_close(result, expected)


def test_history_spacing_and_past_frames_change_forecast():
    day_model = ConvLSTMPredictor((2, 3, 5), 3, hidden=4, history_dt_hours=24.)
    six_hour_model = ConvLSTMPredictor((2, 3, 5), 3, hidden=4, history_dt_hours=6.)
    six_hour_model.load_state_dict(day_model.state_dict())
    history, leads, origin = sample()
    output = day_model(history, leads, origin)[0]
    assert not torch.allclose(output, six_hour_model(history, leads, origin)[0])
    changed = history.clone()
    changed[:, 0] += 5
    assert not torch.allclose(output, day_model(changed, leads, origin)[0])
    assert not torch.allclose(output, day_model(history, leads, origin + 6 * 3_600_000_000_000)[0])


def test_raw_origin_information_is_used_but_future_sequence_and_latent_bypass_rejected():
    grid = (2, 3, 5)
    model = ConvLSTMPredictor(grid, 3, hidden=4, information_channels=2)
    history, leads, origin = sample()
    information = torch.randn(2, 30, requires_grad=True)
    output = model(history, leads, origin, information)[0]
    output.square().mean().backward()
    assert torch.isfinite(information.grad).all() and information.grad.abs().sum() > 0
    assert not torch.allclose(output, model(history, leads, origin, information + 10)[0])
    for invalid in (None, information[:, None], information[:, :-1], information * float('nan')):
        with pytest.raises(ValueError, match='origin information'):
            model(history, leads, origin, invalid)
    latent = ConvLSTMPredictor(grid, 3, hidden=4)
    with pytest.raises(ValueError, match='encoder only'):
        latent(history, leads, origin, information)


@pytest.mark.parametrize('leads', [[], [0.], [-6.], [12., 6.], [6., 6.],
                                 [float('nan')], [float('inf')]])
def test_invalid_leads_rejected(leads):
    model = ConvLSTMPredictor((2, 3, 5), 3, hidden=4)
    history, _, origin = sample()
    with pytest.raises(ValueError, match='Lead hours'):
        model(history, torch.tensor(leads), origin)


@pytest.mark.parametrize('change', ['short', 'flat', 'nan', 'integer', 'empty', 'origin_shape', 'origin_nan'])
def test_invalid_history_and_origin_rejected(change):
    model = ConvLSTMPredictor((2, 3, 5), 3, hidden=4)
    history, leads, origin = sample()
    if change == 'short':
        history = history[:, :2]
    elif change == 'flat':
        history = history.flatten(1)
    elif change == 'nan':
        history[0, 0, 0] = float('nan')
    elif change == 'integer':
        history = history.long()
    elif change == 'empty':
        history = history[:0]
    elif change == 'origin_shape':
        origin = origin[:, None]
    else:
        origin = origin.float() * float('nan')
    with pytest.raises(ValueError):
        model(history, leads, origin)


@pytest.mark.parametrize('kwargs', [
    {'latent_grid': (2, 3)}, {'latent_grid': (2, 0, 3)}, {'latent_grid': (2, 3., 5)},
    {'history_steps': 0}, {'history_steps': True}, {'hidden': -1},
    {'history_dt_hours': 0.}, {'history_dt_hours': float('inf')},
    {'history_dt_hours': float('nan')}, {'information_channels': -1},
    {'information_channels': True},
])
def test_invalid_configuration_rejected(kwargs):
    config = {'latent_grid': (2, 3, 5), 'history_steps': 3, 'hidden': 4}
    config.update(kwargs)
    with pytest.raises(ValueError):
        ConvLSTMPredictor(**config)
