import math

import pytest
import torch
from torch import nn

from climate_manifold.downstream.guided_transformer import GuidedTransformerPredictor


def _model(**kwargs):
    config = dict(raw_grid=(2, 5, 7), guide_grid=(3, 3, 4), history_steps=3,
                  hidden=16, depth=1, patch_size=2, history_dt_hours=6.)
    config.update(kwargs)
    return GuidedTransformerPredictor(**config)


def test_observed_guide_receives_forecast_gradients_on_irregular_grids():
    torch.manual_seed(17)
    predictor = _model(information_channels=1)
    history = torch.randn(2, 3, 70, requires_grad=True)
    encoder = nn.Linear(70, 36)
    guide = encoder(history)
    guide.retain_grad()
    information = torch.randn(2, 35, requires_grad=True)
    output, std = predictor(history, torch.tensor([6., 24., 72.]), torch.tensor([0, 1]),
                            information, guide_history=guide)
    assert output.shape == (2, 3, 70)
    assert std is None and torch.isfinite(output).all()
    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    assert encoder.weight.grad.abs().sum() > 0
    assert guide.grad[:, :-1].abs().sum() > 0
    assert history.grad[:, :-1].abs().sum() > 0
    assert information.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


def test_guide_history_order_lead_query_and_checkpoint_roundtrip():
    torch.manual_seed(19)
    predictor = _model().eval()
    history, guide = torch.randn(2, 3, 70), torch.randn(2, 3, 36)
    leads = torch.tensor([6., 24.])
    first, _ = predictor(history, leads, guide_history=guide)
    changed_guide, _ = predictor(history, leads, guide_history=guide + 2.)
    assert not torch.allclose(first, changed_guide)
    # Keep the latest raw state fixed: the difference must come from history.
    reordered = history[:, [1, 0, 2]]
    second, _ = predictor(reordered, leads, guide_history=guide[:, [1, 0, 2]])
    assert not torch.allclose(first, second)
    singleton, _ = predictor(history, leads[-1:], guide_history=guide)
    torch.testing.assert_close(first[:, -1:], singleton)
    clone = _model().eval()
    clone.load_state_dict(predictor.state_dict())
    torch.testing.assert_close(first, clone(history, leads, guide_history=guide)[0])


def test_zero_guide_is_matched_predictor_without_input_dependent_guidance():
    torch.manual_seed(7)
    learned, zero = _model().eval(), _model(guide_mode='zero').eval()
    zero.load_state_dict(learned.state_dict())
    history, guide = torch.randn(2, 3, 70), torch.randn(2, 3, 36)
    leads = torch.tensor([6., 18.])
    assert sum(p.numel() for p in zero.parameters()) == sum(p.numel() for p in learned.parameters())
    expected = learned(history, leads, guide_history=torch.zeros_like(guide))[0]
    torch.testing.assert_close(zero(history, leads, guide_history=guide)[0], expected)
    torch.testing.assert_close(zero(history, leads)[0], expected)


@pytest.mark.parametrize('grid,patch_size', [((1, 1, 1), 4), ((2, 3, 5), 2)])
def test_raw_only_padding_and_periodic_edge_grids(grid, patch_size):
    predictor = _model(raw_grid=grid, guide_grid=None, patch_size=patch_size, periodic_lon=True)
    history = torch.randn(2, 3, math.prod(grid))
    forecast, _ = predictor(history, torch.tensor([6.]))
    assert forecast.shape == (2, 1, math.prod(grid))
    assert torch.isfinite(forecast).all()


def test_future_guide_or_information_sequence_is_rejected():
    predictor = _model()
    history = torch.zeros(2, 3, 70)
    with pytest.raises(ValueError, match='observed history'):
        predictor(history, torch.tensor([6.]), guide_history=torch.zeros(2, 4, 36))
    with pytest.raises(ValueError, match='requires observed'):
        predictor(history, torch.tensor([6.]))
    with pytest.raises(ValueError, match='Raw-only'):
        _model(guide_grid=None)(history, torch.tensor([6.]), guide_history=torch.zeros(2, 3, 36))
    with pytest.raises(ValueError, match='origin information'):
        _model(information_channels=1)(history, torch.tensor([6.]),
                                      information=torch.zeros(2, 3, 35),
                                      guide_history=torch.zeros(2, 3, 36))
