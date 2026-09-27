"""CLI, joint optimization and standalone checkpoint gates for split constraints."""
from dataclasses import asdict
import json

import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_joint_training import _sealed_a_checkpoint
from test_spatial_training import _args, _capture, _assert_finite_gradient
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import (
    CONSTRAINT_PAIRS, initialize_manifold, load_predictor, objective_weights,
    prepare_constraint_pair, train, windows,
)


def _split_args(archive, tmp_path, pair='pinn_statistical', *extra):
    return _args(archive, tmp_path/f'{pair}.pt', '--constraint-pair', pair,
                 '--information', str(tmp_path/'pinn-information.npz'), *extra)


@pytest.mark.parametrize('pair', CONSTRAINT_PAIRS)
@pytest.mark.parametrize('family', ['neural_ode', 'climode'])
def test_split_training_joint_gradients_metadata_and_roundtrip(
        pinn_prepared, tmp_path, monkeypatch, pair, family):
    _, _, _, archive = pinn_prepared
    args = _split_args(archive, tmp_path, pair, '--model', family)
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    model, initial = captured[0]
    manifold = model.bridge.manifold
    for prefix, component in (
            ('bridge.manifold.core.manifold.encoder.', manifold.core.manifold.encoder),
            ('bridge.manifold.core.manifold.decoder.', manifold.core.manifold.decoder),
            ('bridge.manifold.info_head.', manifold.info_head),
            ('predictor.', model.predictor)):
        _assert_finite_gradient(component)
        assert any(not torch.equal(value, model.state_dict()[key])
                   for key, value in initial.items() if key.startswith(prefix))
    restored, payload = load_predictor(checkpoint)
    assert payload['constraint_pair'] == pair
    assert payload['constraint_path'] == 'observed_reconstruction'
    assert payload['training_contract']['constraint_path'] == 'observed_reconstruction'
    assert 'constraint_pair' not in payload['training_contract']
    contract = payload['constraint_contract']
    assert contract['version'] == 'climate_manifold.reconstruction_constraints.v1'
    assert contract['observed_pair'] == 'origin-6h,origin'
    assert contract['groups'] == pair.split('_')
    assert contract['pinn_tendency_supervision'] is False
    weights = payload['objective_weights']
    assert weights['reconstruction'] == .1
    assert weights['information'] == weights['physics'] == 0
    for group, name in (('pinn', 'pinn'), ('statistical', 'distribution'), ('static', 'static')):
        assert bool(weights[name]) == (group in pair.split('_'))
    assert (manifold.pinn is not None) == ('pinn' in pair.split('_'))
    assert payload['forecast_parameters'] + payload['constraint_parameters'] == payload['trainable_parameters']
    assert payload['representation_training'] == 'jointly_trained' and not payload['a_frozen']
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
    assert metrics['selection']['state_mse'] == payload['best_selection_state_mse']
    assert 'information_future' not in metrics['train']
    assert metrics['train']['physics'] == metrics['train']['information'] == 0
    if manifold.pinn is not None:
        assert metrics['train']['pinn_tendency'] == 0
        _assert_finite_gradient(manifold.pinn)
    _, _, data = initialize_manifold(args)
    batch = next(iter(DataLoader(windows(data, restored.a_config, 'validation', max_windows=2,
                                        reconstruction_constraints=True), batch_size=2)))
    inputs = (batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6., 12.]))
    model.eval()
    with torch.no_grad():
        expected, actual = model(*inputs)['mean'], restored(*inputs)['mean']
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        # Standalone inference never consumes the auxiliary observations/labels.
        for key in ('constraint_states', 'constraint_information', 'targets'):
            batch[key].fill_(float('nan'))
        torch.testing.assert_close(restored(*inputs)['mean'], actual, rtol=0, atol=0)
    # Model reconstruction relies only on embedded metadata, not input sidecars.
    sidecar = tmp_path/'pinn-information.npz'
    hidden = tmp_path/'temporarily-hidden-information.npz'
    sidecar.rename(hidden)
    try:
        independent, _ = load_predictor(checkpoint)
        with torch.no_grad():
            torch.testing.assert_close(independent(*inputs)['mean'], actual, rtol=0, atol=0)
    finally:
        hidden.rename(sidecar)
    report = evaluate(checkpoint, archive, tmp_path/f'{pair}.evaluation.json',
                      information=sidecar, max_cases=2)
    for key in ('constraint_pair', 'constraint_path', 'constraint_contract',
                'split_objective_weights', 'forecast_parameters', 'constraint_parameters'):
        assert report[key] == payload[key]
    assert report['finite_forecast_fraction'] == 1.


@pytest.mark.parametrize('extra,match', [
    (('--training-mode', 'frozen'), 'requires joint latent'),
    (('--bridge', 'raw'), 'requires joint latent'),
    (('--bridge', 'decoded'), 'requires joint latent'),
    (('--anchor', 'origin'), 'requires joint latent'),
    (('--regularization', 'none'), 'requires joint latent'),
    (('--representation', 'plain_ae'), 'requires joint latent'),
    (('--mode', 'surface'), 'requires enriched'),
    (('--reconstruction-weight', '0'), 'positive finite reconstruction'),
    (('--statistical-weight', '0'), 'positive finite statistical'),
    (('--pinn-weight', '0'), 'positive finite pinn'),
    (('--information-weight', '0'), 'does not use --information-weight'),
    (('--distribution-weight', '.1'), 'does not use --distribution-weight'),
    (('--physics-weight', '0'), 'does not use --physics-weight'),
])
def test_split_cli_rejects_incompatible_or_ambiguous_objectives(tmp_path, extra, match):
    args = _split_args(tmp_path/'absent.npz', tmp_path, 'pinn_statistical', *extra)
    with pytest.raises(ValueError, match=match):
        prepare_constraint_pair(args)


def test_statistical_static_auto_disables_pinn_and_uses_exact_two_groups(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    args = _split_args(archive, tmp_path, 'statistical_static')
    model, metadata, _ = initialize_manifold(args)
    assert model.pinn is None and metadata['pinn_config'] is None and not args.pinn
    weights = asdict(objective_weights(args, model))
    assert weights == dict(reconstruction=.1, physics=0., information=0., static=.05, distribution=.1, pinn=0.)
    for extra in (('--pinn',), ('--pinn-weight', '.1'), ('--static-weight', '0')):
        with pytest.raises(ValueError):
            initialize_manifold(_split_args(archive, tmp_path, 'statistical_static', *extra))


def test_fresh_statistical_static_drops_reference_pinn_but_pretrained_rejects_it(pinn_prepared, tmp_path):
    original, _, data, archive = pinn_prepared
    reference = _sealed_a_checkpoint(original, data, archive, tmp_path/'pinn-information.npz', tmp_path/'a.pt')
    args = _split_args(archive, tmp_path, 'statistical_static', '--a-checkpoint', str(reference))
    fresh, metadata, _ = initialize_manifold(args)
    assert fresh.pinn is None and metadata['pinn_config'] is None
    args.initialization = 'pretrained'
    with pytest.raises(ValueError, match='cannot inherit a pretrained PINN'):
        initialize_manifold(args)


def test_split_training_requires_observed_pair(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    args = _split_args(archive, tmp_path, 'statistical_static', '--history-steps', '1')
    with pytest.raises(ValueError, match='history_span_steps >= 2'):
        initialize_manifold(args)
