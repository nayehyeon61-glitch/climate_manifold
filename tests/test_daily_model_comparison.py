"""The daily model matrix substitutes M; it never stacks M with Transformer."""
import json

import pytest
import torch
from torch.utils.data import DataLoader

from test_daily_era5 import daily_prepared, _args
from test_spatial_training import _capture, _assert_finite_gradient
from climate_manifold.downstream.train import train, load_predictor, initialize_manifold, windows
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.compare import compare


FAMILIES = ('transformer', 'mlp', 'neural_ode', 'climode', 'convlstm', 'simvp', 'fourcastnet', 'climax')


def test_daily_raw_latent_model_and_loss_matrix(daily_prepared, tmp_path, monkeypatch):
    """Real optimization, gradients, reload, validation/test, and paired scores."""
    _, archive, info, _ = daily_prepared
    captured = _capture(monkeypatch)
    reports, tests = [], []
    for family in FAMILIES:
        latent_initial = None
        for bridge, loss in (('raw', None), ('latent', 'w2'), ('latent', 'signed_measure')):
            name = f'{family}-{bridge}-{loss}'
            args = _args(archive, info, tmp_path/(name+'.pt'), bridge)
            args.model = family
            args.climode_step_hours = 6.
            if loss:
                args.statistical_loss = loss
            capture_index = len(captured)
            checkpoint = train(args)
            trained, initial = captured[capture_index]
            _assert_finite_gradient(trained.predictor)
            if bridge == 'latent':
                manifold = trained.bridge.manifold
                for component in (manifold.core.manifold.encoder, manifold.core.manifold.decoder,
                                  manifold.info_head, trained.reconstruction_decoder):
                    _assert_finite_gradient(component)
                # Changing only the statistical objective retains identical
                # initialization of E, M, D, and the auxiliary decoders.
                if latent_initial is None:
                    latent_initial = initial
                else:
                    assert initial.keys() == latent_initial.keys()
                    for key in initial:
                        torch.testing.assert_close(initial[key], latent_initial[key], rtol=0, atol=0)
            restored, payload = load_predictor(checkpoint)
            assert restored.config.model == family and restored.config.bridge == bridge
            expected = 'encoder -> predictor -> decoder' if bridge == 'latent' else 'observations -> field predictor'
            assert payload['experiment']['path'] == expected
            assert payload['lead_hours'] == [24., 48.]
            assert payload['objective_weights']['pinn'] == payload['objective_weights']['static'] == 0
            assert restored.conditional_flow is None
            assert payload['guide_contract'] is None
            _, _, data = initialize_manifold(args)
            batch = next(iter(DataLoader(windows(data, restored.a_config, 'validation', max_windows=2), batch_size=2)))
            inputs = (batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([24., 48.]))
            trained.eval()
            with torch.no_grad():
                before, after = trained(*inputs), restored(*inputs)
                torch.testing.assert_close(before['mean'], after['mean'], rtol=0, atol=0)
                assert after['mean'].shape == batch['targets'][:, :2].shape
                assert (after['predicted_latent'] is not None) == (bridge == 'latent')
                batch['targets'].fill_(float('nan'))
                torch.testing.assert_close(restored(*inputs)['mean'], after['mean'], rtol=0, atol=0)
            path = tmp_path/(name+'.validation.json')
            report = evaluate(checkpoint, archive, path, information=info, max_cases=2)
            assert report['finite_forecast_fraction'] == 1. and not report['failed_origins']
            reports.append(path)
            tests.append((checkpoint, tmp_path/(name+'.test.json'), report))

    validation = compare(reports, tmp_path/'comparison.validation.json')
    assert validation['ranking_allowed'] and len(validation['seed_summary']) == 3*len(FAMILIES)
    assert {row['model'] for row in validation['direct_comparison']['effects']} == set(FAMILIES)
    assert (tmp_path/'comparison.validation.raw-effects.csv').is_file()

    # Test is evaluated only after every checkpoint is fixed.
    for checkpoint, path, validation_report in tests:
        report = evaluate(checkpoint, archive, path, information=info, split='test', max_cases=2)
        assert report['finite_forecast_fraction'] == 1.
        assert not set(report['origin_times']) & set(validation_report['origin_times'])
        assert report['checkpoint_sha256'] == validation_report['checkpoint_sha256']
    test = compare([p for _, p, _ in tests], tmp_path/'comparison.test.json')
    assert test['ranking_allowed'] and len(test['seed_summary']) == 3*len(FAMILIES)


@pytest.mark.parametrize('family', FAMILIES)
def test_daily_models_still_reject_pinn(daily_prepared, tmp_path, family):
    _, archive, info, _ = daily_prepared
    args = _args(archive, info, tmp_path/'unused.pt', 'latent')
    args.model = family
    args.constraint_pair = 'pinn_statistical'
    args.pinn = True
    with pytest.raises(ValueError, match='Daily support requires'):
        initialize_manifold(args)
