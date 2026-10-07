"""Auxiliary latent-rollout audits preserve causal forecasts and held-out boundaries."""
from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from test_dual_dynamics_training import dual_args
from climate_manifold.downstream.dynamics_diagnostics import (
    DynamicsLatentDiagnostics, SignedFieldDiagnostics, dynamics_evaluation_windows)
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import train, load_predictor


def rollout_checkpoint(prepared, root, supervision='field', bridge='guided'):
    archive = prepared[-1]
    args = dual_args(archive, root, bridge, 'conv_gru',
                    '--dynamics-auxiliary-architecture', 'latent_rollout',
                    '--dynamics-supervision', supervision)
    args.output = str(root / (bridge+'-'+supervision+'.pt'))
    return train(args)


def test_extended_window_selection_preserves_original_split_labels_before_stride(pinn_prepared):
    model, _, data, _ = pinn_prepared
    config = replace(model.config, history_stride=2, history_steps=3)
    # A deliberately explicit split verifies exact indexing rather than relying
    # on build_split to happen to leave a sufficiently large boundary gap.
    data = {**data, 'split': {**data['split'], 'test': list(range(150, 161))}}
    ds, metadata = dynamics_evaluation_windows(data, config, 'test', 24, stride=3, max_cases=2)
    assert ds.starts == [150, 153]
    assert ds.config.horizon_steps == 24 and config.horizon_steps == 20
    assert data['split']['test'] == list(range(150, 161))
    assert metadata['horizon_eligible_origin_count'] == 7
    assert metadata['excluded_for_rollout_coverage_count'] == 4
    assert metadata['last_allowed_label_index'] == 160+config.history_span_steps+20-1
    sample = ds[0]
    seq = sample['dynamics_target_sequence']
    torch.testing.assert_close(seq[::2][:3], sample['history'], atol=0, rtol=0)
    torch.testing.assert_close(seq[config.history_span_steps:], sample['targets'], atol=0, rtol=0)
    with pytest.raises(ValueError, match='no origins'):
        dynamics_evaluation_windows(data, config, 'test', 100)
    for wrong in (19, 20.0, True):
        with pytest.raises(ValueError, match='integer'):
            dynamics_evaluation_windows(data, config, 'test', wrong)


def test_latent_variance_is_across_origins_not_spatial_coordinates():
    fixed = torch.tensor([[[0., 100., -100.], [5., 105., -95.]]])
    metric = DynamicsLatentDiagnostics([6, 12])
    metric.update(fixed, fixed, fixed[:, 0])
    metric.update(fixed, fixed, fixed[:, 0])
    result = metric.result()
    assert result['aggregate']['target_across_origin_coordinate_variance'] == 0
    assert result['aggregate']['target_latent_rms'] > 50
    assert result['aggregate']['variance_floor_active']
    assert result['aggregate']['latent_rmse'] == 0
    # Shift every fixed coordinate by +/-1 across origins: population variance 1.
    target = torch.cat((fixed-1, fixed+1))
    stream, pooled = DynamicsLatentDiagnostics([6, 12]), DynamicsLatentDiagnostics([6, 12])
    pooled.update(target+2, target, target[:, 0])
    for row in target:
        stream.update(row[None]+2, row[None], row[None, 0])
    assert stream.result() == pooled.result()
    assert pooled.result()['aggregate']['target_across_origin_coordinate_variance'] == 1
    assert pooled.result()['aggregate']['normalized_latent_rmse'] == 2


def test_signed_audit_retains_location_and_mass(pinn_prepared):
    model, batch, data, _ = pinn_prepared
    target = batch['targets'][:, :2]
    metrics = SignedFieldDiagnostics(data['schema'], [6, 12])
    metrics.update(target, target)
    result = metrics.result()
    assert result['case_count'] == 2
    assert result['aggregate']['signed_spatial_js'] == 0
    assert result['aggregate']['signed_mass_mse'] == 0
    assert set(result['per_variable']) == {'msl', 't2m', 'u10', 'v10'}


@pytest.mark.parametrize('supervision,bridge', [('field','guided'), ('field_latent','latent')])
def test_rollout_audit_exports_separate_auxiliary_physical_scores_and_latent_targets(
        pinn_prepared, tmp_path, supervision, bridge):
    checkpoint = rollout_checkpoint(pinn_prepared, tmp_path, supervision, bridge)
    report = evaluate(checkpoint, pinn_prepared[-1], tmp_path/'audit.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2,
                      dynamics_diagnostics=True, dynamics_rollout_steps=4,
                      forecast_output=tmp_path/'audit.npz')
    assert report['lead_hours'] == [6., 12.]
    assert report['dynamics_evaluation_contract'] == {
        'version':'climate_manifold.dynamics_evaluation.v1', 'rollout_steps':4,
        'step_hours':6., 'target_semantics':'same_length_strided_truth_window',
        'main_forecast_horizon_steps':2}
    audit = report['dynamics_auxiliary_evaluation']
    assert audit['lead_hours'] == [6., 12., 18., 24.]
    assert audit['successful_case_count'] == audit['attempted_case_count'] == 2
    assert audit['successful_origin_times'] == report['successful_origin_times']
    assert audit['scores']['case_count'] == audit['latent_diagnostics']['case_count'] == 2
    assert not audit['teacher_forcing'] and not audit['downstream_forecast_model_used']
    assert report['main_forecast_signed_diagnostics']['case_count'] == 2
    assert audit['signed_diagnostics']['case_count'] == 2
    assert audit['diagnostic_seconds'] > 0 and report['inference_seconds'] > 0
    assert audit['cuda_peak_memory_bytes'] is None
    with np.load(tmp_path/'audit.npz') as saved:
        assert saved['mean'].shape[0] == 2
        assert saved['auxiliary_dynamics_surface_fields'].shape[0] == 4
        assert saved['auxiliary_dynamics_teacher_latent'].shape == saved['auxiliary_dynamics_predicted_latent'].shape
        assert np.isfinite(saved['auxiliary_dynamics_predicted_latent']).all()


def test_default_evaluation_never_calls_auxiliary_head_and_diagnostics_validation(
        pinn_prepared, tmp_path, monkeypatch):
    checkpoint = rollout_checkpoint(pinn_prepared, tmp_path)
    model, payload = load_predictor(checkpoint)
    def forbidden(*args, **kwargs):
        raise AssertionError('Auxiliary rollout must not run during ordinary M evaluation')
    monkeypatch.setattr(model.dynamics_auxiliary_head, 'rollout', forbidden)
    module = importlib.import_module('climate_manifold.downstream.evaluate')
    monkeypatch.setattr(module, 'load_predictor', lambda *args: (model, payload))
    result = evaluate(checkpoint, pinn_prepared[-1], tmp_path/'ordinary.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2)
    assert 'dynamics_evaluation_contract' not in result
    assert 'dynamics_auxiliary_evaluation' not in result
    with pytest.raises(ValueError, match='requires --dynamics-diagnostics'):
        evaluate(checkpoint, pinn_prepared[-1], tmp_path/'invalid.json', dynamics_rollout_steps=3)
    # A historical direct head remains usable by the ordinary evaluator, but
    # cannot be described as an autoregressive latent rollout.
    model.config = replace(model.config, dynamics_auxiliary_architecture='direct')
    with pytest.raises(ValueError, match='latent_rollout'):
        evaluate(checkpoint, pinn_prepared[-1], tmp_path/'invalid.json', dynamics_diagnostics=True)


def test_auxiliary_nonfinite_failure_does_not_mark_main_forecast_failed(
        pinn_prepared, tmp_path, monkeypatch):
    checkpoint = rollout_checkpoint(pinn_prepared, tmp_path)
    model, payload = load_predictor(checkpoint)
    def broken(*args):
        raise FloatingPointError('unstable auxiliary recurrence')
    monkeypatch.setattr(model.dynamics_auxiliary_head, 'rollout', broken)
    module = importlib.import_module('climate_manifold.downstream.evaluate')
    monkeypatch.setattr(module, 'load_predictor', lambda *args: (model, payload))
    result = evaluate(checkpoint, pinn_prepared[-1], tmp_path/'failure.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2,
                      dynamics_diagnostics=True)
    assert result['finite_forecast_fraction'] == 1 and not result['failed_origins']
    assert result['scores']['case_count'] == 2
    aux = result['dynamics_auxiliary_evaluation']
    assert aux['finite_auxiliary_fraction'] == 0 and aux['successful_case_count'] == 0
    assert len(aux['auxiliary_failed_origins']) == 2 and aux['scores']['case_count'] == 0
    assert result['main_forecast_signed_diagnostics']['case_count'] == 2


def test_field_and_hybrid_use_identical_extended_origins(pinn_prepared, tmp_path):
    reports = []
    for supervision in ('field', 'field_latent'):
        checkpoint = rollout_checkpoint(pinn_prepared, tmp_path, supervision)
        reports.append(evaluate(checkpoint, pinn_prepared[-1], tmp_path/(supervision+'.json'),
                                information=tmp_path/'pinn-information.npz', max_cases=2,
                                origin_stride=2, dynamics_diagnostics=True, dynamics_rollout_steps=5))
    assert reports[0]['origin_times'] == reports[1]['origin_times']
    assert reports[0]['dynamics_origin_selection'] == reports[1]['dynamics_origin_selection']
    assert reports[0]['dynamics_evaluation_contract'] == reports[1]['dynamics_evaluation_contract']


def test_future_latent_teacher_changes_only_audit_not_main_forecast(pinn_prepared, tmp_path, monkeypatch):
    checkpoint = rollout_checkpoint(pinn_prepared, tmp_path)
    module = importlib.import_module('climate_manifold.downstream.evaluate')
    original = module.future_dynamics_targets
    reports = []
    for index in range(2):
        if index:
            monkeypatch.setattr(module, 'future_dynamics_targets',
                                lambda *args: original(*args)+50)
        reports.append(evaluate(checkpoint, pinn_prepared[-1], tmp_path/f'teacher-{index}.json',
                                information=tmp_path/'pinn-information.npz', max_cases=2,
                                dynamics_diagnostics=True, forecast_output=tmp_path/f'teacher-{index}.npz'))
    assert reports[0]['scores'] == reports[1]['scores']
    assert (reports[0]['dynamics_auxiliary_evaluation']['scores']
            == reports[1]['dynamics_auxiliary_evaluation']['scores'])
    for key in ('mean', 'auxiliary_dynamics_predicted_latent', 'auxiliary_dynamics_surface_fields'):
        with np.load(tmp_path/'teacher-0.npz') as first, np.load(tmp_path/'teacher-1.npz') as second:
            np.testing.assert_array_equal(first[key], second[key])
    assert (reports[0]['dynamics_auxiliary_evaluation']['latent_diagnostics']['aggregate']['latent_rmse']
            != reports[1]['dynamics_auxiliary_evaluation']['latent_diagnostics']['aggregate']['latent_rmse'])
