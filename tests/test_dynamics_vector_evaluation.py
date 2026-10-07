"""Vector-only audits score latent trajectories without using an untrained decoder."""
import importlib

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from test_dynamics_consistency_evaluation import rollout_checkpoint
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import load_predictor


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_vector_only_audit_never_decodes_and_exports_only_auxiliary_latents(
        pinn_prepared, tmp_path, monkeypatch, bridge):
    checkpoint = rollout_checkpoint(pinn_prepared, tmp_path, 'vector_only', bridge)
    model, payload = load_predictor(checkpoint)
    assert all(not p.requires_grad for p in model.dynamics_auxiliary_head.decoder.parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError('The vector-only D_d was not trained and must never be evaluated')

    monkeypatch.setattr(model.dynamics_auxiliary_head, 'decode', forbidden)
    # Even nonfinite readout weights cannot affect the latent-only audit or M.
    with torch.no_grad():
        for parameter in model.dynamics_auxiliary_head.decoder.parameters():
            parameter.fill_(float('nan'))
    module = importlib.import_module('climate_manifold.downstream.evaluate')
    monkeypatch.setattr(module, 'load_predictor', lambda *args: (model, payload))
    report = evaluate(checkpoint, pinn_prepared[-1], tmp_path/'vector.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2,
                      dynamics_diagnostics=True, dynamics_rollout_steps=4,
                      forecast_output=tmp_path/'vector.npz')
    assert report['finite_forecast_fraction'] == 1.
    assert report['scores']['case_count'] == 2
    assert report['dynamics_decoder_contract'] == payload['dynamics_decoder_contract']
    assert not report['dynamics_decoder_contract']['trained']
    assert report['main_forecast_signed_diagnostics']['case_count'] == 2
    audit = report['dynamics_auxiliary_evaluation']
    assert audit['decoder_status'] == 'untrained_frozen_not_evaluated'
    assert audit['evaluated_outputs'] == ['latent']
    assert audit['path'] == 'E_d_observed_history_to_autoregressive_F_d'
    assert audit['scores'] is audit['signed_diagnostics'] is None
    assert audit['latent_diagnostics']['case_count'] == 2
    assert audit['successful_case_count'] == audit['attempted_case_count'] == 2
    assert audit['finite_auxiliary_fraction'] == 1 and not audit['auxiliary_failed_origins']
    assert audit['successful_origin_times'] == report['successful_origin_times']
    assert audit['lead_hours'] == [6., 12., 18., 24.]
    assert audit['diagnostic_seconds'] > 0 and report['inference_seconds'] > 0
    assert not audit['teacher_forcing'] and not audit['downstream_forecast_model_used']
    with np.load(tmp_path/'vector.npz') as saved:
        assert saved['mean'].shape[0] == 2 and np.isfinite(saved['mean']).all()
        assert 'auxiliary_dynamics_surface_fields' not in saved
        assert 'auxiliary_dynamics_surface_truth' not in saved
        predicted = saved['auxiliary_dynamics_predicted_latent']
        assert predicted.shape[0] == 4 and np.isfinite(predicted).all()
        assert saved['auxiliary_dynamics_teacher_latent'].shape == predicted.shape


def test_vector_teacher_changes_only_latent_audit_not_forecast_or_rollout(
        pinn_prepared, tmp_path, monkeypatch):
    checkpoint = rollout_checkpoint(pinn_prepared, tmp_path, 'vector_only')
    module = importlib.import_module('climate_manifold.downstream.evaluate')
    original = module.future_dynamics_targets
    reports = []
    for index in range(2):
        if index:
            monkeypatch.setattr(module, 'future_dynamics_targets',
                                lambda *args: original(*args)+50)
        reports.append(evaluate(checkpoint, pinn_prepared[-1], tmp_path/f'teacher-{index}.json',
                                information=tmp_path/'pinn-information.npz', max_cases=2,
                                dynamics_diagnostics=True,
                                forecast_output=tmp_path/f'teacher-{index}.npz'))
    assert reports[0]['scores'] == reports[1]['scores']
    assert reports[0]['main_forecast_signed_diagnostics'] == reports[1]['main_forecast_signed_diagnostics']
    with np.load(tmp_path/'teacher-0.npz') as first, np.load(tmp_path/'teacher-1.npz') as second:
        for key in ('mean', 'auxiliary_dynamics_predicted_latent'):
            np.testing.assert_array_equal(first[key], second[key])
    assert (reports[0]['dynamics_auxiliary_evaluation']['latent_diagnostics']['aggregate']['latent_rmse']
            != reports[1]['dynamics_auxiliary_evaluation']['latent_diagnostics']['aggregate']['latent_rmse'])


def test_vector_nonfinite_audit_does_not_mark_main_forecast_failed(
        pinn_prepared, tmp_path, monkeypatch):
    checkpoint = rollout_checkpoint(pinn_prepared, tmp_path, 'vector_only')
    model, payload = load_predictor(checkpoint)

    def broken(*args, **kwargs):
        assert kwargs == {'decode_fields': False}
        raise FloatingPointError('unstable vector-only recurrence')

    monkeypatch.setattr(model.dynamics_auxiliary_head, 'rollout', broken)
    module = importlib.import_module('climate_manifold.downstream.evaluate')
    monkeypatch.setattr(module, 'load_predictor', lambda *args: (model, payload))
    result = evaluate(checkpoint, pinn_prepared[-1], tmp_path/'failure.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2,
                      dynamics_diagnostics=True, forecast_output=tmp_path/'failure.npz')
    assert result['finite_forecast_fraction'] == 1 and not result['failed_origins']
    assert result['scores']['case_count'] == 2
    aux = result['dynamics_auxiliary_evaluation']
    assert aux['finite_auxiliary_fraction'] == 0 and aux['successful_case_count'] == 0
    assert len(aux['auxiliary_failed_origins']) == 2
    assert aux['scores'] is aux['signed_diagnostics'] is None
    with np.load(tmp_path/'failure.npz') as saved:
        assert 'auxiliary_dynamics_predicted_latent' not in saved
        assert 'auxiliary_dynamics_surface_fields' not in saved
        assert np.isfinite(saved['mean']).all()


def test_all_three_supervision_modes_keep_same_main_horizon_and_extended_origins(
        pinn_prepared, tmp_path):
    reports = []
    for mode in ('field', 'field_latent', 'vector_only'):
        checkpoint = rollout_checkpoint(pinn_prepared, tmp_path, mode)
        reports.append(evaluate(checkpoint, pinn_prepared[-1], tmp_path/(mode+'.json'),
                                information=tmp_path/'pinn-information.npz', max_cases=2,
                                origin_stride=2, dynamics_diagnostics=True,
                                dynamics_rollout_steps=5))
    for key in ('origin_times', 'lead_hours', 'dynamics_origin_selection', 'dynamics_evaluation_contract'):
        assert reports[0][key] == reports[1][key] == reports[2][key]
    for report in reports[:2]:
        audit = report['dynamics_auxiliary_evaluation']
        assert audit['scores']['case_count'] == audit['signed_diagnostics']['case_count'] == 2
        assert audit['evaluated_outputs'] == ['latent', 'physical_fields']
        assert audit['decoder_status'] == 'trained_field_decoder'
