"""Observed reconstruction constraints use co-located, strictly causal pairs."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.downstream.train import CausalWindows, windows


class InformationReads:
    def __init__(self, data):
        self.data, self.reads = data, []

    def __getitem__(self, index):
        self.reads.append(index)
        return self.data[index]


def test_constraint_pair_uses_adjacent_observations_with_sparse_history(pinn_prepared):
    manifold, _, data, _ = pinn_prepared
    config = replace(manifold.config, history_stride=4)
    information = InformationReads(data['information'])
    dataset = CausalWindows(data['states'], data['times'], config, [7], data['mean'],
                            data['scale'], data['schema'], information=information,
                            reconstruction_constraints=True)
    row = dataset[0]
    origin = 7 + config.history_span_steps - 1
    expected = (data['states'][origin-1:origin+1]-data['mean'])/data['scale']
    torch.testing.assert_close(row['constraint_states'], torch.tensor(expected, dtype=torch.float32))
    torch.testing.assert_close(row['constraint_information'],
                               torch.tensor(data['information'][origin-1:origin+1]))
    torch.testing.assert_close(row['history'][-1], row['constraint_states'][-1])
    assert row['constraint_dt_hours'].tolist() == [6.]
    assert information.reads == [origin, slice(origin-1, origin+1)]
    assert 'information_targets' not in row
    # The preceding dense state is not a fabricated 24-hour predictor sample.
    assert not torch.equal(row['history'][-2], row['constraint_states'][0])


def test_pair_stays_inside_each_split_observed_window_and_ignores_future_information(pinn_prepared):
    manifold, _, data, _ = pinn_prepared
    for split in ('train', 'calibration', 'validation', 'test'):
        dataset = windows(data, manifold.config, split, max_windows=1,
                          reconstruction_constraints=True)
        start = dataset.starts[0]
        origin = start+manifold.config.history_span_steps-1
        assert start <= origin-1 < origin
        row = dataset[0]
        old = dataset.information
        poisoned = old.copy()
        poisoned[origin+1:] = np.nan
        dataset.information = poisoned
        again = dataset[0]
        for key in ('history', 'information', 'constraint_states', 'constraint_information', 'constraint_dt_hours'):
            torch.testing.assert_close(row[key], again[key], rtol=0, atol=0)
        assert 'information_targets' not in again
    # Target splits remain disjoint; observed contexts may overlap by design.
    c = manifold.config
    last_train_target = data['split']['train'][-1]+c.history_span_steps-1+c.horizon_steps
    first_selection_target = data['split']['calibration'][0]+c.history_span_steps
    assert first_selection_target > last_train_target


def test_constraint_dataset_rejects_missing_pair_or_future_target_route(pinn_prepared):
    manifold, _, data, _ = pinn_prepared
    for config, info, future in (
            (replace(manifold.config, history_steps=1), data['information'], False),
            (manifold.config, None, False),
            (manifold.config, data['information'], True)):
        with pytest.raises(ValueError, match='reconstruction constraints'):
            CausalWindows(data['states'], data['times'], config, [0], data['mean'], data['scale'],
                          data['schema'], information=info, information_targets=future,
                          reconstruction_constraints=True)
