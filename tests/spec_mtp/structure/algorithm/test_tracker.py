"""Tracker construction keeps the configured metric and GLUE identity."""
import pytest

from spec_mtp.structure.algorithm.config import AlgorithmConfig
from spec_mtp.structure.algorithm.tracker import make_tracker


@pytest.mark.parametrize('settings,names,subset', [
    (None, {'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']}, 'cola'),
    ({'metric': {'test': ['mse', 'rmse']}}, {'test': ['MSE', 'RMSE']}, 'cola'),
    ({'metric': 'glue', 'glue_subset': 'mrpc'}, {'train': ['GLUE'], 'test': ['GLUE']}, 'mrpc'),
    ({'metric': 'glue', 'glue_subset': ''}, {'train': ['GLUE'], 'test': ['GLUE']}, 'cola'),
])
def test_tracker_configuration_matches_algorithm_settings(tmp_path, settings, names, subset):
    config = None if settings is None else AlgorithmConfig(config=settings)
    tracker = make_tracker(tmp_path, config)
    assert tracker.assets_dir == tmp_path
    assert tracker.metrics.names == names
    assert tracker.metrics.glue_subset == subset
    assert tracker.state_path.is_relative_to(tmp_path)
    assert tracker.jsonl_path.is_file()
