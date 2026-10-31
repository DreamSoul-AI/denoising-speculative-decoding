"""Synthetic pilot wiring: language, isolation of gain selection and correctness gates."""

import json

import pytest
import torch

from studies.trajectory_observation.structure import pilot


def test_language_rows_are_distributions_over_branching_successors():
    language = pilot.make_language(6, 2, 2.0, torch.Generator().manual_seed(0))
    assert torch.allclose(language.sum(-1), torch.ones(6, 6))
    assert ((language > 0).sum(-1) == 2).all()
    rows = pilot.sample(language, 6, 4, 10, torch.Generator().manual_seed(1))
    assert (rows[:, 0] == 6).all() and rows[:, 1:].max() < 6
    for row in rows:
        for position in range(3, 10):
            assert language[row[position - 2], row[position - 1], row[position]] > 0


def test_tiny_seed_keeps_ar_output_and_selects_gain_on_fit_prompts_only():
    config = json.loads(json.dumps(pilot.DEFAULT))
    config.update(content_vocab=8, max_new_tokens=6, trees=[{'num_masks': 2, 'branches': [3, 2]}])
    config['model'].update(hidden_size=16, intermediate_size=32, max_position_embeddings=64)
    config['train'].update(sequences=32, length=16, steps=3, batch=4)
    config['prompts'] = {'fit': 2, 'holdout': 2, 'cross': 2, 'length': 8}
    result = pilot.run_seed(config, 5)
    entry = result['trees'][0]
    assert set(entry['fit_grid']['esp']) == {'0.0', '0.25', '0.5', '0.75', '1.0'}
    for base in pilot.BASES:
        assert entry['selected_gain'][base] in pilot.JOINT_GAINS
    for split in ('holdout', 'cross'):
        arms = entry['splits'][split]['arms']
        assert set(arms) == {f'{b}_{a}' for b in pilot.BASES for a in ('trajectory', 'observation', 'joint')}
        assert all(arm['ar_equal_all'] and arm['argmax_all_passed'] for arm in arms.values())
        assert arms['esp_joint']['gain'] == entry['selected_gain']['esp']
        assert all(pair['incomparable'] == 0 for pair in entry['splits'][split]['paired_vs_esp_trajectory'].values())


def test_output_inside_source_tree_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='outside the source tree'):
        pilot.run(pilot.DEFAULT, pilot.Path(pilot.__file__).parent / 'never-created')
