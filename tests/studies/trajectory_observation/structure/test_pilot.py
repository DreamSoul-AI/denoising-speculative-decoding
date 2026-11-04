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
    config['variants'] = [list(variant) for variant in pilot.SUFFIX]
    result = pilot.run_seed(config, 5)
    entry = result['trees'][0]
    assert set(entry['fit_grid']['esp_rejection']) == {'0.0', '0.25', '0.5', '0.75', '1.0'}
    for base in pilot.BASES:
        for tag in ('none', 'rejection', 'top1', 'top1_rejection'):
            assert entry['selected_gain'][f'{base}_{tag}'] in pilot.JOINT_GAINS
    names = ('trajectory', *(f'{arm}{suffix}' for suffix in pilot.SUFFIX.values() for arm in ('observation', 'joint')))
    for split in ('holdout', 'cross'):
        body = entry['splits'][split]
        arms = body['arms']
        assert set(arms) == {f'{b}_{a}' for b in pilot.BASES for a in names}
        assert all(arm['ar_equal_all'] and arm['argmax_all_passed'] for arm in arms.values())
        assert arms['esp_joint_rejection']['gain'] == entry['selected_gain']['esp_rejection']
        assert arms['ema_joint_rejection']['condition'] == 'rejection'
        assert arms['esp_joint_top1_rejection']['form'] == 'top1'
        assert arms['esp_joint_top1']['gain'] == entry['selected_gain']['esp_top1']
        assert all(len(rows) == 2 for rows in body['rows'].values())
        for pairs in (body['paired_vs_esp_trajectory'], body['paired_vs_own_trajectory']):
            assert all(pair['incomparable'] == 0 for pair in pairs.values())


@pytest.mark.parametrize('tree', [{'num_masks': 1, 'branches': [3]}, {'num_masks': 2, 'branches': [3, 2]},
                                  {'num_masks': 3, 'branches': [2, 2, 2]}])
def test_oracle_masks_accept_full_depth_after_prefill(tree):
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder, ar_generate
    from spec_mtp.structure.algorithm.mask import build_mask_provider
    from spec_mtp.model import HFCausalLM

    torch.manual_seed(3)
    model = HFCausalLM(tiny_config={'vocab_size': 50, 'hidden_size': 32, 'intermediate_size': 64,
                                    'num_layers': 2, 'num_heads': 4, 'num_kv_heads': 2,
                                    'max_position_embeddings': 128})
    model.hf.float()
    prompt = torch.randint(0, 50, (1, 9), generator=torch.Generator().manual_seed(4))
    eval_cfg = {'max_new_tokens': 25, 'temperature': 0.0, 'stop_on_eos': False}
    reference = ar_generate(model, prompt, eval_cfg)[0]
    spec = pilot.spec_for('esp', tree, 0.0)
    # A random model often repeats one token; parent-repeat pruning would ban the true token.
    spec['tree']['pruning'] = False
    provider = pilot.OracleMaskProvider(build_mask_provider(spec, model.embedding_table), model.embedding_table,
                                        reference[0, prompt.shape[1]:].tolist())
    output, stats = SpecDecoder(model, spec, eval_cfg, mask_policy=provider).generate(prompt)
    assert torch.equal(output, reference)
    k = tree['num_masks']
    later = [len(row['accepted_nodes']) for row in stats['trace'][1:] if not row['stopped']]
    assert later and all(accepted == k for accepted in later)


def test_output_inside_source_tree_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='outside the source tree'):
        pilot.run(pilot.DEFAULT, pilot.Path(pilot.__file__).parent / 'never-created')
