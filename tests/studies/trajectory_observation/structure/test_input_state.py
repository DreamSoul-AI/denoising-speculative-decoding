"""E1 split isolation, dev-only selection and pooled retained metrics."""

import pytest

from studies.trajectory_observation.structure.input_state import choose_inputs, candidate_specs, pooled, select_arm, encode_inputs


def test_e1_reserves_unseen_balanced_eval_before_dev():
    data = [{'instruction': f'{c}-{i}', 'context': '', 'category': str(c)}
            for c in range(8) for i in range(7)]
    chosen = choose_inputs(data, {'0-0'}, 9)
    assert len(chosen['dev']) == 8 and len(chosen['eval']) == 32
    dev = {r['text'] for r in chosen['dev']}
    evaluation = {r['text'] for r in chosen['eval']}
    assert not dev.intersection(evaluation) and '0-0' not in dev | evaluation
    assert all(sum(r['category'] == str(c) for r in chosen['eval']) == 4 for c in range(8))
    assert chosen == choose_inputs(data, {'0-0'}, 9)


def test_e1_selection_requires_dev_pairs_and_preserves_declared_ties():
    rows = [{'split': 'dev', 'qid': str(i), 'arm': arm, 'A': 2, 'C': 2, 'S': 4, 'N': 5}
            for arm in ('first', 'second') for i in range(2)]
    assert select_arm(rows, ['first', 'second']) == 'first'
    with pytest.raises(ValueError, match='dev rows only'):
        select_arm(rows + [{**rows[0], 'split': 'eval'}], ['first'])
    with pytest.raises(ValueError, match='unpaired'):
        select_arm(rows[:-1], ['first', 'second'])
    assert pooled(rows) == {'A': 8, 'C': 8, 'S': 16, 'N': 20, 'prompts': 4,
                           'accepted_per_call': 1., 'tau': 2.}


def test_e1_grid_keeps_tree_and_corrections_fixed():
    arms = candidate_specs()
    assert len(arms) == 23
    assert all(s['num_masks'] == 1 and s['tree']['branches'] == [14] for s in arms.values())
    assert all(not any(k in s for k in ('correction', 'observation', 'hybrid_gain')) for s in arms.values())
    assert sum(a.startswith('holt_') for a in arms) == 9
    assert sum(a.startswith('kalman_') for a in arms) == 9


def test_e1_template_is_rendered_before_explicit_token_encoding():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
            return messages[0]['content']

        def encode(self, text, **kwargs):
            assert text == 'raw' and kwargs == {'add_special_tokens': False}
            return list(range(600))
    row = {'qid': 'x', 'category': 'c', 'text': 'raw'}
    result = encode_inputs(Tokenizer(), {'dev': [row]})
    assert result['dev'][0]['input_ids'] == list(range(88, 600))
    assert 'input_ids' not in row


def test_e1_bootstrap_resamples_pairs_and_rejects_missing_prompts():
    from studies.trajectory_observation.structure.input_state_analysis import paired_comparison
    rows = [{'split': 'eval', 'qid': str(i), 'arm': arm, 'A': 2, 'C': 2, 'S': 4, 'N': 5,
             'tokens': [1, 2, 3, 4, 5]} for arm in ('a', 'b') for i in range(3)]
    result = paired_comparison(rows, 'a', 'b', resamples=100)
    assert result['tau_delta'] == 0 and result['exploratory_paired_ci95'] == [0, 0]
    assert result['equal_output_prompts'] == result['equal_length_prompts'] == 3
    with pytest.raises(ValueError, match='pairs differ'):
        paired_comparison(rows[:-1], 'a', 'b', resamples=100)


def test_e1_analysis_reaudits_raw_and_rejects_changed_row(tmp_path):
    import json
    import torch
    from dataclasses import asdict
    from spec_mtp.model import HFCausalLM
    from spec_mtp.structure.algorithm.decode.loop import generate
    from spec_mtp.structure.algorithm.mask import build_mask_provider
    from spec_mtp.structure.api.decode_api import DecodeRequest, Prompt
    from studies.trajectory_observation.structure.input_state import file_hash, write_json, ProvenanceSink
    from studies.trajectory_observation.structure.input_state_analysis import analyze_run

    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size': 31, 'hidden_size': 16, 'intermediate_size': 32,
                                   'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1})
    write_json(tmp_path / 'fixture-model.json', model.config.to_dict())
    manifest = {'fixture-model.json': file_hash(tmp_path / 'fixture-model.json')}
    write_json(tmp_path / 'model-manifest.json', manifest)
    write_json(tmp_path / 'source.json', {'fixture': 'synthetic randomly initialized model'})
    write_json(tmp_path / 'inputs.json', {'dev': [], 'eval': [{'qid': 'p', 'input_ids': [1, 2]}]})
    config = {'model_manifest': str(tmp_path / 'model-manifest.json'),
              'source_manifest': str(tmp_path / 'source.json'), 'inputs': str(tmp_path / 'inputs.json')}
    write_json(tmp_path / 'config.json', config)
    sources = {'code': file_hash(config['source_manifest']), 'inputs': file_hash(config['inputs']),
               'model': file_hash(config['model_manifest']), 'config': file_hash(tmp_path / 'config.json')}
    selected = ['order0', 'holt_0.1_0.9', 'kalman_0.01_0.01', 'fixed_gain']
    write_json(tmp_path / 'summary.json', {'complete': True, 'selected': selected})
    rows = []
    for arm, kind in zip(selected, ('order0', 'holt', 'kalman', 'fixed_gain')):
        spec = {'method': 'input_trajectory', 'num_masks': 1, 'impl': 'efficient',
                'tree': {'mode': 'static', 'branches': [2]}, 'trajectory': {'kind': kind}}
        request = DecodeRequest(Prompt('p', (1, 2), 'fixture'), spec, max_new_tokens=3,
                                stop_on_eos=False, sources=sources)
        path = tmp_path / (arm + '.jsonl')
        sink = ProvenanceSink(path, manifest, model.describe())
        try:
            result = generate(request, model, build_mask_provider(spec, model.embedding_table), sink)
        finally:
            sink.close()
        rows.append({'split': 'eval', 'qid': 'p', 'category': 'fixture', 'arm': arm,
                     **{k: result.stats['acceptance_metrics'][k] for k in ('A', 'C', 'S')},
                     'N': len(result.committed), 'tokens': list(result.committed),
                     'audit': asdict(result), 'raw': path.name})
    path = tmp_path / 'rows.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    report = analyze_run(tmp_path)
    assert report['audited_rows'] == 4 and report['audited_tokens'] == 12
    assert (tmp_path / 'results.csv').is_file() and (tmp_path / 'results.md').is_file()
    rows[0]['A'] += 1
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='metrics differ'):
        analyze_run(tmp_path)
