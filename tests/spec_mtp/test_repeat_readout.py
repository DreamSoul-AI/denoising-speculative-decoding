"""Independent-repeat readout retains AR diagnostics and complete argmax coverage."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def reader(monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    path = ROOT / 'studies/esp_table1_nonofficial_revalidation/analyze_results.py'
    spec = importlib.util.spec_from_file_location('sd_repeat_readout', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def example(m):
    ids, output, ar = [1, 2], [1, 2, 3, 4, 5], [1, 2, 3, 7, 5]
    token_hash = lambda x: m.hashlib.sha256(json.dumps(x).encode()).hexdigest()
    audit = {'passed': True, 'checked_committed_tokens': 3,
        'input_sha256': token_hash(ids), 'output_sha256': token_hash(output),
        'committed': [{'chosen': token, 'argmax': token} for token in output[2:]]}
    row = {'prompt': 0, 'category': 'a', 'new_tokens': 3, 'committed': 2, 'decode_calls': 1,
           'accepted': 1, 'spec_time': 0, 'ar_time': 0, 'exact_match': 0}
    stats = dict(row, prefill_calls=1, block_complexity=30, verification_argmax_audit=audit)
    raw = {'rows': [row], 'evidence': [{'input_ids': ids, 'output_ids': output, 'ar_ids': ar, 'stats': stats}]}
    return raw, [audit], [{'question_id': 121, 'category': 'a', 'input_ids': ids}]


def test_ar_mismatch_is_reported_without_invalidating_tau(monkeypatch):
    m = reader(monkeypatch)
    rows, tokens = m.audited_rows(*example(m), 30)
    metrics = m.summarize(rows)
    assert metrics['argmax_all_passed'] and metrics['argmax_checked_tokens'] == 3
    assert metrics['AR_exact_rate'] == 0 and metrics['pooled_tau'] == 2
    assert tokens[0]['first_new_difference_index'] == 1
    assert m.first_difference([3, 4], [3]) == 1
    assert m.first_difference([3], [3]) is None


@pytest.mark.parametrize('damage', ['bad_choice', 'missing_token', 'wrong_budget', 'wrong_input'])
def test_incomplete_or_wrong_verifier_audit_cannot_produce_valid_cell(monkeypatch, damage):
    m = reader(monkeypatch)
    raw, audits, prompts = example(m)
    if damage == 'bad_choice':
        audits[0]['committed'][1]['argmax'] = 7
    elif damage == 'missing_token':
        audits[0]['committed'].pop()
    elif damage == 'wrong_budget':
        raw['evidence'][0]['stats']['block_complexity'] = 60
    else:
        raw['evidence'][0]['input_ids'] = [8, 2]
    with pytest.raises(AssertionError):
        m.audited_rows(raw, audits, prompts, 30)


def test_tau_pairing_uses_ratio_and_rejects_question_reordering(monkeypatch):
    m = reader(monkeypatch)
    def row(qid, committed, calls, accepted):
        return {'question_id': qid, 'category': 'a', 'committed': committed,
            'decode_calls': calls, 'accepted': accepted, 'new_tokens': committed + 1}
    tokens = [{'question_id': q, 'output_sha256': str(q), 'AR_sha256': str(q)} for q in (1, 2)]
    old = {'rows': [row(1, 3, 2, 1), row(2, 4, 2, 2)], 'token_audits': tokens}
    new = {'rows': [row(1, 4, 3, 2), row(2, 2, 1, 1)], 'token_audits': tokens}
    pair = m.compare(old, new)
    assert pair['accepted'] == {'win': 1, 'tie': 0, 'loss': 1}
    assert pair['calls'] == {'win': 1, 'tie': 0, 'loss': 1}
    assert pair['tau'] == {'win': 0, 'tie': 1, 'loss': 1}
    assert pair['output_hash_equal_count'] == pair['AR_hash_equal_count'] == 2
    new['rows'].reverse()
    with pytest.raises(AssertionError):
        m.compare(old, new)


def test_input_contract_uses_real_old_run_eos_and_derived_device(monkeypatch):
    import copy
    m = reader(monkeypatch)
    cfg = {'algorithm': {'config': {'eval': {'temperature': 0., 'max_new_tokens': 256}, 'spec': {'num_masks': 1}}},
           'system': {'config': {'device': 'cuda'}}}
    old = {'settings': {'eval': {'temperature': 0., 'max_new_tokens': 256, 'device': 'cuda'},
                        'spec': {'num_masks': 1}}, 'eos_ids': [151645], 'model_digest': 'weights'}
    m.verify_contract(copy.deepcopy(old), old, cfg)
    for key, value in [('eos_ids', [151643]), ('model_digest', 'changed')]:
        new = copy.deepcopy(old)
        new[key] = value
        with pytest.raises(AssertionError):
            m.verify_contract(new, old, cfg)
    new = copy.deepcopy(old)
    new['settings']['eval']['device'] = 'cpu'
    with pytest.raises(AssertionError):
        m.verify_contract(new, old, cfg)


@pytest.mark.parametrize("budget", [10, 30])
def test_completed_checkpoint_reconciles_all480_pairs(monkeypatch, budget, historical_file):
    m = reader(monkeypatch)
    snapshot = m.read(historical_file(f'studies/esp_table1_nonofficial_revalidation/docs/bc{budget}_checkpoint_v1.json'))
    old = next(c for c in m.read(historical_file('studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json'))['cells']
               if c['BC'] == budget and c['num_masks'] == 1)
    assert snapshot['native_status'] == 'succeeded' and snapshot['validation_policy'] == 'verifier-argmax-v1'
    assert len(snapshot['rows']) == len(snapshot['token_audits']) == 480
    for key, value in m.summarize(snapshot['rows']).items():
        assert snapshot['totals'][key] == pytest.approx(value)
    assert snapshot['vs_original'] == m.compare(old, snapshot)
    assert snapshot['vs_original']['tau'] == {'win': 0, 'tie': 480, 'loss': 0}
    assert snapshot['totals']['committed'] + 480 == snapshot['totals']['new_tokens'] == snapshot['totals']['argmax_checked_tokens']
    assert snapshot['totals']['main_calls'] == snapshot['totals']['decode_calls'] + 480
    assert snapshot['files']['argmax_audit.jsonl'] == old['argmax_audit_sha256']


def test_cross_checkout_pending_readout_writes_only_separate_output(monkeypatch, tmp_path):
    m = reader(monkeypatch)
    monkeypatch.syspath_prepend(str(ROOT / 'studies/esp_table1_nonofficial_revalidation'))
    import revalidation_contract
    from rpipe.structure.artifact import provenance
    from rpipe.structure.make import expand
    from studies._shared.structure import sources
    monkeypatch.setattr(revalidation_contract, 'check_refs', lambda: None)
    monkeypatch.setattr(provenance, 'check_frozen', lambda *args: None)
    monkeypatch.setattr(sources, 'check_frozen', lambda *args: None)
    monkeypatch.setattr(sources, 'manifest', lambda: {})
    monkeypatch.setattr(expand, 'load_study_yaml', lambda *args: {})
    execution = tmp_path / 'execution'
    study, base, output = execution / 'studies/repeat', tmp_path / 'baseline', tmp_path / 'readout'
    (study / 'docs').mkdir(parents=True)
    (base / 'docs').mkdir(parents=True)
    monkeypatch.setattr(m, 'BASE', base)
    for path, value in [(base / 'docs/protocol_evidence_v1.json', {'cells': [
            {'BC': bc, 'role': 'primary'} for bc in (10, 30, 60)]}),
            (base / 'docs/prompts_manifest.json', {'rendered_prompts': [{}] * 480}),
            (base / 'docs/modelscope_snapshot.json', {}), (study / 'docs/source_refs.json', {})]:
        path.write_text(json.dumps(value))
    (base / 'analyze_results.py').write_text('fixture')
    entries = [{'factors': {'algorithm.config': {'spec': {'block_complexity': bc, 'num_masks': 1,
        'method': 'esp', 'tree': {'mode': 'dynamic', 'pruning': True},
        'mask': {'init': 'mean_prompt', 'lam': .1, 'update': True}}}}, 'runs': [{'id': str(bc)}]}
        for bc in (10, 30, 60)]
    (study / 'index.json').write_text(json.dumps({'experiments': entries}))
    before = {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}
    monkeypatch.chdir(execution)
    m.main(study_dir=study, output_dir=output)
    body = m.read(output / 'repeat_pending.json')
    assert body['complete'] is False and body['execution_root'] == str(execution)
    assert [c['BC'] for c in body['cells']] == [10, 30, 60]
    assert all(c['complete'] is False for c in body['cells'])
    assert not (output / 'repeat_evidence_v1.json').exists()
    assert before == {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AssertionError, match='frozen execution checkout root'):
        m.main(study_dir=study, output_dir=output)


def test_cpu_acceptance_checkpoint_matches_completed_cells_and_published_pairs(monkeypatch, historical_file):
    m = reader(monkeypatch)
    proof = m.read(historical_file('studies/esp_table1_nonofficial_revalidation/docs/cpu_acceptance_checkpoint_v1.json'))
    assert not proof['complete_8b_matrix'] and proof['no_model_forward'] and proof['no_timing']
    assert [c['BC'] for c in proof['table1_cells'] if c['complete']] == [10, 30]
    for cell in proof['table1_cells'][:2]:
        path = historical_file(cell['checkpoint'])
        assert m.sha(path) == cell['checkpoint_sha256']
        original = m.read(path)
        assert cell['pooled_tau'] == original['totals']['pooled_tau']
        assert cell['paired'] == {k: v for k, v in original['vs_original'].items() if k != 'rows'}
    paired = m.read(historical_file('studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json'))
    recheck = proof['four_b_published_reaggregation']
    assert (recheck['cells_checked'], recheck['pairs_checked']) == (11, 20)
    assert recheck['cells'] == {k: v['totals'] for k, v in paired['cells'].items()}
    assert recheck['pairs'] == [{k: v for k, v in p.items() if k != 'rows'} for p in paired['pairs']]
