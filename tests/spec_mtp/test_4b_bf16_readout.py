import importlib.util
import json
from pathlib import Path
import pytest

STUDY = Path(__file__).resolve().parents[2]/'studies/qwen3_4b_bf16_five_methods'


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.syspath_prepend(str(STUDY))
    spec = importlib.util.spec_from_file_location('bf16_pair_reader', STUDY/'analyze_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pairs_retain_losses_ties_and_inverse_call_direction(reader):
    rows = [dict(question_id=i, category='a', accepted=a, model_calls=c, new_tokens=20)
            for i, (a, c) in enumerate(((3, 8), (4, 6), (1, 4)))]
    other = [dict(r, accepted=a, model_calls=c, new_tokens=19) for r, (a, c) in
             zip(rows, ((5, 6), (2, 8), (1, 4)))]
    result = reader.compare_rows(rows, other)
    assert result['accepted'] == result['calls'] == {'win': 1, 'tie': 1, 'loss': 1}
    assert result['total_delta_second_minus_first'] == {'accepted': 0, 'model_calls': 0, 'new_tokens': -3}
    assert [r['calls'] for r in result['rows']] == ['win', 'loss', 'tie']
    for bad in ([], other[:-1], list(reversed(other)), [dict(r, question_id=0) for r in other]):
        with pytest.raises(AssertionError):
            reader.compare_rows(rows, bad)


@pytest.mark.parametrize('output,reference,first', [([1,2], [1,2], None), ([1,3], [1,2], 1),
    ([1], [1,2], 1), ([1,2], [1], 1), ([], [1], 0)])
def test_ar_divergence_is_diagnostic_including_length_only(reader, output, reference, first):
    assert reader.first_difference(output, reference) == first


def test_pending_has_all_11_cells_without_invented_numbers_or_evidence_link(reader, tmp_path, capsys):
    (tmp_path/'docs').mkdir()
    reader.__file__ = str(tmp_path/'analyze_results.py')
    Path(reader.__file__).write_text('fixture')
    reader.main()
    body = json.loads((tmp_path/'docs/paired_pending.json').read_text())
    assert body['complete'] is False and tuple(body['cells']) == reader.CASES
    assert all(c == {'native_status': 'pending'} for c in body['cells'].values())
    assert 'paired_evidence_v1.json' not in (tmp_path/'docs/STUDY_REPORT.md').read_text()
    assert '"observed": 0' in capsys.readouterr().out


def test_readout_retains_ar_mismatch_and_audit_counts_after_native_validation(reader, tmp_path, monkeypatch):
    from studies._shared.structure import validation
    import untimed_pairing
    checked = []
    monkeypatch.setattr(validation, 'verify_evidence', lambda *args: checked.append(True))
    monkeypatch.setattr(untimed_pairing, 'verify_inputs', lambda *args: None)
    folder = tmp_path/'run'; assets = folder/'assets'; assets.mkdir(parents=True)
    settings = {'task': 'esp_mtp'}
    prompts = [dict(question_id=i, category='a', input_ids=[10]) for i in range(480)]
    stats = dict(accepted=1, committed=2, decode_calls=1, prefill_calls=1, prefill_time=0., decode_time=0.,
                 verification_argmax_audit=dict(passed=True, checked_committed_tokens=3))
    payloads = {'result.json': {'status': 'succeeded', 'spec_mtp': dict(passed=True,
        validation_policy='verifier-argmax-v1', validator_sha256='validator', sources_digest='source')},
        'config.yaml': {}, 'assets/spec_mtp_sources.json': {},
        'assets/decode_inputs.json': dict(model_digest='model', runtime={}),
        'assets/pairing_contract.json': dict(dtype='torch.bfloat16', device='cuda:0', tied_storage=True,
            manifest_checked=True, embedding_shape=[20,4], bf16_reduced_precision_reduction=True, deterministic=False),
        'assets/decode_observations.json': {'out': {'rows': [dict(spec_time=0.)]*480,
            'evidence': [dict(input_ids=[10], output_ids=[10,1,2,3], ar_ids=None, stats=stats)]*480}}}
    for name, value in payloads.items():
        (folder/name).write_text(json.dumps(value))
    reference = [dict(question_id=i, output_ids=[1,9,3][:3-i%3], model_calls=3-i%3) for i in range(480)]
    cell, outputs = reader.load_cell(tmp_path, folder, {'algorithm': {'config': settings}}, prompts, {},
                                    {'config': dict(vocab_size=20, hidden_size=4)}, reference)
    assert checked == [True] and not outputs
    assert cell['totals']['argmax_checked_tokens'] == 1440 and cell['totals']['AR_exact_rate'] == 0
    assert cell['totals']['pooled_tau'] == 2 and cell['totals']['model_calls'] == 960
    assert all(r['first_new_difference_index'] == 1 for r in cell['rows'])
    assert {r['AR_new_tokens'] for r in cell['rows']} == {1,2,3}
    assert cell['totals']['calls_vs_AR'] == dict(win=160, tie=160, loss=160)
    assert cell['totals']['calls_delta_vs_AR'] == 0
    assert [r['calls_vs_AR'] for r in cell['rows'][:3]] == ['win', 'tie', 'loss']
    assert 'output_ids' not in json.dumps(cell) and 'input_ids' not in json.dumps(cell)
    reference[0]['question_id'] = -1
    with pytest.raises(AssertionError):
        reader.load_cell(tmp_path, folder, {'algorithm': {'config': settings}}, prompts, {},
                         {'config': dict(vocab_size=20, hidden_size=4)}, reference)


@pytest.fixture
def published_reader(reader, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'analyze_results', reader)
    spec = importlib.util.spec_from_file_location('bf16_published_counts', STUDY / 'verify_published.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_counts_reaggregate_without_rewriting_original(published_reader):
    path = STUDY / 'docs/paired_evidence_v1.json'
    before = path.read_bytes()
    result = published_reader.verify(path)
    assert (result['cells_checked'], result['pairs_checked']) == (11, 20)
    assert result['source_sha256'] == '8e85ac6ea9984047abf0ffd5d6653b28b6aadd74aa876139fa10465dc5bbf274'
    assert result['cells']['30/input_first']['pooled_tau'] == pytest.approx(1.51195758)
    assert path.read_bytes() == before


@pytest.mark.parametrize('damage', ['totals', 'pair', 'missing_pair', 'prompt_order'])
def test_published_counts_reject_inconsistent_or_incomplete_pairs(published_reader, tmp_path, damage):
    from itertools import combinations
    from hashlib import sha256
    # Generated count fixture exercises the reader, never impersonates historical model evidence.
    rows = [dict(question_id=i, category='synthetic', accepted=1, committed=2, decode_calls=1,
                 new_tokens=3, model_calls=2, argmax_checked_tokens=3,
                 AR_exact=True, calls_delta_vs_AR=0, calls_vs_AR='tie') for i in range(480)]
    body = dict(complete=True, argmax_is_gate_AR_is_diagnostic=True, timing_disabled=True, cells={}, pairs=[])
    for name in published_reader.CASES:
        body['cells'][name] = dict(native_status='succeeded', rows=[dict(row) for row in rows],
                                  totals=published_reader.summarize(rows, name == 'AR'))
    for bc in (30, 60):
        for left, right in combinations(published_reader.METHODS, 2):
            a, b = f'{bc}/{left}', f'{bc}/{right}'
            body['pairs'].append(dict(first=a, second=b, **published_reader.compare_rows(rows, rows)))
    valid = tmp_path / 'synthetic.json'
    valid.write_text(json.dumps(body))
    assert published_reader.verify(valid)['source_sha256'] == sha256(valid.read_bytes()).hexdigest()
    if damage == 'totals':
        body['cells']['30/input_first']['totals']['pooled_tau'] += .1
    elif damage == 'pair':
        body['pairs'][0]['calls']['win'] += 1
    elif damage == 'missing_pair':
        body['pairs'].pop()
    else:
        body['cells']['AR']['rows'].reverse()
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(body))
    with pytest.raises(AssertionError):
        published_reader.verify(path)
