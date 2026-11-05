"""Native evidence readout: paired counts, AR diagnostics, no generation or timing."""
if not __debug__:
    raise RuntimeError("paired evidence requires Python assertions enabled")

import hashlib
import json
from itertools import combinations
from pathlib import Path

METHODS = ('frozen_mean', 'input_zero', 'input_first', 'output_zero', 'output_first')
CASES = ('AR',) + tuple(f'{bc}/{method}' for bc in (30, 60) for method in METHODS)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def case_name(settings):
    if settings['task'] == 'ar_baseline':
        return 'AR'
    assert settings['task'] == 'esp_mtp'
    spec = settings['spec']
    if spec['method'] == 'esp':
        method = 'input_zero' if spec['mask']['update'] else 'frozen_mean'
    elif spec['method'] == 'ema_velocity':
        method = 'input_first'
    else:
        assert spec['method'] == 'output_hidden' and spec['output']['order'] in (0, 1)
        method = ('output_zero', 'output_first')[spec['output']['order']]
    name = f"{spec['block_complexity']}/{method}"
    assert name in CASES
    return name


def first_difference(output, reference):
    if output == reference:
        return None
    return next((i for i, (a, b) in enumerate(zip(output, reference)) if a != b),
                min(len(output), len(reference)))


def compare_rows(first, second):
    assert first and len(first) == len(second)
    rows = []
    assert len({r['question_id'] for r in first}) == len(first)
    for left, right in zip(first, second):
        assert (left['question_id'], left['category']) == (right['question_id'], right['category'])
        assert all(r[k] >= 0 for r in (left, right) for k in ('accepted', 'model_calls'))
        delta = {k: right[k] - left[k] for k in ('accepted', 'model_calls', 'new_tokens')}
        rows.append({'question_id': left['question_id'], 'category': left['category'],
                     'delta_second_minus_first': delta,
                     'accepted': 'win' if delta['accepted'] > 0 else 'loss' if delta['accepted'] < 0 else 'tie',
                     'calls': 'win' if delta['model_calls'] < 0 else 'loss' if delta['model_calls'] > 0 else 'tie'})
    return {'n': len(rows), 'accepted': {s: sum(r['accepted'] == s for r in rows) for s in ('win', 'tie', 'loss')},
            'calls': {s: sum(r['calls'] == s for r in rows) for s in ('win', 'tie', 'loss')},
            'total_delta_second_minus_first': {k: sum(r['delta_second_minus_first'][k] for r in rows)
                                               for k in ('accepted', 'model_calls', 'new_tokens')}, 'rows': rows}


def summarize(rows, is_ar):
    totals = {k: sum(r[k] for r in rows) for k in ('new_tokens', 'model_calls')}
    if not is_ar:
        totals.update({k: sum(r[k] for r in rows) for k in ('accepted', 'committed', 'decode_calls', 'argmax_checked_tokens')})
        totals.update(pooled_tau=totals['committed']/totals['decode_calls'],
                      prompt_mean_tau=sum(r['committed']/max(r['decode_calls'], 1) for r in rows)/len(rows),
                      AR_exact_count=sum(r['AR_exact'] for r in rows),
                      AR_exact_rate=sum(r['AR_exact'] for r in rows)/len(rows),
                      calls_delta_vs_AR=sum(r['calls_delta_vs_AR'] for r in rows),
                      calls_vs_AR={s: sum(r['calls_vs_AR'] == s for r in rows) for s in ('win', 'tie', 'loss')})
    return totals


def load_cell(study, folder, cfg, prompts, manifest, snapshot, reference, *, reader=None):
    from studies._shared.structure.evidence import file_hash
    from studies._shared.structure.validation import verify_evidence
    from studies.qwen3_4b_bf16_five_methods.untimed_pairing import verify_inputs
    read_json = read if reader is None else reader.read_json
    if reader is not None:
        file_hash = reader.sha
        for name in ('config.yaml', 'result.json', 'assets/decode_observations.json', 'assets/decode_inputs.json',
                     'assets/pairing_contract.json', 'assets/spec_mtp_sources.json', 'assets/summary.csv'):
            reader.sha(folder/name)
    assets = folder/'assets'
    result = read_json(folder/'result.json')
    gate = result['spec_mtp']
    assert result['status'] == 'succeeded' and gate['passed']
    assert gate['validation_policy'] == 'verifier-argmax-v1'
    verify_evidence(cfg, assets)
    inputs = read_json(assets/'decode_inputs.json')
    verify_inputs(inputs, manifest, snapshot)
    contract = read_json(assets/'pairing_contract.json')
    assert contract['dtype'] == 'torch.bfloat16' and contract['device'].startswith('cuda')
    assert contract['tied_storage'] and contract['manifest_checked']
    assert contract['embedding_shape'] == [snapshot['config']['vocab_size'], snapshot['config']['hidden_size']]
    assert contract['bf16_reduced_precision_reduction'] and not contract['deterministic']
    settings = cfg['algorithm']['config']
    is_ar = settings['task'] == 'ar_baseline'
    raw = read_json(assets/'decode_observations.json')['out']
    assert len(raw['rows']) == len(raw['evidence']) == len(prompts) == 480
    assert is_ar or reference is not None
    rows, outputs = [], []
    for i, (p, row, item) in enumerate(zip(prompts, raw['rows'], raw['evidence'])):
        assert item['input_ids'] == p['input_ids']
        output = item['output_ids'][len(p['input_ids']):]
        stats = item['stats']
        common = {'question_id': p['question_id'], 'category': p['category'], 'generator_seed': 123+i,
                  'new_tokens': len(output), 'output_sha256': hashlib.sha256(json.dumps(output).encode()).hexdigest(),
                  'hit_context_limit': stats.get('hit_context_limit', False)}
        if is_ar:
            assert stats['ar_time'] == row['ar_time'] == 0
            common['model_calls'] = stats['ar_calls']
            outputs.append({'question_id': p['question_id'], 'output_ids': output, 'model_calls': stats['ar_calls']})
        else:
            assert stats['prefill_time'] == stats['decode_time'] == row['spec_time'] == 0
            assert item.get('ar_ids') is None and 'ar_time' not in row
            assert stats['verification_argmax_audit']['passed'] is True
            assert stats['prefill_calls'] == 1
            ar = reference[i]
            assert ar['question_id'] == p['question_id']
            delta = stats['decode_calls']+stats['prefill_calls']-ar['model_calls']
            common.update({k: stats[k] for k in ('accepted', 'committed', 'decode_calls')})
            common.update(model_calls=stats['decode_calls']+stats['prefill_calls'],
                          argmax_checked_tokens=stats['verification_argmax_audit']['checked_committed_tokens'],
                          AR_exact=output == ar['output_ids'], AR_new_tokens=len(ar['output_ids']),
                          AR_model_calls=ar['model_calls'], calls_delta_vs_AR=delta,
                          calls_vs_AR='win' if delta < 0 else 'loss' if delta > 0 else 'tie',
                          first_new_difference_index=first_difference(output, ar['output_ids']))
        rows.append(common)
    files = {str(p.relative_to(folder)).replace('\\', '/'): file_hash(p) for p in
             (folder/'config.yaml', folder/'result.json', assets/'decode_observations.json',
              assets/'decode_inputs.json', assets/'pairing_contract.json', assets/'spec_mtp_sources.json')}
    return {'run_id': folder.name, 'native_status': result['status'], 'validator_sha256': gate['validator_sha256'],
            'source_digest': gate['sources_digest'], 'model_digest': inputs['model_digest'],
            'runtime': inputs['runtime'], 'actual_settings': settings, 'contract': contract,
            'files_sha256': files, 'totals': summarize(rows, is_ar), 'rows': rows}, outputs


def verify(path):
    if not __debug__:
        raise RuntimeError("paired evidence requires Python assertions enabled")
    path = Path(path)
    body = read(path)
    assert body['complete'] and body['argmax_is_gate_AR_is_diagnostic'] and body['timing_disabled']
    assert set(body['cells']) == set(CASES)
    reference = [(r['question_id'], r['category']) for r in body['cells']['AR']['rows']]
    assert len(reference) == len(set(reference)) == 480
    for name, cell in body['cells'].items():
        assert cell['native_status'] == 'succeeded'
        assert [(r['question_id'], r['category']) for r in cell['rows']] == reference
        assert summarize(cell['rows'], name == 'AR') == cell['totals']
        if name != 'AR':
            assert all(r['argmax_checked_tokens'] == r['new_tokens'] == r['committed'] + 1
                       and r['model_calls'] == r['decode_calls'] + 1 for r in cell['rows'])
    expected = [(f'{bc}/{a}', f'{bc}/{b}') for bc in (30, 60) for a, b in combinations(METHODS, 2)]
    assert [(p['first'], p['second']) for p in body['pairs']] == expected
    for pair in body['pairs']:
        actual = compare_rows(body['cells'][pair['first']]['rows'], body['cells'][pair['second']]['rows'])
        assert actual == {k: v for k, v in pair.items() if k not in ('first', 'second')}
    return {'scope': 'Published counter reaggregation only; no native replay or model/token audit',
            'native_run_revalidated': False, 'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cells_checked': len(body['cells']), 'pairs_checked': len(body['pairs']),
            'cells': {name: cell['totals'] for name, cell in body['cells'].items()},
            'pairs': [{k: v for k, v in pair.items() if k != 'rows'} for pair in body['pairs']]}
