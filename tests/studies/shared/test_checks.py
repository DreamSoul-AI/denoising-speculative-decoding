"""SpecMTP Run / Study gates on synthetic assets (no decoding)."""

import csv
import json
from types import SimpleNamespace

import pytest

from studies._shared.flow import checks
from studies._shared.structure import sources
from studies._shared.structure.evidence import digest


def _assets(tmp_path, rate='1.0', with_sources=True, overall=True, name='a'):
    assets = tmp_path / name
    assets.mkdir(parents=True)
    rows = [{'category': 'synthetic', 'exact_match_rate': rate, 'accept_d1': '5'}]
    if overall:
        rows.append({'category': 'overall', 'exact_match_rate': rate, 'accept_d1': '5'})
    with (assets / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg = _config()
    saved = sources.manifest()
    saved['config_digest'] = digest(cfg)
    study_dir = assets.parents[2]
    if (study_dir / 'study.yaml').exists():
        from rpipe.structure.artifact.provenance import source_files
        from rpipe.structure.make.expand import load_study_yaml
        saved['study_files'] = source_files(study_dir, load_study_yaml(study_dir))
    if with_sources:
        (assets / 'spec_mtp_sources.json').write_text(json.dumps(saved), encoding='utf-8')
    inputs = {'settings': {'run': {'seed': 0}, 'eval': {'run_ar_baseline': True, 'device': 'cpu'}, 'spec': {}},
              'files': {}, 'device': 'cpu', 'prompts': [{'input_ids': [1, 2], 'category': 'synthetic'}]}
    stats = {'new_tokens': 6, 'committed': 5, 'accepted': 5, 'prefill_calls': 1, 'decode_calls': 1,
             'trace': [{'root_position': 2, 'cache_before': 2, 'accepted_nodes': [1, 2, 3, 4, 5],
                        'stopped': True, 'cache_after': None, 'committed_tokens': [4, 5, 6, 7, 8]}]}
    row = {'category': 'synthetic', **{k: stats[k] for k in ('new_tokens', 'committed', 'accepted', 'decode_calls')},
           'spec_time': 1, 'ar_time': 1, 'exact_match': float(rate) if rate else 1,
           'accept_d1': 5, 'attempt_d1': 5}
    item = {'input_ids': [1, 2], 'output_ids': [1, 2, 3, 4, 5, 6, 7, 8],
            'ar_ids': [1, 2, 3, 4, 5, 6, 7, 8], 'stats': stats}
    import hashlib
    token_hash = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    stats['verification_argmax_audit'] = {'passed': True, 'checked_committed_tokens': 6,
        'input_sha256': token_hash(item['input_ids']), 'output_sha256': token_hash(item['output_ids']),
        'committed': [{'forward': 0, 'row': 1, 'chosen': 3, 'argmax': 3}] +
            [{'forward': 1, 'row': i, 'chosen': token, 'argmax': token} for i, token in enumerate([4,5,6,7,8])]}
    if rate not in ('1.0',''):
        item['ar_ids'][-1] = 99
        row['exact_match'] = 0
    raw = {'task': 'esp_mtp', 'inputs_digest': digest(inputs), 'out': {'rows': [row], 'evidence': [item]}}
    (assets / 'decode_inputs.json').write_text(json.dumps(inputs))
    (assets / 'decode_observations.json').write_text(json.dumps(raw))
    if overall and rate == '1.0':
        from studies._shared.structure.evaluate import _aggregate
        flat = [{'category': k, **v} for k, v in _aggregate([row]).items()]
        with (assets / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    return assets


def _config(task='esp_mtp', run_ar=True):
    return {'algorithm': {'config': {'task': task, 'eval': {'run_ar_baseline': run_ar}}}}


def test_exact_match_one_passes(tmp_path):
    report = checks.evaluate_run(_config(), _assets(tmp_path))
    assert report['passed'] and report['exact_match_rate'] == 1.0 and report['sources_digest'] == sources.manifest()['digest']


@pytest.mark.parametrize('kwargs, reason', [
    ({'overall': False}, 'no single overall row'),
    ({'with_sources': False}, 'spec_mtp_sources'),
])
def test_gate_failures(tmp_path, kwargs, reason):
    report = checks.evaluate_run(_config(), _assets(tmp_path, **kwargs))
    assert not report['passed']
    assert any(reason in item for item in report['reasons'])


def test_unaudited_esp_run_fails(tmp_path):
    assets = _assets(tmp_path)
    path = assets / 'decode_observations.json'
    raw = json.loads(path.read_text())
    del raw['out']['evidence'][0]['stats']['verification_argmax_audit']
    path.write_text(json.dumps(raw))
    report = checks.evaluate_run(_config(), assets)
    assert not report['passed'] and 'argmax' in report['reasons'][0]


def test_run_gate_raises_and_records(tmp_path):
    assets = _assets(tmp_path)
    _corrupt_audit(assets, 'passed', False)
    ctx = SimpleNamespace(config=_config(), layout=SimpleNamespace(assets_dir=assets),
                          state={'result_draft': {'metrics': {}}})
    with pytest.raises(checks.SpecMtpGateError):
        checks.run_gate(ctx)
    assert ctx.state['result_draft']['spec_mtp']['passed'] is False


def _study(tmp_path, runs, axes=None, seeds=(123,)):
    import yaml

    study = tmp_path / 'study'
    (study / 'runs').mkdir(parents=True)
    (study / 'experiment_config.yaml').write_text(yaml.safe_dump(_config()))
    (study / 'study.yaml').write_text(yaml.safe_dump(
        {'study': 'x', 'axes': axes or {'algorithm.config.variant': ['a', 'b']}, 'seeds': list(seeds)}))
    experiments = []
    for variant, status, passed, source_digest in runs:
        run_id = f'run_{variant}'
        experiments.append({'factors': {'algorithm.config.variant': variant},
                            'runs': [{'id': run_id, 'run_dir': run_id, 'seed': 123}]})
        if status is None:
            continue
        (study / 'runs' / run_id).mkdir()
        _assets(study / 'runs' / run_id, name='assets')
        (study / 'runs' / run_id / 'config.yaml').write_text(__import__('yaml').safe_dump(_config()))
        digest_value = sources.manifest()['digest'] if source_digest == 'd' else source_digest
        (study / 'runs' / run_id / 'result.json').write_text(json.dumps(
            {'status': status, 'spec_mtp': {'passed': passed, 'sources_digest': digest_value}}))
    (study / 'index.json').write_text(json.dumps({'study': 'x', 'experiments': experiments}))
    return SimpleNamespace(scope='study', study_dir=study, state={})


def test_study_gate_complete_pass(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')])
    assert checks.study_gate(ctx)['passed']
    assert json.loads((ctx.study_dir / checks.STUDY_CHECK_NAME).read_text())['complete']


def test_study_gate_subset_is_incomplete(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd')])
    with pytest.raises(checks.SpecMtpGateError, match='incomplete'):
        checks.study_gate(ctx)
    assert ctx.state['spec_mtp']['expected_factors_match'] is False


def test_study_gate_missing_result_is_incomplete(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', None, None, None)])
    with pytest.raises(checks.SpecMtpGateError, match='incomplete'):
        checks.study_gate(ctx)
    assert ctx.state['spec_mtp']['missing_runs'] == ['run_b']


def test_study_gate_failed_run_fails(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'failed', False, 'd')])
    with pytest.raises(checks.SpecMtpGateError, match='failed'):
        checks.study_gate(ctx)


def test_study_gate_mixed_source_digest_fails(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd1'), ('b', 'succeeded', True, 'd2')])
    with pytest.raises(checks.SpecMtpGateError):
        checks.study_gate(ctx)


def test_study_gate_seed_mismatch_is_incomplete(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')], seeds=(123, 124))
    with pytest.raises(checks.SpecMtpGateError, match='incomplete'):
        checks.study_gate(ctx)


def test_study_gate_skips_run_scope():
    assert checks.study_gate(SimpleNamespace(scope='run')) == {}


# ------------------------------------------------------------ edge branches

def test_ar_baseline_needs_summary(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    (empty / 'spec_mtp_sources.json').write_text('{"digest": "d"}')
    assert checks.evaluate_run(_config('ar_baseline'), empty)['reasons'] == ['missing summary.csv']
    assert checks.evaluate_run(_config('ar_baseline'), _assets(tmp_path))['passed']


def test_alignment_probe_needs_its_csv(tmp_path):
    assets = _assets(tmp_path)
    assert checks.evaluate_run(_config('alignment_probe'), assets)['reasons'] == ['missing alignment_metrics.csv']
    (assets / 'alignment_metrics.csv').write_text('layer\n1\n')
    assert checks.evaluate_run(_config('alignment_probe'), assets)['passed']


def test_empty_rate_is_missing(tmp_path):
    report = checks.evaluate_run(_config(), _assets(tmp_path, rate=''))
    assert report['passed'] and report['reported_exact_match_rate'] is None and report['exact_match_rate']==1


def test_two_overall_rows_are_rejected(tmp_path):
    assets = _assets(tmp_path)
    with (assets / 'summary.csv').open('a', newline='', encoding='utf-8') as handle:
        handle.write('overall,1.0,5\n')
    assert 'summary.csv has no single overall row' in checks.evaluate_run(_config(), assets)['reasons']


def test_task_name_is_case_insensitive(tmp_path):
    assert checks.evaluate_run(_config('ESP_MTP'), _assets(tmp_path))['task'] == 'esp_mtp'


def test_run_gate_without_draft_still_gates(tmp_path):
    ctx = SimpleNamespace(config=_config(), layout=SimpleNamespace(assets_dir=_assets(tmp_path)), state={})
    assert checks.run_gate(ctx)['passed']
    assets = _assets(tmp_path, name='b')
    _corrupt_audit(assets, 'passed', False)
    bad = SimpleNamespace(config=_config(), layout=SimpleNamespace(assets_dir=assets), state={})
    with pytest.raises(checks.SpecMtpGateError, match='argmax'):
        checks.run_gate(bad)


def test_study_gate_empty_index_is_incomplete(tmp_path):
    ctx = _study(tmp_path, [])
    with pytest.raises(checks.SpecMtpGateError, match='incomplete'):
        checks.study_gate(ctx)
    assert ctx.state['spec_mtp']['complete'] is False


def test_study_gate_explicit_expected_factors(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')])
    expected = [{'algorithm.config.variant': 'b'}, {'algorithm.config.variant': 'a'}]
    assert checks.study_gate(ctx, expected_factors=expected)['passed']


def test_declared_factors_without_axes():
    assert checks.declared_factors({}) == [{}]


@pytest.mark.parametrize('name', ['decode_inputs.json', 'decode_observations.json', 'spec_mtp_sources.json'])
def test_final_gate_requires_raw_evidence(tmp_path, name):
    assets = _assets(tmp_path)
    (assets / name).unlink()
    ctx = SimpleNamespace(config=_config(), layout=SimpleNamespace(assets_dir=assets), state={})
    with pytest.raises(checks.SpecMtpGateError):
        checks.run_gate(ctx)


def test_final_gate_reports_ar_mismatch_without_blocking_argmax(tmp_path):
    assets = _assets(tmp_path)
    path = assets / 'decode_observations.json'
    body = json.loads(path.read_text())
    body['out']['evidence'][0]['ar_ids'][-1] = 99
    path.write_text(json.dumps(body))
    ctx = SimpleNamespace(config=_config(), layout=SimpleNamespace(assets_dir=assets), state={})
    report=checks.run_gate(ctx)
    assert report['passed'] and report['exact_match_rate']==0
    assert report['reported_exact_match_rate']==1 and report['AR_reported_exact_mismatches']==[0]


def test_final_gate_rejects_source_drift(tmp_path):
    assets = _assets(tmp_path)
    path = assets / 'spec_mtp_sources.json'
    body = json.loads(path.read_text())
    body['files']['spec_mtp/added.py'] = 'bad'
    path.write_text(json.dumps(body))
    ctx = SimpleNamespace(config=_config(), layout=SimpleNamespace(assets_dir=assets), state={})
    with pytest.raises(checks.SpecMtpGateError, match='sources changed'):
        checks.run_gate(ctx)


def test_final_gate_rejects_config_drift(tmp_path):
    assets = _assets(tmp_path)
    config = _config()
    config['model'] = {'name': 'changed'}
    ctx = SimpleNamespace(config=config, layout=SimpleNamespace(assets_dir=assets), state={})
    with pytest.raises(checks.SpecMtpGateError, match='configuration changed'):
        checks.run_gate(ctx)


def test_study_gate_rechecks_missing_evidence(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')])
    (ctx.study_dir / 'runs/run_b/assets/decode_observations.json').unlink()
    with pytest.raises(checks.SpecMtpGateError):
        checks.study_gate(ctx)
    assert ctx.state['spec_mtp']['failed_runs'] == ['run_b']


def test_legacy_axis_seed_is_not_an_experiment_factor():
    assert checks.declared_factors({'axes': {'seed': [1, 2], 'x': ['a']}}) == [{'x': 'a'}]


def test_final_gate_rejects_live_input_file_drift(tmp_path):
    from studies._shared.structure.evidence import file_hash
    assets = _assets(tmp_path)
    source = tmp_path / 'prompt.txt'
    source.write_text('before')
    path = assets / 'decode_inputs.json'
    inputs = json.loads(path.read_text())
    inputs['files'] = {str(source): file_hash(source)}
    path.write_text(json.dumps(inputs))
    raw_path = assets / 'decode_observations.json'
    raw = json.loads(raw_path.read_text())
    raw['inputs_digest'] = digest(inputs)
    raw_path.write_text(json.dumps(raw))
    source.write_text('after')
    with pytest.raises(checks.SpecMtpGateError, match='input changed'):
        checks.run_gate(SimpleNamespace(config=_config(), layout=SimpleNamespace(assets_dir=assets), state={}))


def test_cuda_device_alias_does_not_look_like_config_drift(tmp_path):
    assets = _assets(tmp_path)
    path = assets / 'decode_inputs.json'
    inputs = json.loads(path.read_text())
    inputs['device'] = 'cuda:0'
    inputs['settings']['eval']['device'] = 'cuda'
    path.write_text(json.dumps(inputs))
    raw_path = assets / 'decode_observations.json'
    raw = json.loads(raw_path.read_text())
    raw['inputs_digest'] = digest(inputs)
    raw_path.write_text(json.dumps(raw))
    checks.verify_evidence(_config(), assets)


def test_study_quality_failure_marks_generic_envelope_incomplete(tmp_path):
    ctx = _study(tmp_path, [('a', 'succeeded', True, 'd'), ('b', 'succeeded', True, 'd')])
    ctx.state['process'] = {'complete': True}
    (ctx.study_dir / 'runs/run_b/assets/decode_observations.json').unlink()
    with pytest.raises(checks.SpecMtpGateError):
        checks.study_gate(ctx)
    envelope = json.loads((ctx.study_dir / 'process.json').read_text())
    assert not envelope['complete'] and not envelope['spec_mtp']['passed']


def test_declared_factors_follow_mapping_axis_override_contract():
    base = {'algorithm': {'mode': 'eval', 'config': {
        'version': 'base-v1', 'eval': {'max_new_tokens': 100}, 'mask': 'frozen_mean'}}}
    study = {'fixed': {'algorithm': {'config': {'version': 'fixed-v2'}}},
             'axes': {'algorithm.config': [
                 {'task': 'esp_mtp', 'mask': 'gamma_last'}, {'task': 'ar_baseline'}]},
             'seeds': [123, 456]}
    expected = [
        {'algorithm.config': {'version': 'base-v1', 'eval': {'max_new_tokens': 100},
                              'mask': 'gamma_last', 'task': 'esp_mtp'}},
        {'algorithm.config': {'version': 'base-v1', 'eval': {'max_new_tokens': 100},
                              'mask': 'frozen_mean', 'task': 'ar_baseline'}},
    ]
    # A whole algorithm.config axis replaces that fixed subtree, then merges
    # into the base. Keep the official make semantics, rather than inventing
    # recursive base/fixed/axis composition here.
    assert checks.declared_factors(study, base) == expected


def test_declared_factors_bind_inherited_conditions():
    # Same declared patch, different frozen base: effective factors must differ.
    study = {'axes': {'algorithm.config': [{'task': 'ar_baseline'}]}, 'seeds': [123]}
    first = {'algorithm': {'config': {'eval': {'max_new_tokens': 100}}}}
    second = {'algorithm': {'config': {'eval': {'max_new_tokens': 50}}}}
    assert checks.declared_factors(study, first) != checks.declared_factors(study, second)


def _corrupt_audit(assets, key, value):
    path = assets / 'decode_observations.json'
    raw = json.loads(path.read_text())
    raw['out']['evidence'][0]['stats']['verification_argmax_audit'][key] = value
    path.write_text(json.dumps(raw))


@pytest.mark.parametrize('key,value', [('passed',False), ('checked_committed_tokens',5),
    ('input_sha256','bad'), ('output_sha256','bad'), ('committed',[])])
def test_argmax_audit_required_even_when_ar_exact(tmp_path,key,value):
    assets = _assets(tmp_path)
    _corrupt_audit(assets,key,value)
    assert not checks.evaluate_run(_config(),assets)['passed']
    with pytest.raises(checks.SpecMtpGateError,match='argmax'):
        checks.verify_evidence(_config(),assets)


@pytest.mark.parametrize('key,value', [('chosen',99),('argmax',99),('forward',3),('row',3)])
def test_argmax_token_and_owner_tampering_fails(tmp_path,key,value):
    assets = _assets(tmp_path)
    path = assets / 'decode_observations.json'
    raw = json.loads(path.read_text())
    raw['out']['evidence'][0]['stats']['verification_argmax_audit']['committed'][0][key] = value
    path.write_text(json.dumps(raw))
    assert not checks.evaluate_run(_config(),assets)['passed']


def test_ar_mismatch_is_diagnostic_and_native_gate_passes(tmp_path):
    from studies._shared.structure.evaluate import _aggregate
    assets = _assets(tmp_path,rate='0.0')
    raw = json.loads((assets/'decode_observations.json').read_text())
    flat = [{'category':k,**v} for k,v in _aggregate(raw['out']['rows']).items()]
    with (assets/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    ctx=SimpleNamespace(config=_config(),layout=SimpleNamespace(assets_dir=assets),state={'result_draft':{}})
    report=checks.run_gate(ctx)
    assert report['passed'] and report['argmax_checked_tokens']==6 and report['exact_match_rate']==0
    assert report['AR_diagnostics']==[{'prompt_index':0,'exact':False,'first_new_difference_index':5}]
    assert ctx.state['result_draft']['spec_mtp']['validation_policy']=='verifier-argmax-v1'
    assert checks.evaluate_run(_config(run_ar=False),assets)['passed']


def test_jsonl_argmax_audit_must_match_raw(tmp_path):
    assets = _assets(tmp_path)
    (assets/'argmax_audit.jsonl').write_text('{}\n')
    assert not checks.evaluate_run(_config(),assets)['passed']


def test_revalidation_keeps_failure_and_execute_bytes_and_is_idempotent(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    from studies._shared.structure import validation
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('revalidation_tool',root/'tools/revalidate_run.py')
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    ctx=_study(tmp_path,[('a','failed',False,'d')])
    run=ctx.study_dir/'runs/run_a'
    body=json.loads((run/'result.json').read_text())
    body.update(control={},metrics={},error='SpecMtpGateError: old AR gate')
    (run/'result.json').write_text(json.dumps(body))
    previous=(run/'result.json').read_bytes()
    raw=(run/'assets/decode_observations.json').read_bytes()
    import spec_mtp.decode as decoding
    monkeypatch.setattr(decoding.SpecDecoder,'generate',lambda *a:pytest.fail('execute must not rerun'))
    result=tool.revalidate_run(ctx.study_dir,'run_a',Path(validation.__file__))
    assert result['status']=='succeeded' and result['spec_mtp']['argmax_all_passed']
    migration=result['spec_mtp']['revalidation']
    assert migration['execute_rerun'] is False
    assert (run/'assets'/migration['previous_result_asset']).read_bytes()==previous
    assert (run/'assets/decode_observations.json').read_bytes()==raw
    saved=(run/'result.json').read_bytes()
    tool.revalidate_run(ctx.study_dir,'run_a',Path(validation.__file__))
    assert (run/'result.json').read_bytes()==saved


def test_cached_ar_summary_does_not_block_argmax_gate(tmp_path):
    assets=_assets(tmp_path)
    path=assets/'summary.csv'
    with path.open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        row['exact_match_rate']='0.2'
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ctx=SimpleNamespace(config=_config(),layout=SimpleNamespace(assets_dir=assets),state={'result_draft':{'metrics':{}}})
    report=checks.run_gate(ctx)
    assert report['passed'] and report['exact_match_rate']==1 and report['reported_exact_match_rate']==0.2
    assert ctx.state['result_draft']['metrics']['test_exactmatchrate']==1
