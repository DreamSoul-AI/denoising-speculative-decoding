"""cpu pure IO/stats: native saved-byte admission, no tensors or model execution."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import yaml

from studies._shared.structure import paired_4b
from studies._shared.structure.evidence import digest

reader = importlib.import_module('studies.output_space_4b_recheck_v2.flow.process.run')
ROOT = Path(__file__).resolve().parents[5]


def producer_config(settings):
    return {'seed': 123, 'algorithm': {'config': settings},
            'data': {'name': 'specbench', 'config': {'path': 'datasets/spec_bench/question.jsonl', 'take_per_category': 2}},
            'model': {'name': 'hf_causal', 'config': {'name_or_path': 'Qwen/Qwen3-4B', 'hub': 'modelscope', 'torch_dtype': 'float32'}},
            'system': {'config': {'device': 'cuda'}}}


def producer_settings(cfg):
    from types import SimpleNamespace
    from studies._shared.structure.config import legacy_cfg
    from studies._shared.structure.algorithm import effective_spec
    algorithm = SimpleNamespace(setting=lambda key, default=None: cfg['algorithm']['config'].get(key, default))
    data = {'name': 'specbench', 'path': 'D:/synthetic/landed/question.jsonl', 'take_per_category': 2}
    model = {'type': 'hf_causal', 'name_or_path': 'D:/synthetic/model', 'hub': 'modelscope',
             'torch_dtype': 'float32', 'cache_dir': 'D:/synthetic/cache', 'local_files_only': True}
    saved = legacy_cfg(SimpleNamespace(meta={'data_cfg': data}), SimpleNamespace(meta={'model_cfg': model}),
                       SimpleNamespace(device='cuda'), algorithm, cfg['seed'])
    saved.spec = effective_spec(saved.spec, algorithm)
    return saved


def fixture_study(tmp_path, monkeypatch, name, count=2, settings_damage=None):
    """Synthetic saved bytes exercise the protocol, never impersonate old measurement artifacts."""
    from spec_mtp.structure.artifact import provenance
    execution_root = tmp_path / 'saved'
    study = execution_root / 'studies' / name
    docs = study / 'docs'
    docs.mkdir(parents=True)

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    prompts = [dict(question_id=17+i, category='fixture', input_ids=[9+i]) for i in range(count)]
    manifest_sha = save(docs / 'prompts_manifest.json', {'rendered_prompts': prompts})
    snapshot_sha = save(docs / 'modelscope_snapshot.json', {})
    gate = dict(passed=True, sources_digest='670c744e0535dcad')
    runs, index = {}, []
    for case in ('AR', 'input', 'zero', 'one'):
        spec = {} if case == 'AR' else ({'method': 'esp'} if case == 'input' else
            {'method': 'output_hidden', 'output': {'order': int(case == 'one')}})
        settings = dict(task='ar_baseline' if case == 'AR' else 'esp_mtp', eval={'warmup': False})
        if spec:
            settings['spec'] = spec
        cfg = producer_config(settings)
        run_id = 'synthetic-' + case
        folder = study / 'runs' / run_id
        assets = folder / 'assets'
        assets.mkdir(parents=True)
        (folder / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        save(folder / 'result.json', {'status': 'succeeded', 'spec_mtp': gate})
        inputs = dict(dtype='torch.float32', backend='eager', device='cuda:0', settings=producer_settings(cfg),
                      prompts=[{'input_ids': p['input_ids']} for p in prompts],
                      model_digest='synthetic-model', runtime={'device': 'cuda:0'})
        if case == 'input' and settings_damage is not None:
            if settings_damage == 'raw_algorithm':
                inputs['settings'] = settings
            elif settings_damage == 'seed':
                inputs['settings']['run']['seed'] = 999
            elif settings_damage == 'device':
                inputs['settings']['eval']['device'] = 'cpu'
            elif settings_damage == 'data':
                inputs['settings']['data']['take_per_category'] = 99
            else:
                inputs['settings']['model']['torch_dtype'] = 'bfloat16'
        inputs_sha = save(assets / 'decode_inputs.json', inputs)
        save(assets / 'spec_mtp_sources.json', {'digest': gate['sources_digest'], 'config_digest': digest(cfg)})
        rows, evidence = [], []
        for p in prompts:
            accepted = {'AR': 0, 'input': 1, 'zero': 2, 'one': 0}[case]
            stats = dict(accepted=accepted, decode_calls=2, prefill_calls=1, new_tokens=2,
                         accept_by_depth=[0, accepted, 0], prefill_time=0, decode_time=0,
                         ar_time=0, ar_calls=2, ar_new_tokens=2)
            output = p['input_ids'] + [1, 2]
            evidence.append(dict(input_ids=p['input_ids'], output_ids=output, ar_ids=output, stats=stats))
            rows.append(dict(exact_match=1, spec_time=0, ar_time=0, ar_calls=2))
        raw = {'rows': rows, 'evidence': evidence}
        raw_sha = save(assets / 'decode_observations.json', {'inputs_digest': digest(inputs), 'out': raw})
        runs[case] = dict(run_id=run_id, raw_sha256=raw_sha, inputs_sha256=inputs_sha,
                         model_digest=inputs['model_digest'], runtime=inputs['runtime'], spec=spec, out=raw, gate=gate)
        index.append({'runs': [{'id': run_id}]})
    save(study / 'index.json', {'experiments': index})
    save(study / 'spec_mtp_study_check.json', {'complete': True, 'passed': True})
    save(study / 'process.json', {'complete': True, 'spec_mtp': {'passed': True}})
    body = dict(quality_passed=True, source_digest=gate['sources_digest'], model_digest='synthetic-model',
                runs=runs, manifest_sha256=manifest_sha, snapshot_sha256=snapshot_sha,
                tool_sha256='1' * 64, plan_sha256='2' * 64, **paired_4b.recompute_pairs(prompts, runs))
    path = docs / 'execution_v1.json'
    save(path, body)
    frozen = path.read_bytes()

    def pinned(repository, commit, member, sha):
        assert repository == ROOT and commit == 'a' * 40 and sha == hashlib.sha256(frozen).hexdigest()
        assert member == f'studies/{name}/docs/execution_v1.json'
        return frozen

    monkeypatch.setattr(provenance, 'read_historical_bytes', pinned)
    return dict(study_name=name, study_dir=study, execution_root=execution_root, execution=path,
                repository=ROOT, commit='a' * 40, execution_git_sha256=hashlib.sha256(frozen).hexdigest(),
                output_root=tmp_path / 'new-readout')


@pytest.mark.parametrize('name', ['output_space_4b_holdout', 'output_space_4b_recheck', 'output_space_4b_recheck_v2'])
def test_saved_three_arm_readout_uses_original_bytes_and_separate_output(tmp_path, monkeypatch, name):
    args = fixture_study(tmp_path, monkeypatch, name)
    before = args['execution'].read_bytes()
    body = reader.analyze(**args)
    assert body['paired_summaries'][0]['accepted_total_delta'] == 2
    assert body['paired_summaries'][1]['accepted_total_delta'] == -2
    assert body['total_generated_tokens_including_audits'] == 28
    output = args['output_root']
    scope = json.loads((output / 'readout_scope.json').read_bytes())
    assert scope['native_run_revalidated'] is False and scope['original_plan_bytes_verified'] is False
    assert scope['original_tool_sha256'] == '1' * 64
    binding = json.loads((output / 'readout_inputs.json').read_bytes())
    assert str(args['study_dir'] / 'runs/synthetic-input/assets/decode_inputs.json') in binding['input_sources']
    assert args['execution'].read_bytes() == before
    with pytest.raises(FileExistsError):
        reader.analyze(**args)


@pytest.mark.parametrize('damage', ['raw_bytes', 'input_bytes', 'source_identity', 'native_gate', 'unsafe_run', 'published_fields'])
def test_saved_readout_rejects_changed_native_inputs_without_output(tmp_path, monkeypatch, damage):
    args = fixture_study(tmp_path, monkeypatch, 'output_space_4b_recheck_v2')
    assets = args['study_dir'] / 'runs/synthetic-input/assets'
    if damage in ('raw_bytes', 'input_bytes'):
        path = assets / ('decode_observations.json' if damage == 'raw_bytes' else 'decode_inputs.json')
        path.write_bytes(path.read_bytes() + b' ')
    elif damage == 'source_identity':
        path = assets / 'spec_mtp_sources.json'
        data = json.loads(path.read_bytes()); data['digest'] = 'changed'
        path.write_text(json.dumps(data), encoding='utf-8')
    elif damage == 'native_gate':
        path = assets.parent / 'result.json'
        path.write_text(json.dumps({'status': 'failed', 'spec_mtp': {'passed': False}}), encoding='utf-8')
    elif damage == 'unsafe_run':
        path = args['study_dir'] / 'index.json'
        data = json.loads(path.read_bytes()); data['experiments'][0]['runs'][0]['id'] = '../escape'
        path.write_text(json.dumps(data), encoding='utf-8')
    else:
        data = json.loads(args['execution'].read_bytes()); data['new-field'] = 'not historical'
        args['execution'].write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises((ValueError, AssertionError)):
        reader.analyze(**args)
    assert not args['output_root'].exists()


@pytest.mark.parametrize('damage', ['raw_after_read', 'helper_after_read'])
def test_saved_readout_rejects_drift_before_publication(tmp_path, monkeypatch, damage):
    args = fixture_study(tmp_path, monkeypatch, 'output_space_4b_recheck_v2')
    raw = args['study_dir'] / 'runs/synthetic-input/assets/decode_observations.json'
    fake = tmp_path / 'fake-paired-helper.py'
    fake.write_text('before', encoding='utf-8')
    monkeypatch.setattr(paired_4b, '__file__', str(fake))
    original = paired_4b.recompute_pairs

    def change(*a, **kw):
        result = original(*a, **kw)
        target = raw if damage == 'raw_after_read' else fake
        target.write_bytes(target.read_bytes() + b' ')
        return result

    monkeypatch.setattr(paired_4b, 'recompute_pairs', change)
    with pytest.raises(ValueError, match='changed'):
        reader.analyze(**args)
    assert not args['output_root'].exists()


@pytest.mark.parametrize('damage', ['none', 'reused_bytes', 'missing_reused_group'])
def test_saved_input_control_binds_both_original_execution_groups(tmp_path, monkeypatch, damage):
    from spec_mtp.structure.artifact import provenance
    originals = [fixture_study(tmp_path, monkeypatch, name, count=count) for name, count in
                 (('output_space_4b_holdout', 4), ('output_space_4b_recheck_v2', 26))]
    execution_root = originals[0]['execution_root']
    study = execution_root / 'studies/output_space_4b_input_control'
    (study / 'docs').mkdir(parents=True)

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    groups, entries, previous = [], [], {}
    generated = forwards = 0
    for item in originals:
        name = item['study_name']
        old = json.loads(item['execution'].read_bytes())
        raw = json.loads(json.dumps(old['runs']['input']['out']))
        for evidence in raw['evidence']:
            evidence['stats']['hit_context_limit'] = False
        spec = {'method': 'ema_velocity', 'tree': {'mode': 'static', 'branches': [1, 1], 'pruning': False}}
        settings = {'task': 'esp_mtp', 'spec': spec, 'eval': {'warmup': False}}
        cfg = producer_config(settings)
        run_id = 'synthetic-ema-' + str(old['n'])
        folder = study / 'runs' / run_id
        assets = folder / 'assets'
        assets.mkdir(parents=True)
        (folder / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        inputs = dict(dtype='torch.float32', backend='eager', device='cuda:0', settings=producer_settings(cfg),
                      prompts=[{'input_ids': e['input_ids']} for e in raw['evidence']],
                      model_digest=old['model_digest'], runtime=old['runs']['AR']['runtime'])
        gate = {'passed': True, 'sources_digest': '670c744e0535dcad'}
        save(folder / 'result.json', {'status': 'succeeded', 'spec_mtp': gate})
        save(assets / 'spec_mtp_sources.json', {'digest': gate['sources_digest'], 'config_digest': digest(cfg)})
        inputs_sha = save(assets / 'decode_inputs.json', inputs)
        raw_sha = save(assets / 'decode_observations.json', {'inputs_digest': digest(inputs), 'out': raw})
        reused_sha = hashlib.sha256(item['execution'].read_bytes()).hexdigest()
        group, tokens, calls = paired_4b.recompute_control(old, raw, inputs, spec, run_id, name)
        group.update(inputs_sha256=inputs_sha, raw_sha256=raw_sha, reused_execution_sha256=reused_sha)
        groups.append(group); generated += tokens; forwards += calls
        entries.append({'runs': [{'id': run_id}]})
        previous[name] = {'execution_sha256': reused_sha}
    groups.sort(key=lambda group: -group['n'])
    save(study / 'index.json', {'experiments': entries})
    save(study / 'spec_mtp_study_check.json', {'complete': True, 'passed': True})
    save(study / 'process.json', {'complete': True, 'spec_mtp': {'passed': True}})
    save(study / 'docs/preceding_results.json', previous)
    body = {'quality_passed': True, 'source_digest': '670c744e0535dcad', 'groups': groups,
            'new_generated_tokens_including_audits': generated,
            'new_generation_forwards_including_audits': forwards, 'tool_sha256': '1' * 64, 'plan_sha256': '2' * 64}
    execution = study / 'docs/execution_v1.json'
    save(execution, body)
    frozen = execution.read_bytes()
    monkeypatch.setattr(provenance, 'read_historical_bytes', lambda *args: frozen)
    if damage == 'reused_bytes':
        oldpath = originals[0]['execution']
        oldpath.write_bytes(oldpath.read_bytes() + b' ')
    elif damage == 'missing_reused_group':
        save(study / 'docs/preceding_results.json', {'output_space_4b_holdout': previous['output_space_4b_holdout']})
    args = dict(study_name='output_space_4b_input_control', study_dir=study, execution_root=execution_root,
                execution=execution, repository=ROOT, commit='a' * 40,
                execution_git_sha256=hashlib.sha256(frozen).hexdigest(), output_root=tmp_path / 'control-output')
    if damage != 'none':
        with pytest.raises(AssertionError):
            reader.analyze(**args)
        assert not args['output_root'].exists()
    else:
        result = reader.analyze(**args)
        assert [group['n'] for group in result['groups']] == [26, 4]
        assert result['new_generated_tokens_including_audits'] == 120
        assert result['new_generation_forwards_including_audits'] == 150
        assert all(group['both_outputs_gain_over_EMA'] is False for group in result['groups'])


@pytest.mark.parametrize('damage', ['raw_algorithm', 'seed', 'device', 'data', 'model'])
def test_saved_run_requires_actual_producer_settings_schema(tmp_path, monkeypatch, damage):
    args = fixture_study(tmp_path, monkeypatch, 'output_space_4b_recheck_v2', settings_damage=damage)
    with pytest.raises(AssertionError):
        reader.analyze(**args)
    assert not args['output_root'].exists()
