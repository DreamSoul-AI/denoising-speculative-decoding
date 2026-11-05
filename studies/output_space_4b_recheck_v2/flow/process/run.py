from studies._shared.flow.process import run
__all__ = ['run']

from pathlib import Path
import json
from studies.esp_table1_bf16_full_argmax.flow.process.run import SavedReadout
from studies._shared.structure import paired_4b

STUDIES = ('output_space_4b_holdout', 'output_space_4b_recheck',
           'output_space_4b_recheck_v2', 'output_space_4b_input_control')


class PairedReadout(SavedReadout):
    """Use the existing byte-bound reader/publisher with this 4B theme's own source set."""
    def _source_files(self):
        import inspect
        from spec_mtp.structure.artifact.provenance import file_sha256
        from studies._shared.structure import algorithm, config, evidence, sources
        checkout = Path(__file__).resolve().parents[4]
        paths = [path for name in STUDIES for pillar in ('structure', 'flow')
                 for path in (checkout / 'studies' / name / pillar).rglob('*.py')]
        paths += [Path(paired_4b.__file__), Path(inspect.getfile(SavedReadout)),
                  Path(evidence.__file__), Path(sources.__file__),
                  Path(config.__file__), Path(algorithm.__file__)]
        return {str(path.resolve()): file_sha256(path) for path in paths}


def _case(settings):
    if settings['task'] == 'ar_baseline':
        return 'AR'
    spec = settings['spec']
    assert settings['task'] == 'esp_mtp'
    if spec['method'] == 'esp':
        return 'input'
    assert spec['method'] == 'output_hidden' and spec['output']['order'] in (0, 1)
    return 'zero' if spec['output']['order'] == 0 else 'one'


def _saved_run(session, folder, original, *, control=False):
    import yaml
    from studies._shared.structure.evidence import digest

    cfg = yaml.safe_load(session.read_text(folder / 'config.yaml'))
    result = session.read_json(folder / 'result.json')
    assets = folder / 'assets'
    inputs = session.read_json(assets / 'decode_inputs.json')
    observation = session.read_json(assets / 'decode_observations.json')
    sources = session.read_json(assets / 'spec_mtp_sources.json')
    assert result['status'] == 'succeeded' and result['spec_mtp']['passed']
    assert sources['digest'] == result['spec_mtp']['sources_digest'] == '670c744e0535dcad'
    assert sources['config_digest'] == digest(cfg) and observation['inputs_digest'] == digest(inputs)
    assert inputs['dtype'] == 'torch.float32' and inputs['backend'] == 'eager'
    assert cfg['algorithm']['config']['eval']['warmup'] is False
    assert inputs['settings'] == _recorded_settings(cfg, inputs)
    assert session.sha(assets / 'decode_inputs.json') == original['inputs_sha256']
    assert session.sha(assets / 'decode_observations.json') == original['raw_sha256']
    raw = observation['out']
    assert raw == original['raw' if control else 'out']
    return cfg, result, inputs, raw


def _recorded_settings(cfg, inputs):
    """Rebuild the producer schema from recorded device/materialization metadata, without loading files."""
    from types import SimpleNamespace
    from studies._shared.structure.config import legacy_cfg
    from studies._shared.structure.algorithm import effective_spec

    saved = inputs['settings']
    assert set(saved) == {'run', 'data', 'model', 'eval', 'spec'}
    device = saved['eval']['device']
    assert isinstance(device, str) and device == str(cfg['system']['config']['device'])
    actual_device = inputs['device']
    assert actual_device == device or (device == 'cuda' and actual_device.startswith('cuda:'))
    data, model = saved['data'], saved['model']
    assert isinstance(data, dict) and isinstance(model, dict)
    assert data['name'] == str(cfg['data']['name']).lower()
    assert model['type'] == cfg['model']['name']
    # Materialized paths retain their recorded identity; no model/data file is reopened here.
    for key, value in (cfg['data'].get('config') or {}).items():
        if key not in ('path', 'source_path'):
            assert data.get(key) == value
    for key, value in (cfg['model'].get('config') or {}).items():
        if key not in ('name_or_path', 'cache_dir', 'local_files_only'):
            assert model.get(key) == value
    settings = cfg['algorithm']['config']
    algorithm = SimpleNamespace(setting=lambda key, default=None: settings.get(key, default))
    result = legacy_cfg(SimpleNamespace(meta={'data_cfg': data}), SimpleNamespace(meta={'model_cfg': model}),
                        SimpleNamespace(device=device), algorithm, cfg['seed'])
    result.spec = effective_spec(result.spec, algorithm)
    return result


def analyze(study_name, study_dir, execution_root, execution, repository, commit, execution_git_sha256, output_root):
    """Reaggregate pinned saved scalars/tokens; never replay or re-sign a native source gate."""
    from spec_mtp.structure.artifact.provenance import read_historical_bytes

    if study_name not in STUDIES:
        raise ValueError('unknown 4B paired Study')
    study = Path(study_dir).resolve(strict=True)
    native = Path(execution_root).resolve(strict=True) / 'studies' / study_name
    execution = Path(execution).resolve(strict=True)
    repository = Path(repository).resolve(strict=True)
    session = PairedReadout(study, output_root, Path(execution_root).resolve(strict=True),
                            native, execution.parent, repository)
    member = f'studies/{study_name}/docs/execution_v1.json'
    original = session.read_json(execution)
    if original != json.loads(read_historical_bytes(repository, commit, member, execution_git_sha256)):
        raise ValueError('runtime execution differs from the explicitly pinned Git report')
    assert original['quality_passed'] and original['source_digest'] == '670c744e0535dcad'
    session.historical.append({'commit': commit, 'path': member, 'sha256': execution_git_sha256,
                               'runtime_sha256': session.sha(execution)})
    quality = session.read_json(study / 'spec_mtp_study_check.json')
    process = session.read_json(study / 'process.json')
    assert quality['passed'] and quality['complete'] and process['complete'] and process['spec_mtp']['passed']
    index = session.read_json(study / 'index.json')['experiments']
    ids = [item['runs'][0]['id'] for item in index if len(item['runs']) == 1]
    if (len(ids) != len(index) or len(set(ids)) != len(ids)
            or any(not isinstance(item, str) or Path(item).name != item or item in ('.', '..') for item in ids)):
        raise ValueError('saved paired index requires unique safe single-run IDs')
    if study_name == 'output_space_4b_input_control':
        body = _control(session, study, native, Path(execution_root).resolve(strict=True), index, original)
    else:
        body = _three_arm(session, study, native, index, original)
    # Every scientific field remains bound to the original published report.
    session.write_text(session.output / 'paired_readout.json', json.dumps(body, indent=2, allow_nan=False) + '\n')
    session.write_text(session.output / 'readout_scope.json', json.dumps({
        'scope': 'saved scalar/token/hash reaggregation only; no model forwards or new native acceptance',
        'native_run_revalidated': False, 'original_plan_sha256': original.get('plan_sha256'),
        'original_plan_bytes_verified': False, 'original_tool_sha256': original.get('tool_sha256'),
        'original_tool_bytes_verified': False, 'source_digest': original['source_digest'],
    }, indent=2) + '\n')
    session.finish()
    return body


def _three_arm(session, study, native, index, original):
    assert len(index) == 4
    prompts = session.read_json(study / 'docs/prompts_manifest.json')['rendered_prompts']
    assert session.sha(study / 'docs/prompts_manifest.json') == original['manifest_sha256']
    assert session.sha(study / 'docs/modelscope_snapshot.json') == original['snapshot_sha256']
    assert len({p['question_id'] for p in prompts}) == len(prompts)
    runs = {}
    for exp in index:
        run_id = exp['runs'][0]['id']
        names = [name for name, item in original['runs'].items() if item['run_id'] == run_id]
        assert len(names) == 1
        name = names[0]
        cfg, result, inputs, raw = _saved_run(session, native / 'runs' / run_id, original['runs'][name])
        assert _case(cfg['algorithm']['config']) == name and name not in runs
        assert len(prompts) == len(inputs['prompts']) == len(raw['rows']) == len(raw['evidence'])
        for p, q, e in zip(prompts, inputs['prompts'], raw['evidence']):
            assert p['input_ids'] == q['input_ids'] == e['input_ids']
            assert not e['stats'].get('hit_context_limit', False)
            if name == 'AR':
                assert e['stats']['ar_time'] == 0
            else:
                assert e['stats']['prefill_time'] == e['stats']['decode_time'] == 0
        item = original['runs'][name]
        assert inputs['model_digest'] == item['model_digest'] and inputs['runtime'] == item['runtime']
        assert result['spec_mtp'] == item['gate'] and cfg['algorithm']['config'].get('spec', {}) == item['spec']
        runs[name] = dict(item, out=raw)
    assert set(runs) == {'input', 'zero', 'one', 'AR'}
    assert len({r['model_digest'] for r in runs.values()}) == 1
    assert all(r['runtime'] == runs['AR']['runtime'] for r in runs.values())
    derived = paired_4b.recompute_pairs(prompts, runs)
    if any(original[key] != value for key, value in derived.items()):
        raise ValueError('saved paired counts differ from the pinned report')
    return dict(original, **derived)


def _control(session, study, native, execution_root, index, original):
    previous = session.read_json(study / 'docs/preceding_results.json')
    pending = set(previous)
    groups = []
    generated = forwards = 0
    for exp in index:
        run_id = exp['runs'][0]['id']
        bound = [group for group in original['groups'] if group['run_id'] == run_id]
        assert len(bound) == 1
        record = bound[0]
        cfg, _, inputs, raw = _saved_run(session, native / 'runs' / run_id, record, control=True)
        spec = cfg['algorithm']['config']['spec']
        assert spec['method'] == 'ema_velocity' and spec['tree'] == {'mode': 'static', 'branches': [1, 1], 'pruning': False}
        name = 'output_space_4b_recheck_v2' if len(raw['rows']) == 26 else 'output_space_4b_holdout'
        assert name in pending and name == record['reused_study']
        pending.remove(name)
        path = execution_root / 'studies' / name / 'docs/execution_v1.json'
        assert session.sha(path) == previous[name]['execution_sha256'] == record['reused_execution_sha256']
        old = session.read_json(path)
        assert old['quality_passed'] and inputs['model_digest'] == old['model_digest']
        assert inputs['runtime'] == old['runs']['AR']['runtime']
        assert len(raw['rows']) == len(raw['evidence']) == len(old['paired_rows']) == len(inputs['prompts'])
        group, tokens, calls = paired_4b.recompute_control(old, raw, inputs, spec, run_id, name)
        group.update(inputs_sha256=record['inputs_sha256'], raw_sha256=record['raw_sha256'],
                     reused_execution_sha256=record['reused_execution_sha256'])
        assert group == record
        groups.append(group)
        generated += tokens
        forwards += calls
    assert not pending
    groups.sort(key=lambda group: -group['n'])
    assert groups == original['groups']
    assert generated == original['new_generated_tokens_including_audits']
    assert forwards == original['new_generation_forwards_including_audits']
    return dict(original, groups=groups)


def main(argv=None, *, study_name='output_space_4b_recheck_v2'):
    import argparse
    parser = argparse.ArgumentParser(description='cpu pure IO/stats of explicitly bound historical 4B pairs')
    for name in ('study-dir', 'execution-root', 'execution', 'repository', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--execution-git-sha256', required=True)
    args = parser.parse_args(argv)
    analyze(study_name, args.study_dir, args.execution_root, args.execution, args.repository,
            args.commit, args.execution_git_sha256, args.output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
