"""cpu pure IO/stats for native saved-byte trust and synthetic verdicts."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from studies.output_space_cross_task.flow.process.native import analyze
from studies.output_space_cross_task.structure import native_readout as protocol
from studies._shared.structure.config import legacy_cfg
from studies._shared.structure.algorithm import effective_spec
from studies._shared.structure.evidence import digest

ROOT = Path(__file__).resolve().parents[5]


def saved_index(tmp_path, monkeypatch):
    from spec_mtp.structure.artifact import provenance
    native, _ = protocol.fixture()
    native.update(version='synthetic-version', source_digest='670c744e0535dcad', model_digest='synthetic-model')
    files, blobs, experiments = {}, {}, []

    def save(role, value, *, member=None, text=False):
        path = tmp_path / 'inputs' / (role.replace(':', '-') + ('.txt' if text else '.json'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if text else json.dumps(value), encoding='utf-8')
        entry = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if member:
            entry.update(member=member, git_sha256=entry['sha256'])
            blobs[member] = path.read_bytes()
        files[role] = entry
        return entry['sha256']

    native['tool_sha256'] = save('native_tool', 'synthetic original exporter bytes; never execute',
        member='studies/output_space_cross_task/check_hypotheses.py', text=True)
    for i, run in enumerate(native['runs']):
        run_id, case = 'synthetic-' + str(i), run['case']
        spec = {} if case == 'AR' else ({'method': 'esp'} if case == 'frozen_mean_chain' else
            {'method': 'output_hidden', 'output': {'order': int(case == 'output_order1')}})
        settings = {'version': 'synthetic-version', 'task': 'ar_baseline' if case == 'AR' else 'esp_mtp',
                    'eval': {'warmup': False}, 'spec': spec}
        cfg = {'seed': 123, 'algorithm': {'config': settings}, 'system': {'config': {'device': 'cuda'}}}
        algorithm = SimpleNamespace(setting=lambda key, default=None: settings.get(key, default))
        captured = legacy_cfg(SimpleNamespace(meta={'data_cfg': {'name': 'synthetic'}}),
            SimpleNamespace(meta={'model_cfg': {'type': 'synthetic'}}), SimpleNamespace(device='cuda'), algorithm, 123)
        captured.spec = effective_spec(captured.spec, algorithm)
        inputs = {'settings': captured, 'device': 'cuda:0', 'eos_ids': [], 'model_digest': 'synthetic-model',
                  'runtime': {'synthetic': True}, 'prompts': [{'input_ids': [0]} for _ in protocol.QIDS]}
        for evidence in run['raw']['evidence']:
            evidence['input_ids'] = [0]
        save(run_id + ':inputs', inputs)
        raw_sha = save(run_id + ':raw', {'inputs_digest': digest(inputs), 'out': run['raw']})
        save(run_id + ':config', yaml.safe_dump(cfg), text=True)
        save(run_id + ':result', {'status': 'succeeded', 'spec_mtp': {'passed': True, 'sources_digest': native['source_digest']}})
        save(run_id + ':sources', {'digest': native['source_digest'], 'config_digest': digest(cfg)})
        run.update(run_id=run_id, spec=spec, raw_sha256=raw_sha, runtime=inputs['runtime'], model_digest=inputs['model_digest'])
        experiments.append({'runs': [{'id': run_id}]})
    save('index', {'experiments': experiments})
    save('process', {'complete': True, 'spec_mtp': {'passed': True}})
    save('manifest', {'rendered_prompts': [{'question_id': qid, 'input_ids': [0]} for qid in protocol.QIDS]})
    member = 'studies/output_space_cross_task/docs/execution_v1.json'
    save('native', native, member=member)
    report = files.pop('native')
    report.pop('sha256')
    index = {'schema': 1, 'kind': 'export', 'repository': str(ROOT), 'commit': 'a' * 40,
             'reports': {'native': report}, 'files': files, 'producer_sources': {}}
    path = tmp_path / 'control' / 'index.json'
    path.parent.mkdir()
    path.write_text(json.dumps(index), encoding='utf-8')

    def historical(repo, commit, name, expected):
        assert repo == ROOT and commit in ('a' * 40, 'b' * 40)
        raw = blobs[name]
        assert hashlib.sha256(raw).hexdigest() == expected
        return raw

    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    return path, index, blobs


def test_native_export_reads_original_observation_bound_inputs_without_resigning(tmp_path, monkeypatch):
    path, _, _ = saved_index(tmp_path, monkeypatch)
    before = path.read_bytes()
    output = tmp_path / 'output'
    result = analyze('export', path, hashlib.sha256(before).hexdigest(), output)
    assert result['question_ids'] == protocol.QIDS
    assert len(result['runs']) == 4
    assert json.loads((output / 'native_scope.json').read_bytes())['native_run_revalidated'] is False
    assert (output / 'window_selection_readout.json').is_file()
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        analyze('export', path, hashlib.sha256(before).hexdigest(), output)


@pytest.mark.parametrize('damage', ['wrong_index', 'input_digest', 'missing_exporter', 'wrong_role'])
def test_native_admission_rejects_unbound_inputs_before_outputs(tmp_path, monkeypatch, damage):
    path, index, _ = saved_index(tmp_path, monkeypatch)
    if damage == 'input_digest':
        entry = index['files']['synthetic-0:inputs']
        raw = json.loads(Path(entry['path']).read_bytes()); raw['eos_ids'] = [999]
        Path(entry['path']).write_text(json.dumps(raw), encoding='utf-8')
        entry['sha256'] = hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()
    elif damage == 'missing_exporter':
        index['files'].pop('native_tool')
    elif damage == 'wrong_role':
        index['reports']['native']['member'] = 'studies/output_space_cross_task/docs/canonical_diagnosis_v1.json'
    path.write_text(json.dumps(index), encoding='utf-8')
    output = tmp_path / 'output'
    sha = '0' * 64 if damage == 'wrong_index' else hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, AssertionError, KeyError)):
        analyze('export', path, sha, output)
    assert not output.exists()


def test_synthetic_native_mre_is_json_only_and_refuses_device(tmp_path):
    result = analyze('mre', None, None, tmp_path / 'mre')
    assert result['synthetic_only_no_model_evidence'] and result['verdict']['H2'].startswith('not_tested')
    with pytest.raises(ValueError):
        analyze('mre', None, None, tmp_path / 'bad', device='cpu')
    assert not (tmp_path / 'bad').exists()


def test_native_dispatch_preserves_sampling_and_state_commands(monkeypatch):
    driver = importlib.import_module('studies.output_space_cross_task.flow.process.run')
    target = importlib.import_module('studies.output_space_cross_task.flow.process.native')
    seen = []
    monkeypatch.setattr(target, 'main', lambda argv: seen.append(argv) or 17)
    assert driver.main(['native', 'mre', '--output-root', 'unused']) == 17
    assert seen == [['mre', '--output-root', 'unused']]


@pytest.mark.parametrize('wrong_member', [False, True])
def test_native_exporter_source_records_its_own_commit_and_rejects_wrong_role(tmp_path, monkeypatch, wrong_member):
    path, index, blobs = saved_index(tmp_path, monkeypatch)
    entry = index['files']['native_tool']
    entry['commit'] = 'b' * 40
    if wrong_member:
        old = entry['member']
        entry['member'] = 'studies/output_space_cross_task/analyze_canonical.py'
        blobs[entry['member']] = blobs[old]
    path.write_text(json.dumps(index), encoding='utf-8')
    output = tmp_path / 'output'
    if wrong_member:
        with pytest.raises(ValueError, match='exporter'):
            analyze('export', path, hashlib.sha256(path.read_bytes()).hexdigest(), output)
        assert not output.exists()
    else:
        analyze('export', path, hashlib.sha256(path.read_bytes()).hexdigest(), output)
        binding = json.loads((output / 'readout_inputs.json').read_bytes())
        assert {'commit': 'b' * 40, 'path': entry['member'], 'sha256': entry['git_sha256']} in binding['historical_sources']
