"""Pinned saved JSON admission and publication without tensors or model execution."""

import hashlib
import json
from pathlib import Path

import pytest

from studies.output_space_cross_task.flow.process.run import analyze, main, protocol, PRODUCERS, PLAN_ROOT


@pytest.fixture
def saved_index(tmp_path, monkeypatch):
    from spec_mtp.structure.artifact import provenance

    root = tmp_path / 'inputs'
    root.mkdir()
    repository = root / 'repository'
    repository.mkdir()
    blobs = {}
    def entry(name, value):
        path = root / name
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(data)
        member = 'saved/' + name
        blobs[member] = data
        return {'path': str(path), 'member': member, 'git_sha256': hashlib.sha256(data).hexdigest()}
    tool, plan = entry('producer.py', b'# actual old producer bytes\n'), entry('producer-plan.md', b'# frozen old plan\n')
    producer = {'tool': tool['git_sha256'], 'plan': plan['git_sha256']}
    reports = {name: entry(name + '.json', {'quality_passed': True, 'source_digest': protocol.SOURCE_DIGEST,
        'source_binding': producer, 'units': [{'qid': qid, 'seed': 123, 'selection': {'intervention_root': None}}]})
        for name, qid in [('new', 242), ('old', 241)]}
    reports['summary'] = entry('summary.json', {'quality_passed': True})
    producer_sources = {}
    for role, names in PRODUCERS['first-delta'].items():
        producer_sources[role] = {'tool': dict(tool), 'plan': dict(plan)}
        for claim, member in [('tool', 'studies/_shared/structure/' + names[0]), ('plan', PLAN_ROOT + names[1])]:
            original = producer_sources[role][claim]['member']
            blobs[member] = blobs[original]
            producer_sources[role][claim]['member'] = member
    analysis_plan = entry('analysis-plan.md', b'# actual original first-delta plan\n')
    blobs[PLAN_ROOT + 'FIRST_DELTA_PLAN.md'] = blobs[analysis_plan['member']]
    analysis_plan['member'] = PLAN_ROOT + 'FIRST_DELTA_PLAN.md'
    analysis_plan['sha256'] = analysis_plan['git_sha256']
    index = {'schema': 1, 'kind': 'first-delta', 'repository': str(repository), 'commit': 'a' * 40,
        'reports': reports, 'files': {'analysis_plan': analysis_plan},
        'producer_sources': producer_sources}
    path = root / 'index.json'
    def save():
        path.write_text(json.dumps(index), encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []
    def historical(repository, commit, member, sha):
        calls.append(member)
        data = blobs[member]
        assert hashlib.sha256(data).hexdigest() == sha
        return data
    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    return {'root': root, 'path': path, 'index': index, 'save': save, 'blobs': blobs,
            'calls': calls, 'output': tmp_path / 'readout', 'historical': historical}


def test_first_delta_reads_pinned_producer_bytes_and_preserves_infeasible_units(saved_index):
    case = saved_index
    sha = case['save']()
    before = {path: path.read_bytes() for path in case['root'].rglob('*') if path.is_file()}
    result = analyze('first-delta', case['path'], sha, case['output'])
    assert [row['qid'] for row in result['rows']] == [241, 242]
    assert all(row['status'] == 'infeasible_not_scored' for row in result['rows'])
    assert result['readout_model_forwards'] == result['readout_sampling_draws'] == 0
    binding = json.loads((case['output'] / 'readout_inputs.json').read_text())
    assert binding['historical_sources'] and binding['readout_sources']
    assert binding['output_sources']['first_delta_analysis_v1.json']
    assert before == {path: path.read_bytes() for path in case['root'].rglob('*') if path.is_file()}


@pytest.mark.parametrize('fault', ['index_sha', 'quality', 'digest', 'git_report', 'producer_bytes', 'plan_hash', 'unpinned_plan', 'existing'])
def test_first_delta_rejects_unbound_history_before_output(saved_index, fault):
    case, index = saved_index, saved_index['index']
    if fault in ('quality', 'digest'):
        report = json.loads(case['blobs']['saved/new.json'])
        report['quality_passed' if fault == 'quality' else 'source_digest'] = False if fault == 'quality' else 'other'
        data = json.dumps(report).encode()
        case['blobs']['saved/new.json'] = data
        Path(index['reports']['new']['path']).write_bytes(data)
        index['reports']['new']['git_sha256'] = hashlib.sha256(data).hexdigest()
    elif fault == 'git_report':
        Path(index['reports']['new']['path']).write_text('{}')
    elif fault == 'producer_bytes':
        Path(index['producer_sources']['new']['tool']['path']).write_bytes(b'changed old producer')
    elif fault == 'plan_hash':
        index['files']['analysis_plan']['sha256'] = '0' * 64
    elif fault == 'unpinned_plan':
        index['files']['analysis_plan'].pop('member')
        index['files']['analysis_plan'].pop('git_sha256')
    elif fault == 'existing':
        case['output'].mkdir()
        (case['output'] / 'keep').write_bytes(b'old output')
    sha = case['save']()
    with pytest.raises((ValueError, FileExistsError)):
        analyze('first-delta', case['path'], '0' * 64 if fault == 'index_sha' else sha, case['output'])
    assert not case['output'].exists() if fault != 'existing' else (case['output'] / 'keep').read_bytes() == b'old output'


def test_first_delta_uses_initial_resolved_report_when_original_alias_changes(saved_index, monkeypatch):
    from spec_mtp.structure.artifact import provenance

    case = saved_index
    alias = case['root'] / 'new-alias.json'
    original = Path(case['index']['reports']['new']['path'])
    replacement = case['root'] / 'replacement.json'
    replacement.write_text('{}')
    target = [original]
    resolve = Path.resolve
    def resolved(path, *args, **options):
        return target[0] if path == alias else resolve(path, *args, **options)
    monkeypatch.setattr(Path, 'resolve', resolved)
    case['index']['reports']['new']['path'] = str(alias)
    def historical(*args):
        target[0] = replacement
        return case['historical'](*args)
    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    result = analyze('first-delta', case['path'], case['save'](), case['output'])
    assert [row['qid'] for row in result['rows']] == [241, 242]


@pytest.mark.parametrize('missing', ['--inputs', '--inputs-sha256', '--output-root'])
def test_saved_sampling_cli_requires_explicit_binding(missing):
    values = ['first-delta']
    for flag in ('--inputs', '--inputs-sha256', '--output-root'):
        if flag != missing:
            values.extend((flag, 'fixture'))
    with pytest.raises(SystemExit) as error:
        main(values)
    assert error.value.code == 2
