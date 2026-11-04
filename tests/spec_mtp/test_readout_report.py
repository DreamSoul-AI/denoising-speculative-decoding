"""Explicit report exports preserve the saved identity, inputs and prior output."""
import copy
import json
from pathlib import Path

import pytest

from conftest import cli
from spec_mtp.structure.artifact.readout.report import write_numbers


def _study(root):
    root.mkdir(exist_ok=True)
    exp = {'factors': {'case': 'baseline'}, 'runs': [{'id': 'abc123', 'seed': 123}]}
    index = {'study': 'contract', 'experiments': [exp]}
    process = copy.deepcopy(index)
    process.update(complete=False, scope='study')
    (root / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    (root / 'process.json').write_text(json.dumps(process), encoding='utf-8')
    (root / 'docs').mkdir()
    (root / 'docs/NUMBERS.md').write_bytes(b'previous numbers\n')
    return index, process


@pytest.mark.parametrize('fault', ['missing', 'corrupt', 'not_object', 'duplicate', 'missing_run',
                                  'study', 'id', 'seed', 'factors'])
def test_report_refuses_stale_or_invalid_identity_without_losing_numbers(tmp_path, fault):
    study = tmp_path / 'study'
    index, process = _study(study)
    before = (study / 'docs/NUMBERS.md').read_bytes()
    if fault == 'missing':
        (study / 'index.json').unlink()
    elif fault == 'corrupt':
        (study / 'index.json').write_text('{')
    elif fault == 'not_object':
        (study / 'index.json').write_text('[]')
    else:
        if fault == 'duplicate':
            index['experiments'][0]['runs'].append(dict(index['experiments'][0]['runs'][0]))
        elif fault == 'missing_run':
            process['experiments'][0]['runs'] = []
        elif fault == 'study':
            process['study'] = 'old_version'
        elif fault == 'id':
            process['experiments'][0]['runs'][0]['id'] = 'old123'
        elif fault == 'seed':
            process['experiments'][0]['runs'][0]['seed'] = 1
        elif fault == 'factors':
            process['experiments'][0]['factors'] = {'case': 'other'}
        (study / 'index.json').write_text(json.dumps(index))
        (study / 'process.json').write_text(json.dumps(process))
    raw_process = (study / 'process.json').read_bytes()
    output = tmp_path / 'export/numbers.md'
    with pytest.raises(ValueError):
        write_numbers(study, output)
    assert not output.exists()
    assert (study / 'docs/NUMBERS.md').read_bytes() == before
    assert (study / 'process.json').read_bytes() == raw_process


def test_upstream_allows_identity_complete_partial_aggregate(tmp_path):
    study = tmp_path / 'study'
    _study(study)
    previous = {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}
    target = write_numbers(study, tmp_path / 'export/numbers.md')
    assert target == tmp_path / 'export/numbers.md'
    text = target.read_text()
    assert 'abc123' in text and '123' in text
    assert '../study/runs/abc123/assets/logs/events.jsonl' in text
    assert 'STUDY_REPORT.md' not in text
    assert previous == {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}


def test_existing_output_preserves_prior_numbers_and_process(tmp_path):
    study = tmp_path / 'study'
    _study(study)
    target = tmp_path / 'export/numbers.md'
    target.parent.mkdir()
    target.write_bytes(b'previous export\n')
    process = (study / 'process.json').read_bytes()
    with pytest.raises(FileExistsError):
        write_numbers(study, target)
    assert target.read_bytes() == b'previous export\n'
    assert (study / 'process.json').read_bytes() == process


def test_report_write_failure_does_not_modify_saved_inputs(tmp_path, monkeypatch):
    study = tmp_path / 'study'
    _study(study)
    target = tmp_path / 'export/numbers.md'
    process = (study / 'process.json').read_bytes()
    original = Path.open
    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError('injected unavailable target')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', denied)
        with pytest.raises(PermissionError):
            write_numbers(study, target)
    assert not target.exists()
    assert (study / 'process.json').read_bytes() == process


@pytest.mark.parametrize('placement', ['study', 'study_parent', 'source'])
def test_report_rejects_output_that_overlaps_inputs_or_source(tmp_path, placement):
    study = tmp_path / 'study'
    _study(study)
    targets = {'study': study / 'new.md', 'study_parent': tmp_path / 'new.md',
               'source': Path(__file__).resolve().parents[2] / 'new.md'}
    with pytest.raises(ValueError, match='separate'):
        write_numbers(study, targets[placement])
    assert not targets[placement].exists()


def test_spec_cli_report_keeps_upstream_identity_and_specialized_evidence_gates(study_copy):
    study = study_copy(lambda s: s.update(axes={'algorithm.config.variant': ['esp_static']}),
                       name='v031_report')
    cli('make', study, '--num-gpus', '0', check=True)
    cli('launch', study, '--num-gpus', '0', '--console', 'shared', check=True)
    path = study.parent / 'export/numbers.md'
    process_path = study / 'process.json'
    initial = process_path.read_bytes()
    assert cli('report', study).returncode != 0
    cli('report', study, '--out', path, check=True)
    assert process_path.read_bytes() == initial
    previous = path.read_bytes()
    process = json.loads(process_path.read_text())
    process['experiments'][0]['runs'][0]['seed'] = -1
    process_path.write_text(json.dumps(process))
    stale_process = process_path.read_bytes()
    result = cli('report', study, '--out', study.parent / 'other/numbers.md')
    assert process_path.read_bytes() == stale_process
    assert result.returncode != 0 and 'current index' in result.stdout + result.stderr
    assert path.read_bytes() == previous
    cli('process', study, check=True)
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    (study / 'runs' / run_id / 'assets/decode_observations.json').unlink()
    initial = process_path.read_bytes()
    result = cli('report', study, '--out', study.parent / 'rejected/numbers.md')
    assert result.returncode != 0 and process_path.read_bytes() == initial
    assert path.read_bytes() == previous


def test_report_quality_failure_does_not_rewrite_process_or_gate(tmp_path):
    from spec_mtp.flow.context import StudyProcessContext
    from studies._shared.flow.process.run import study_gate
    from studies._shared.structure.validation import SpecMtpGateError

    study = tmp_path / 'study'
    _study(study)
    before = {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}
    context = StudyProcessContext(study, state={'process': json.loads((study / 'process.json').read_text())})
    with pytest.raises(SpecMtpGateError, match='read-only'):
        study_gate(context, persist=False)
    assert before == {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}
