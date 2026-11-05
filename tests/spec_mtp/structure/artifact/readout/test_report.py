"""Explicit exports preserve saved identity and never rewrite read-only inputs."""
import copy
import json
from pathlib import Path

import pytest

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
               'source': Path(__file__).resolve().parents[5] / 'new.md'}
    with pytest.raises(ValueError, match='separate'):
        write_numbers(study, targets[placement])
    assert not targets[placement].exists()
