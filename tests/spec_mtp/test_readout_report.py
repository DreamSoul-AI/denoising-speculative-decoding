"""v0.3.1 report identity/atomic contracts with the downstream quality gate."""

import copy
import json

import pytest

from conftest import cli
from rpipe.structure.artifact import _atomic
from rpipe.structure.artifact.readout.report import write_numbers


def _study(root):
    root.mkdir(exist_ok=True)
    exp = {'factors': {'case': 'baseline'}, 'runs': [{'id': 'abc123', 'seed': 123}]}
    index = {'study': 'contract', 'experiments': [exp]}
    process = copy.deepcopy(index)
    process.update(complete=False, scope='study')  # upstream allows a legitimate partial aggregation
    (root / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    (root / 'process.json').write_text(json.dumps(process), encoding='utf-8')
    (root / 'docs').mkdir()
    (root / 'docs/NUMBERS.md').write_bytes(b'previous numbers\n')
    return index, process


@pytest.mark.parametrize('fault', ['missing', 'corrupt', 'not_object', 'duplicate', 'missing_run',
                                  'study', 'id', 'seed', 'factors'])
def test_report_refuses_stale_or_invalid_identity_without_losing_numbers(tmp_path, fault):
    index, process = _study(tmp_path)
    before = (tmp_path / 'docs/NUMBERS.md').read_bytes()
    if fault == 'missing':
        (tmp_path / 'index.json').unlink()
    elif fault == 'corrupt':
        (tmp_path / 'index.json').write_text('{')
    elif fault == 'not_object':
        (tmp_path / 'index.json').write_text('[]')
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
        (tmp_path / 'index.json').write_text(json.dumps(index))
        (tmp_path / 'process.json').write_text(json.dumps(process))
    raw_process = (tmp_path / 'process.json').read_bytes()
    with pytest.raises(ValueError):
        write_numbers(tmp_path)
    assert (tmp_path / 'docs/NUMBERS.md').read_bytes() == before
    assert (tmp_path / 'process.json').read_bytes() == raw_process


def test_upstream_allows_identity_complete_partial_aggregate(tmp_path):
    _study(tmp_path)
    target = write_numbers(tmp_path)
    assert target == tmp_path / 'docs/NUMBERS.md'
    assert 'abc123' in target.read_text() and '123' in target.read_text()


def test_failed_atomic_replace_preserves_prior_numbers_and_process(tmp_path, monkeypatch):
    _study(tmp_path)
    numbers = (tmp_path / 'docs/NUMBERS.md').read_bytes()
    process = (tmp_path / 'process.json').read_bytes()

    def refuse_replace(source, target):
        raise PermissionError('injected unavailable target')

    monkeypatch.setattr(_atomic, 'atomic_replace', refuse_replace)
    with pytest.raises(PermissionError):
        write_numbers(tmp_path)
    assert (tmp_path / 'docs/NUMBERS.md').read_bytes() == numbers
    assert (tmp_path / 'process.json').read_bytes() == process
    assert not list((tmp_path / 'docs').glob('*.tmp'))


def test_spec_cli_report_keeps_upstream_identity_and_specialized_evidence_gates(study_copy):
    study = study_copy(lambda s: s.update(axes={'algorithm.config.variant': ['esp_static']}),
                       name='v031_report')
    cli('make', study, '--num-gpus', '0', check=True)
    cli('launch', study, '--num-gpus', '0', '--console', 'shared', check=True)
    cli('report', study, check=True)
    path = study / 'docs/NUMBERS.md'
    previous = path.read_bytes()
    process_path = study / 'process.json'
    process = json.loads(process_path.read_text())
    process['experiments'][0]['runs'][0]['seed'] = -1
    process_path.write_text(json.dumps(process))
    stale_process = process_path.read_bytes()
    result = cli('report', study)
    assert process_path.read_bytes() == stale_process
    assert result.returncode != 0 and 'current index' in result.stdout + result.stderr
    assert path.read_bytes() == previous
    cli('process', study, check=True)
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    (study / 'runs' / run_id / 'assets/decode_observations.json').unlink()
    result = cli('report', study)
    assert result.returncode != 0
    assert path.read_bytes() == previous
    envelope = json.loads(process_path.read_text())
    assert envelope['complete'] is False
