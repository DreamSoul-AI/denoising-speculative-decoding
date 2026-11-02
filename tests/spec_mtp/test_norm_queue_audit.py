"""Per-cell accounting calls only the existing readout, never GPU execution."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('norm_queue_audit_test', ROOT / 'studies/_shared/structure/audit_4b_norm_queue.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    study = tmp_path / 'study'
    (study / 'docs').mkdir(parents=True)
    (study / 'docs/source_refs.json').write_text('{}')
    reader = study / 'analyze_results.py'
    reader.write_text('fixture')
    entries = [dict(factors={'algorithm.config': {'direction': name, 'spec': {'ema': {'step_scale': scale}}}}, runs=[{'id': str(i)}])
               for i, (name, scale) in enumerate((n, s) for n in ('template', 'random0', 'random1', 'random2') for s in (.5, 2.))]
    (study / 'index.json').write_text(json.dumps({'experiments': entries}))
    monkeypatch.setattr(m, 'STUDY', study)
    monkeypatch.setattr(m, 'READER', reader)
    return m, study, tmp_path / 'out'


def test_pending_and_native_failure_do_not_run_a_model_or_readout(setup, monkeypatch):
    m, study, output = setup
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('No new completion to audit'))
    state = m.audit_once('unused', output)
    assert state['native_completed'] == 0 and not state['complete']
    folder = study / 'runs/0'
    folder.mkdir(parents=True)
    (folder / 'result.json').write_text('{"status":"failed"}')
    state = m.audit_once('unused', output)
    assert state['phase'] == 'native_failure' and not state['audited_cells']


def test_one_completion_is_accounted_once_with_three_tau_aggregates(setup, monkeypatch):
    m, study, output = setup
    folder = study / 'runs/0'
    folder.mkdir(parents=True)
    (folder / 'result.json').write_text('{"status":"succeeded"}')
    called = []
    def fake_readout(args, **kw):
        assert kw['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert 'run-one' not in args and 'launch' not in args
        target = Path(args[args.index('--output-dir') + 1])
        cell = dict(native_status='succeeded', run_id='0', rows=[{}] * 480, files_sha256={'raw': 'hash'},
                    totals=dict(argmax_checked_tokens=480, new_tokens=480, pooled_tau=1.4,
                                prompt_mean_tau=1.5, category_mean_pooled_tau=1.45))
        (target / 'norm_pending.json').write_text(json.dumps({'complete': False, 'cells': {'template/0.5': cell}}))
        called.append(True)
    monkeypatch.setattr(m.subprocess, 'run', fake_readout)
    a = m.audit_once('unused', output)
    b = m.audit_once('unused', output)
    assert a == b and called == [True] and not a['complete']
    assert a['audited_cells'][0]['totals']['category_mean_pooled_tau'] == 1.45
    assert not list(output.rglob('norm_evidence_v1.json'))


def test_all_eight_wait_for_native_study_process(setup, monkeypatch):
    m, study, output = setup
    for i in range(8):
        folder = study / 'runs' / str(i)
        folder.mkdir(parents=True)
        (folder / 'result.json').write_text('{"status":"succeeded"}')
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('Final native process is pending'))
    state = m.audit_once('unused', output)
    assert state['phase'] == 'waiting_for_native_study_process' and not state['complete']
