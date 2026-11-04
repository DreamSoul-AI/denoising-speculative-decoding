"""Static plan integrity and fail-closed, independently recorded case results."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('sd_test_runner',ROOT / 'tests' / 'run.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_manifest_covers_all_declared_sources():
    body, plan=runner.read_plan(ROOT/'tests/manifest.json','cpu-pr')
    assert body['tests'] and all(row['reason'] for row in plan['exclude'])
    with pytest.raises(ValueError): runner.read_plan(ROOT/'tests/manifest.json','missing')


def test_missing_outcomes_block_completion(tmp_path):
    body,plan=runner.read_plan(ROOT/'tests/manifest.json','cpu-pr')
    record=runner.Recorder(body,plan,tmp_path,{'run_id':'fake'})
    record.selected={'not-executed'}
    session=SimpleNamespace(exitstatus=0)
    record.pytest_sessionfinish(session,0)
    assert session.exitstatus==1
    assert json.loads((tmp_path/'summary.json').read_text())['missing']==['not-executed']


def test_phase_evidence_failure_propagates(tmp_path):
    body,plan=runner.read_plan(ROOT/'tests/manifest.json','cpu-pr')
    record=runner.Recorder(body,plan,tmp_path/'missing-parent',{'run_id':'fake'})
    report=SimpleNamespace(nodeid=body['tests'][0]['test_id'],when='call',failed=False,skipped=False,
        outcome='passed',duration=0.)
    with pytest.raises(OSError): record.pytest_runtest_logreport(report)


@pytest.mark.parametrize('outcome', ['skipped', 'passed'])
def test_expected_phase_identity_is_preserved_without_weakening_gate(tmp_path, outcome):
    body, plan = runner.read_plan(ROOT/'tests/manifest.json', 'cpu-pr')
    record = runner.Recorder(body, plan, tmp_path, {'run_id':'expected'})
    report = SimpleNamespace(nodeid=body['tests'][0]['test_id'], when='call',
        failed=False, skipped=outcome=='skipped', passed=outcome=='passed',
        outcome=outcome, duration=0., wasxfail='expected control', longrepr='expected control')
    record.pytest_runtest_logreport(report)
    saved = json.loads(next(tmp_path.glob('*.call.json')).read_text())
    assert saved['wasxfail'] == 'expected control'
    assert saved['native_status'] == ('xfail' if outcome=='skipped' else 'xpass')
    assert saved['verdict'] == ('failed' if outcome=='skipped' else 'passed')


def test_explicit_plan_selects_only_required_cases_and_rejects_missing_collection(tmp_path):
    body, _ = runner.read_plan(ROOT/'tests/manifest.json', 'cpu-pr')
    required, other = (row['test_id'] for row in body['tests'][:2])
    plan = {'include': [required], 'exclude': [], 'budget_seconds': 30}
    record = runner.Recorder(body, plan, tmp_path, {'run_id': 'selection'})
    dropped = []
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: dropped.extend(items)))
    items = [SimpleNamespace(nodeid=required+'[case]'), SimpleNamespace(nodeid=other)]
    record.pytest_collection_modifyitems(config, items)
    assert record.selected == {required+'[case]'} and [item.nodeid for item in dropped] == [other]
    with pytest.raises(pytest.UsageError, match='missing'):
        record.pytest_collection_modifyitems(config, [SimpleNamespace(nodeid=other)])


def test_phase_evidence_records_actual_plan_and_device_metrics(tmp_path):
    body, plan = runner.read_plan(ROOT/'tests/manifest.json', 'cpu-pr')
    record = runner.Recorder(body, plan, tmp_path, {'run_id': 'gpu', 'plan_id': 'cuda-validation'})
    report = SimpleNamespace(nodeid=body['tests'][0]['test_id'], when='call', failed=False,
        skipped=False, passed=True, outcome='passed', duration=0., user_properties=[('cuda_peak_allocated_bytes', 4096)])
    record.pytest_runtest_logreport(report)
    saved = json.loads(next(tmp_path.glob('*.call.json')).read_text())
    assert saved['plan_id'] == 'cuda-validation' and saved['metrics']['cuda_peak_allocated_bytes'] == 4096


@pytest.mark.parametrize('fault', ['unknown', 'duplicate', 'excluded', 'cpu_type'])
def test_manifest_rejects_invalid_plan_selection(tmp_path, fault):
    body, _ = runner.read_plan(ROOT/'tests/manifest.json', 'cpu-pr')
    selected = body['tests'][0]['test_id']
    plan = {'include': [selected], 'exclude': [], 'cpu_only': False, 'budget_seconds': 30}
    if fault == 'unknown':
        plan['include'] = ['missing-test']
    elif fault == 'duplicate':
        plan['include'] *= 2
    elif fault == 'excluded':
        plan['exclude'] = [{'test_id': selected, 'reason': 'conflict'}]
    else:
        plan['cpu_only'] = 'false'
    body['plans']['invalid'] = plan
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(body), encoding='utf-8')
    with pytest.raises(ValueError):
        runner.read_plan(path, 'invalid')


def test_cuda_plan_records_unavailable_device_without_cpu_fallback(tmp_path, monkeypatch):
    import torch

    body, _ = runner.read_plan(ROOT/'tests/manifest.json', 'cpu-pr')
    plan = {'include': [body['tests'][0]['test_id']], 'exclude': [], 'cpu_only': False, 'budget_seconds': 30}
    monkeypatch.setattr(runner, 'read_plan', lambda *args: (body, plan))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: '')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'inherited-mask')
    output = tmp_path / 'unavailable'
    assert runner.main(['--plan', 'cuda-validation', '--output-root', str(output)]) == 1
    saved = json.loads((output/'summary.json').read_text())
    assert saved['environment']['cpu_only'] is False and saved['environment']['cuda_available'] is False
    assert saved['environment']['cuda_visible_devices'] == 'inherited-mask'
    assert 'no CPU fallback' in saved['reason']
