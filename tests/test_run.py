"""Static plan integrity and fail-closed, independently recorded case results."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
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


def _manifest(tmp_path, *, cuda=False):
    node = 'tests/spec_mtp/test_extension_windows.py::test_no_shared_hit_uses_predeclared_rejection_control'
    body = {'schema': 1, 'defaults': {'level': 'unit', 'type': 'content', 'cost_class': 'c3' if cuda else 'c1',
        'result_type': 'categorical', 'priority': 'p1', 'result_detail': 'summary'},
        'tests': [{'test_id': node, 'target': 'tests.run', 'tags': ['offline_cuda' if cuda else 'offline_cpu']}],
        'plans': {'check': {'cpu_only': not cuda, 'budget_seconds': 30, 'include': [node], 'exclude': []}}}
    if cuda:
        body['plans']['check']['resources'] = {'gpu_count': 1, 'estimated_device_seconds': 10,
            'estimate_basis': 'controlled IO fixture', 'memory_requirement': 'fixture device metadata only',
            'network': False, 'model_download': False}
    path = tmp_path / 'manifest.json'
    def read():
        path.write_text(json.dumps(body), encoding='utf-8')
        return runner.read_plan(path, 'check')
    read()  # Every corruption starts from a separately admitted valid baseline.
    return body, read


@pytest.mark.parametrize('field', tuple(runner.ENUMS))
@pytest.mark.parametrize('fault', ['missing', 'invalid'])
def test_metadata_dimensions_reject_missing_or_unknown_values(tmp_path, field, fault):
    body, read = _manifest(tmp_path)
    if fault == 'missing':
        del body['defaults'][field]
    else:
        body['tests'][0][field] = 'unsupported'
    with pytest.raises(ValueError, match='metadata'):
        read()


@pytest.mark.parametrize('fault', ['blank_target', 'missing_id', 'parameter_id', 'absolute_source', 'unknown_tags',
                                 'both_device_tags', 'duplicate_tags', 'nonstring_tags', 'resource_disagreement'])
def test_metadata_source_and_device_requirements_are_explicit(tmp_path, fault):
    body, read = _manifest(tmp_path)
    row = body['tests'][0]
    if fault == 'blank_target': row['target'] = '  '
    elif fault == 'missing_id': del row['test_id']
    elif fault == 'parameter_id': row['test_id'] += '[parameter]'
    elif fault == 'absolute_source':
        source = tmp_path / 'outside.py'; source.write_text('def test_case(): pass')
        row['test_id'] = source.as_posix() + '::test_case'
    elif fault == 'unknown_tags': row['tags'] = ['structure_cleanup']
    elif fault == 'both_device_tags': row['tags'] = ['offline_cpu', 'offline_cuda']
    elif fault == 'duplicate_tags': row['tags'] *= 2
    elif fault == 'nonstring_tags': row['tags'] = [True]
    else: row.update(cost_class='c3', resources={'gpu_count': 1})
    with pytest.raises(ValueError): read()


@pytest.mark.parametrize('budget', [None, True, 0, -1, float('nan'), float('inf'), '30'])
def test_plan_budget_must_be_a_finite_positive_number(tmp_path, budget):
    body, read = _manifest(tmp_path)
    body['plans']['check']['budget_seconds'] = budget
    with pytest.raises(ValueError, match='budget_seconds'): read()


@pytest.mark.parametrize('fault', ['missing_cpu_mode', 'missing_exclusions', 'reason_whitespace', 'reason_boolean',
                                 'string_inclusion', 'empty_selection', 'invalid_exclusion_shape'])
def test_plan_selection_and_exclusion_reasons_are_structural_contracts(tmp_path, fault):
    body, read = _manifest(tmp_path)
    plan = body['plans']['check']; node = body['tests'][0]['test_id']
    if fault == 'missing_cpu_mode': del plan['cpu_only']
    elif fault == 'missing_exclusions': del plan['exclude']
    elif fault == 'string_inclusion': plan['include'] = node
    elif fault == 'invalid_exclusion_shape': plan['exclude'] = ['not an object']
    else:
        del plan['include']
        plan['exclude'] = [{'test_id': node, 'reason': True if fault == 'reason_boolean' else '   '
                           if fault == 'reason_whitespace' else 'explicitly unselected'}]
    with pytest.raises(ValueError): read()


@pytest.mark.parametrize('fault', ['gpu_downgraded_to_c1', 'cpu_selects_cuda', 'missing_resources', 'zero_gpu_count',
                                 'boolean_gpu_count', 'invalid_device_seconds', 'missing_basis', 'blank_memory',
                                 'network_required', 'test_exceeds_plan_gpu_count'])
def test_plan_resources_and_cuda_cost_cannot_be_bypassed(tmp_path, fault):
    body, read = _manifest(tmp_path, cuda=True)
    plan, row = body['plans']['check'], body['tests'][0]
    resources = plan['resources']
    if fault == 'gpu_downgraded_to_c1': row['cost_class'] = 'c1'
    elif fault == 'cpu_selects_cuda': plan.update(cpu_only=True, resources={})
    elif fault == 'missing_resources': del plan['resources']
    elif fault == 'zero_gpu_count': resources['gpu_count'] = 0
    elif fault == 'boolean_gpu_count': resources['gpu_count'] = True
    elif fault == 'invalid_device_seconds': resources['estimated_device_seconds'] = float('nan')
    elif fault == 'missing_basis': del resources['estimate_basis']
    elif fault == 'blank_memory': resources['memory_requirement'] = ' '
    elif fault == 'network_required': resources['network'] = True
    else: row['resources'] = {'gpu_count': 2}
    with pytest.raises(ValueError): read()


@pytest.mark.parametrize('kind,result_type', [('content', 'numeric'), ('physical', 'categorical')])
def test_measured_result_detail_preserves_categorical_semantics(tmp_path, kind, result_type):
    body, read = _manifest(tmp_path)
    row = body['tests'][0]; row.update(type=kind, result_type=result_type)
    with pytest.raises(ValueError, match='metrics or samples'): read()
    row['result_detail'] = 'metrics'
    accepted, _ = read()
    assert accepted['tests'][0]['result_type'] == result_type


def test_cuda_plan_rejects_insufficient_visible_devices_before_pytest(tmp_path, monkeypatch):
    import torch
    body, read = _manifest(tmp_path, cuda=True)
    body['plans']['check']['resources']['gpu_count'] = 2
    accepted, plan = read()
    monkeypatch.setattr(runner, 'read_plan', lambda *a: (accepted, plan))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'metadata fixture')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: (1, 0))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *a: SimpleNamespace(total_memory=1024))
    monkeypatch.setattr(pytest, 'main', lambda *a, **kw: pytest.fail('unmet resources reached pytest'))
    output = tmp_path / 'insufficient-devices'
    assert runner.main(['--plan', 'check', '--output-root', str(output)]) == 1
    result = json.loads((output / 'summary.json').read_text())
    assert 'GPU count' in result['reason'] and 'no CPU fallback' in result['reason']
    assert result['environment']['requested_resources']['gpu_count'] == 2
