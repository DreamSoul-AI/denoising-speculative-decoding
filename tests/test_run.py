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
    _phase(record, body['tests'][0]['test_id'], 'setup')
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
    missing_root = tmp_path/'missing-attempt'; missing_root.mkdir()
    missing_record = runner.Recorder(body, plan, missing_root, {'run_id': 'missing-attempt'})
    with pytest.raises(pytest.UsageError, match='missing'):
        missing_record.pytest_collection_modifyitems(config, [SimpleNamespace(nodeid=other)])


def test_phase_evidence_records_actual_plan_and_device_metrics(tmp_path):
    body, plan = runner.read_plan(ROOT/'tests/manifest.json', 'cpu-pr')
    record = runner.Recorder(body, plan, tmp_path, {'run_id': 'gpu', 'plan_id': 'cuda-validation'})
    _phase(record, body['tests'][0]['test_id'], 'setup')
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
    path = tmp_path / 'declaration.json'
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
    row['metric_specs'] = [{'name': 'sum', 'unit': 'dimensionless', 'statistic': 'single'}]
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


def _metric():
    return {'name': 'sum', 'unit': 'dimensionless', 'statistic': 'single', 'value': 28.,
        'samples': 1, 'warmup_iterations': 0, 'raw_samples': [28.],
        'context': {'data_size': {'elements': 8}, 'environment': {'device': 'metadata fixture'},
                    'measurement_scope': 'one saved scalar; no tensor execution'}}


def _measurement_recorder(tmp_path, result_type='numeric'):
    body, read = _manifest(tmp_path)
    row = body['tests'][0]
    row.update(result_type=result_type, result_detail='metrics',
               metric_specs=[{'name': 'sum', 'unit': 'dimensionless', 'statistic': 'single'}])
    body, plan = read()
    record = runner.Recorder(body, plan, tmp_path, {'run_id': 'measurement-fixture'})
    node = row['test_id']
    record.pytest_collection_modifyitems(SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda **kw: None)),
                                         [SimpleNamespace(nodeid=node)])
    assert runner._measurement_error(row, {'metrics': [_metric()]}) is None
    return record, node


def _phase(record, node, when, *, properties=(), outcome='passed'):
    record.pytest_runtest_logreport(SimpleNamespace(nodeid=node, when=when,
        outcome=outcome, failed=outcome == 'failed', skipped=outcome == 'skipped', passed=outcome == 'passed',
        duration=0., user_properties=properties, longrepr='controlled phase failure'))


@pytest.mark.parametrize('fault', ['missing', 'empty', 'string', 'name_boolean', 'unit_blank', 'statistic_missing', 'duplicate'])
def test_measurement_declarations_require_named_units_and_statistics(tmp_path, fault):
    body, read = _manifest(tmp_path)
    row = body['tests'][0]
    row.update(result_type='numeric', result_detail='metrics',
               metric_specs=[{'name': 'sum', 'unit': 'dimensionless', 'statistic': 'single'}])
    read()
    if fault == 'missing': del row['metric_specs']
    elif fault == 'empty': row['metric_specs'] = []
    elif fault == 'string': row['metric_specs'] = 'sum'
    elif fault == 'name_boolean': row['metric_specs'][0]['name'] = False
    elif fault == 'unit_blank': row['metric_specs'][0]['unit'] = ' '
    elif fault == 'statistic_missing': del row['metric_specs'][0]['statistic']
    else: row['metric_specs'] *= 2
    with pytest.raises(ValueError, match='metric_specs'): read()


@pytest.mark.parametrize('level', ['unit', 'integration', 'e2e'])
def test_location_metadata_is_limited_to_unit_checks(tmp_path, level):
    body, read = _manifest(tmp_path)
    body['tests'][0].update(type='location', level=level)
    if level == 'unit':
        accepted, _ = read()
        assert accepted['tests'][0]['level'] == 'unit'
    else:
        with pytest.raises(ValueError, match='location.*unit'): read()


@pytest.mark.parametrize('fault', ['missing', 'empty', 'boolean_metrics', 'value_boolean', 'value_none', 'value_nan',
    'value_infinite', 'value_string', 'count_boolean', 'count_zero', 'raw_count', 'raw_boolean', 'raw_nan',
    'context_missing', 'environment_empty', 'scope_blank', 'unit', 'statistic', 'duplicate_metric', 'undeclared_metric'])
def test_invalid_measurements_cannot_complete_as_measured(tmp_path, fault):
    record, node = _measurement_recorder(tmp_path)
    metric = _metric(); properties = [('metrics', [metric])]
    if fault == 'missing': properties = []
    elif fault == 'empty': properties = [('metrics', [])]
    elif fault == 'boolean_metrics': properties = [('metrics', False)]
    elif fault == 'value_boolean': metric['value'] = False
    elif fault == 'value_none': metric['value'] = None
    elif fault == 'value_nan': metric['value'] = float('nan')
    elif fault == 'value_infinite': metric['value'] = float('inf')
    elif fault == 'value_string': metric['value'] = '28'
    elif fault == 'count_boolean': metric['samples'] = True
    elif fault == 'count_zero': metric['samples'] = 0
    elif fault == 'raw_count': metric['raw_samples'] = []
    elif fault == 'raw_boolean': metric['raw_samples'] = [False]
    elif fault == 'raw_nan': metric['raw_samples'] = [float('nan')]
    elif fault == 'context_missing': del metric['context']
    elif fault == 'environment_empty': metric['context']['environment'] = {}
    elif fault == 'scope_blank': metric['context']['measurement_scope'] = ' '
    elif fault == 'unit': metric['unit'] = 'ms'
    elif fault == 'statistic': metric['statistic'] = 'median'
    elif fault == 'duplicate_metric': properties[0][1].append(dict(metric))
    else: metric['name'] = 'not-declared'
    _phase(record, node, 'setup')
    _phase(record, node, 'call', properties=properties)
    _phase(record, node, 'teardown', properties=properties)
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    saved = json.loads(next(tmp_path.glob('*.result.json')).read_text())
    assert session.exitstatus == 1 and saved['status'] == 'error' and saved['verdict'] is None
    assert json.loads((tmp_path/'summary.json').read_text())['measurement_completed'] == 0


@pytest.mark.parametrize('result_type', ['numeric', 'categorical'])
def test_measurement_completion_and_quality_verdict_are_independent(tmp_path, result_type):
    record, node = _measurement_recorder(tmp_path, result_type)
    _phase(record, node, 'setup')
    _phase(record, node, 'call', properties=[('metrics', [_metric()])])
    assert not list(tmp_path.glob('*.result.json'))
    _phase(record, node, 'teardown')
    assert not list(tmp_path.glob('*.result.json'))
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    saved = json.loads(next(tmp_path.glob('*.result.json')).read_text())
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert session.exitstatus == 0
    assert saved['status'] == ('measured' if result_type == 'numeric' else 'passed')
    assert saved['quality_verdict'] == (None if result_type == 'numeric' else 'passed')
    assert summary['measurement_completed'] == 1
    assert summary['quality_passed'] == int(result_type == 'categorical')


@pytest.mark.parametrize('fault', ['setup_failed', 'call_failed', 'teardown_failed', 'missing_call', 'missing_teardown',
                                 'repeat_call', 'repeat_teardown', 'out_of_order'])
def test_measurement_requires_every_successful_phase_before_terminal_publication(tmp_path, fault):
    record, node = _measurement_recorder(tmp_path)
    for when in (('call', 'setup', 'teardown') if fault == 'out_of_order' else ('setup', 'call', 'teardown')):
        if fault == 'missing_' + when: continue
        _phase(record, node, when, properties=[('metrics', [_metric()])] if when == 'call' else [],
               outcome='failed' if fault == when + '_failed' else 'passed')
        if fault == 'repeat_' + when:
            _phase(record, node, when, properties=[('metrics', [_metric()])])
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert session.exitstatus == 1 and summary['measurement_completed'] == 0 and summary['quality_passed'] == 0
    assert all(json.loads(path.read_text())['status'] != 'measured' for path in tmp_path.glob('*.result.json'))


@pytest.mark.parametrize('name', ['metrics', 'cuda_device'])
@pytest.mark.parametrize('result_type', ['numeric', 'categorical'])
def test_duplicate_properties_are_rejected_without_overwriting_first_evidence(tmp_path, name, result_type):
    record, node = _measurement_recorder(tmp_path, result_type)
    first = [_metric()] if name == 'metrics' else 'first actual device'
    properties = [('metrics', [_metric()])] if name != 'metrics' else []
    properties += [(name, first), (name, 'replacement')]
    _phase(record, node, 'setup')
    _phase(record, node, 'call', properties=properties)
    _phase(record, node, 'teardown')
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    call = json.loads(next(tmp_path.glob('*.call.json')).read_text())
    assert call['metrics'][name] == first and session.exitstatus == 1
    assert call['status'] == 'passed' and call['native_status'] == 'passed'
    assert call['verdict'] == (None if result_type == 'numeric' else 'failed')
    assert any('duplicate' in message for message in call['evidence_errors'])


def test_categorical_failure_preserves_collected_measurements(tmp_path):
    record, node = _measurement_recorder(tmp_path, 'categorical')
    _phase(record, node, 'setup')
    _phase(record, node, 'call', properties=[('metrics', [_metric()])], outcome='failed')
    _phase(record, node, 'teardown')
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    result = json.loads(next(tmp_path.glob('*.result.json')).read_text())
    assert session.exitstatus == 1 and result['status'] == result['verdict'] == 'failed'
    assert result['metrics']['metrics'][0]['value'] == 28.


def test_measurement_call_evidence_is_frozen_before_teardown_mutation(tmp_path):
    record, node = _measurement_recorder(tmp_path)
    metric = _metric()
    _phase(record, node, 'setup')
    _phase(record, node, 'call', properties=[('metrics', [metric])])
    metric['value'] = False
    metric['context']['environment'].clear()
    _phase(record, node, 'teardown')
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    result = json.loads(next(tmp_path.glob('*.result.json')).read_text())
    assert session.exitstatus == 0 and result['status'] == 'measured'
    assert result['metrics']['metrics'][0]['value'] == 28.
    assert result['metrics']['metrics'][0]['context']['environment']


def test_measurement_evidence_write_failure_cannot_publish_success(tmp_path, monkeypatch):
    record, node = _measurement_recorder(tmp_path)
    _phase(record, node, 'setup')
    save = record.save
    def broken_save(name, value):
        if name.endswith('.call.json'):
            raise OSError('controlled evidence write failure')
        return save(name, value)
    monkeypatch.setattr(record, 'save', broken_save)
    with pytest.raises(OSError, match='evidence write failure'):
        _phase(record, node, 'call', properties=[('metrics', [_metric()])])
    _phase(record, node, 'teardown')
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1 and not list(tmp_path.glob('*.result.json'))
    assert json.loads((tmp_path/'summary.json').read_text())['measurement_completed'] == 0


def test_cuda_scalar_measurement_records_actual_samples_and_environment(record_property):
    import torch
    assert torch.cuda.is_available(), 'actual CUDA is required; no CPU fallback'
    device = torch.device('cuda:0')
    values = torch.arange(8, device=device, dtype=torch.float32)
    total = values.sum()
    torch.cuda.synchronize(device)
    observed = total.item()
    metric = _metric()
    metric.update(value=observed, raw_samples=[observed])
    metric['context'] = {'data_size': {'elements': values.numel()},
        'environment': {'device': str(total.device), 'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__},
        'measurement_scope': 'single FP32 CUDA sum of eight elements; no performance measurement'}
    record_property('metrics', [metric])
    record_property('cuda_device', torch.cuda.get_device_name(device))
    print('actual_measurement_device=', total.device, 'gpu=', torch.cuda.get_device_name(device))
    assert total.device.type == 'cuda' and observed == 28.


@pytest.mark.parametrize('result_type', ['numeric', 'categorical'])
def test_batch_report_rebuilds_saved_measurement_and_quality_counts(tmp_path, result_type):
    record, node = _measurement_recorder(tmp_path, result_type)
    record.environment.update(plan_id='saved-plan', head='saved-head', budget_seconds=30)
    # Collection identity belongs to the original environment used at collection.
    selection = json.loads((tmp_path/'selection.json').read_text())
    selection['environment'] = runner._identity(record.environment)
    (tmp_path/'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    for phase in ('setup', 'call', 'teardown'):
        _phase(record, node, phase, properties=[('metrics', [_metric()])] if phase == 'call' else [])
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    summary = json.loads((tmp_path/'summary.json').read_text())
    report = runner.render_report(tmp_path)
    assert report == (tmp_path/'report.md').read_text(encoding='utf-8')
    assert session.exitstatus == 0 and summary['report_sha256']
    assert '| 测量完成 | 1 |' in report
    assert '| 质量通过 | ' + str(int(result_type == 'categorical')) + ' |' in report
    assert 'saved-plan' in report and 'saved-head' in report and '设备实际时长/峰值及费用未统一采集，保持未知' in report
    assert '28.0' in report and '.call.json)' in report


@pytest.mark.parametrize('fault', ['head', 'numeric_as_passed', 'counter', 'phase_identity', 'promoted_exit_code'])
def test_batch_report_rejects_changed_saved_identity_and_semantics(tmp_path, fault):
    record, node = _measurement_recorder(tmp_path)
    for phase in ('setup', 'call', 'teardown'):
        _phase(record, node, phase, properties=[('metrics', [_metric()])] if phase == 'call' else [])
    code = 2 if fault == 'promoted_exit_code' else 0
    session = SimpleNamespace(exitstatus=code); record.pytest_sessionfinish(session, code)
    assert runner.render_report(tmp_path)
    if fault == 'head': path = tmp_path/'environment.json'
    elif fault in ('counter', 'promoted_exit_code'): path = tmp_path/'summary.json'
    elif fault == 'phase_identity': path = next(tmp_path.glob('*.call.json'))
    else: path = next(tmp_path.glob('*.result.json'))
    body = json.loads(path.read_text())
    if fault == 'head': body['head'] = 'changed-head'
    elif fault == 'counter': body['measurement_completed'] = 99
    elif fault == 'promoted_exit_code': body['exit_code'] = 0
    elif fault == 'phase_identity': body['environment']['run_id'] = 'another-run'
    else: body['status'] = 'passed'
    path.write_text(json.dumps(body), encoding='utf-8')
    with pytest.raises(ValueError): runner.render_report(tmp_path)


def test_batch_report_preserves_missing_collection_and_unselected_reasons(tmp_path):
    body, _ = runner.read_plan(ROOT/'tests/manifest.json', 'cpu-pr')
    required, excluded = (row['test_id'] for row in body['tests'][:2])
    plan = {'include': [required], 'exclude': [{'test_id': excluded, 'reason': 'explicit cost boundary'}], 'budget_seconds': 30}
    record = runner.Recorder(body, plan, tmp_path, {'run_id': 'failed-collection'})
    with pytest.raises(pytest.UsageError):
        record.pytest_collection_modifyitems(SimpleNamespace(hook=None), [])
    session = SimpleNamespace(exitstatus=4); record.pytest_sessionfinish(session, 4)
    report = (tmp_path/'report.md').read_text(encoding='utf-8')
    assert session.exitstatus != 0 and '缺收集声明' in report and required in report
    assert 'explicit cost boundary' in report and '未知' in report


@pytest.mark.parametrize('fault', ['report', 'summary'])
def test_batch_report_write_failure_never_publishes_success(tmp_path, monkeypatch, fault):
    record, node = _measurement_recorder(tmp_path)
    for phase in ('setup', 'call', 'teardown'):
        _phase(record, node, phase, properties=[('metrics', [_metric()])] if phase == 'call' else [])
    if fault == 'report':
        original = Path.open
        def fail_report(path, *args, **kwargs):
            if path.name == 'report.md': raise OSError('payload must not be displayed')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', fail_report)
    else:
        original = record.save
        def fail_summary(name, value):
            if name == 'summary.json': raise OSError('controlled final write failure')
            return original(name, value)
        monkeypatch.setattr(record, 'save', fail_summary)
    session = SimpleNamespace(exitstatus=0)
    if fault == 'summary':
        with pytest.raises(OSError): record.pytest_sessionfinish(session, 0)
        assert not (tmp_path/'summary.json').exists()
        assert '正式完成状态以' in (tmp_path/'report.md').read_text(encoding='utf-8')
    else:
        record.pytest_sessionfinish(session, 0)
        summary = json.loads((tmp_path/'summary.json').read_text())
        assert session.exitstatus == 1 and summary['error_type'] == 'OSError' and 'report_sha256' not in summary


def test_batch_report_escapes_saved_names_and_omits_failure_payload(tmp_path):
    record, node = _measurement_recorder(tmp_path, 'categorical')
    metric = _metric(); metric['context']['environment']['untrusted'] = '<script>secret</script>'
    record.entries[node]['metric_specs'][0]['unit'] = '<unit>|[fake](https://invalid)'
    metric['unit'] = record.entries[node]['metric_specs'][0]['unit']
    _phase(record, node, 'setup')
    _phase(record, node, 'call', properties=[('metrics', [metric])], outcome='failed')
    _phase(record, node, 'teardown')
    session = SimpleNamespace(exitstatus=1); record.pytest_sessionfinish(session, 1)
    report = (tmp_path/'report.md').read_text(encoding='utf-8')
    assert '<unit>' not in report and '&#124;' in report and '&#91;fake&#93;' in report
    assert '<script>' not in report and 'secret' not in report and 'controlled phase failure' not in report


def test_batch_report_records_source_admission_failure_without_current_manifest_fallback(tmp_path, monkeypatch):
    def invalid_source(*args): raise ValueError('private source payload')
    monkeypatch.setattr(runner, 'read_plan', invalid_source)
    output = tmp_path/'source-failure'
    assert runner.main(['--plan', 'unknown-plan', '--output-root', str(output)]) == 1
    report = runner.render_report(output)
    assert 'ValueError' in report and 'private source payload' not in report and '未保存selection' in report
    assert '[选择](selection.json)' not in report


@pytest.mark.parametrize('member', ['result', 'report'])
@pytest.mark.parametrize('placement', ['outside', 'same_root_alias'])
def test_batch_report_rejects_resolved_evidence_escape_before_hashing(tmp_path, monkeypatch, member, placement):
    record, node = _measurement_recorder(tmp_path)
    for phase in ('setup', 'call', 'teardown'):
        _phase(record, node, phase, properties=[('metrics', [_metric()])] if phase == 'call' else [])
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    victim = next(tmp_path.glob('*.result.json')) if member == 'result' else tmp_path/'report.md'
    resolve, read_bytes = Path.resolve, Path.read_bytes
    def redirected(path, *args, **kwargs):
        if path == victim:
            return (tmp_path.parent if placement == 'outside' else tmp_path)/'bound-other-file.json'
        return resolve(path, *args, **kwargs)
    def forbid_unchecked_read(path):
        assert path != victim, 'escaped evidence was hashed before containment'
        return read_bytes(path)
    monkeypatch.setattr(Path, 'resolve', redirected)
    monkeypatch.setattr(Path, 'read_bytes', forbid_unchecked_read)
    with pytest.raises(ValueError, match='escapes'): runner.render_report(tmp_path)


def test_batch_report_rechecks_inputs_after_markdown_write(tmp_path, monkeypatch):
    record, node = _measurement_recorder(tmp_path)
    for phase in ('setup', 'call', 'teardown'):
        _phase(record, node, phase, properties=[('metrics', [_metric()])] if phase == 'call' else [])
    original = Path.open
    def drifting_write(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        if path.name == 'report.md':
            environment = tmp_path/'environment.json'
            environment.write_text('{"run_id":"drifted"}', encoding='utf-8')
        return stream
    monkeypatch.setattr(Path, 'open', drifting_write)
    session = SimpleNamespace(exitstatus=0); record.pytest_sessionfinish(session, 0)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert session.exitstatus == 1 and summary['error_type'] == 'ValueError' and 'report_sha256' not in summary
