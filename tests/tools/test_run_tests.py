"""Static plan integrity and fail-closed, independently recorded case results."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('sd_test_runner',ROOT/'tools/run_tests.py')
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
