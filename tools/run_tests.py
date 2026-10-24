"""Run the declared offline CPU plan and save per-case evidence outside the source tree.

Usage: python tools/run_tests.py --plan cpu-pr --output-root /external/new-validation-root
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_plan(path, name):
    body = json.loads(Path(path).read_text(encoding='utf-8'))
    if body['schema'] != 1 or name not in body['plans']:
        raise ValueError('unknown validation schema/plan')
    entries = body['tests']
    if not entries or len({row['test_id'] for row in entries}) != len(entries):
        raise ValueError('test declarations must be nonempty and unique')
    for row in entries:
        metadata = {**body['defaults'], **row}
        if any(not metadata.get(k) for k in ('level','type','cost_class','result_type','priority','target','result_detail')):
            raise ValueError('missing test metadata')
        if not (ROOT / row['test_id'].partition('::')[0]).is_file():
            raise ValueError('missing required test source: ' + row['test_id'])
    plan = body['plans'][name]
    excluded = {row['test_id']: row for row in plan['exclude']}
    if len(excluded) != len(plan['exclude']) or not set(excluded).issubset(row['test_id'] for row in entries):
        raise ValueError('unknown/duplicate exclusion')
    if any(not row.get('reason') for row in excluded.values()):
        raise ValueError('exclusions require reasons')
    return body, plan


class Recorder:
    def __init__(self, body, plan, output, environment):
        self.entries = {row['test_id']: {**body['defaults'], **row} for row in body['tests']}
        self.plan, self.output, self.environment = plan, Path(output), environment
        self.selected, self.finished = set(), set()
        self.started = time.monotonic()
        self.failed = False

    def save(self, name, value):
        path = self.output / name
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)

    def pytest_collection_modifyitems(self, config, items):
        import pytest
        excluded = {row['test_id'] for row in self.plan['exclude']}
        kept, dropped, found = [], [], set()
        for item in items:
            base = item.nodeid.partition('[')[0]
            if base not in self.entries:
                raise pytest.UsageError('undeclared collected test: ' + item.nodeid)
            found.add(base)
            (dropped if base in excluded else kept).append(item)
        missing = set(self.entries) - found
        if missing or not kept:
            raise pytest.UsageError('required collection missing: ' + repr(sorted(missing)))
        self.selected = {item.nodeid for item in kept}
        self.save('selection.json', {'selected': sorted(self.selected), 'excluded': self.plan['exclude'],
                                   'excluded_parameter_ids': [item.nodeid for item in dropped]})
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept

    def pytest_runtest_setup(self, item):
        import pytest
        if time.monotonic() - self.started > self.plan['budget_seconds']:
            pytest.exit('CPU plan budget exhausted; remaining cases are missing', returncode=1)

    def pytest_runtest_logreport(self, report):
        base = report.nodeid.partition('[')[0]
        metadata = self.entries[base]
        verdict = 'failed' if report.failed or report.skipped else 'passed'
        native_status = report.outcome
        if hasattr(report, 'wasxfail'):
            native_status = 'xfail' if report.skipped else 'xpass' if report.passed else report.outcome
        self.failed |= verdict == 'failed'
        key = hashlib.sha256(report.nodeid.encode()).hexdigest()
        self.save(key + '.' + report.when + '.json', {
            'run_id': self.environment['run_id'], 'plan_id': 'cpu-pr',
            'attempt': 1, **metadata, 'test_id': report.nodeid, 'environment': self.environment,
            'phase': report.when, 'status': report.outcome, 'verdict': verdict,
            'native_status': native_status,
            **({'wasxfail': report.wasxfail} if hasattr(report, 'wasxfail') else {}),
            'duration': report.duration, 'metrics': {}, 'cost': {},
            'failure': str(report.longrepr) if report.failed or report.skipped else None, 'artifacts': []})
        if report.when == 'teardown':
            self.finished.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        missing = self.selected - self.finished
        if missing or not self.selected or self.failed:
            session.exitstatus = 1
        self.save('summary.json', {'exit_code': int(session.exitstatus), 'selected': len(self.selected),
            'finished': len(self.finished), 'missing': sorted(missing), 'failed_or_skipped': self.failed,
            'environment': self.environment, 'duration': time.monotonic() - self.started})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='cpu-pr')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    if output.is_relative_to(ROOT):
        parser.error('validation output must be outside the source tree')
    body, plan = read_plan(ROOT / 'tests/manifest.json', args.plan)
    output.mkdir(parents=True, exist_ok=False)
    env = {'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'',
           'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUTF8':'1'}
    os.environ.update(env)
    os.environ['PYTHONPATH'] = os.pathsep.join((str(ROOT/'src'), str(ROOT)))
    sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
    def git(*params):
        return subprocess.check_output(['git','-c','safe.directory='+str(ROOT),*params],
            cwd=ROOT, text=True, timeout=20).strip()
    environment = {'run_id': output.name, 'python': sys.version, 'platform': platform.platform(),
        'head': git('rev-parse','HEAD'), 'dirty': bool(git('status','--porcelain')), 'cpu_only': True}
    import pytest
    recorder = Recorder(body, plan, output, environment)
    return int(pytest.main([str(ROOT/'tests'), '--basetemp', str(output/'pytest'),
        '-o', 'cache_dir='+str(output/'cache'), '--junitxml='+str(output/'junit.xml'), '-ra'], plugins=[recorder]))


if __name__ == '__main__':
    raise SystemExit(main())
