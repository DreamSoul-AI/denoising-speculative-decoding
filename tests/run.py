"""Run a declared offline validation plan and save per-case evidence outside the source tree.

Usage: python tests/run.py --plan cpu-pr --output-root /external/new-validation-root
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENUMS = {'level': {'unit', 'integration', 'e2e'}, 'type': {'location', 'content', 'physical'},
         'cost_class': {'c1', 'c2', 'c3', 'c4'}, 'result_type': {'categorical', 'numeric'},
         'priority': {'p1', 'p2', 'p3'}, 'result_detail': {'summary', 'metrics', 'samples'}}


def _resources(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + ' resources must be an object')
    if 'gpu_count' in value and (type(value['gpu_count']) is not int or value['gpu_count'] < 0):
        raise ValueError(label + ' gpu_count must be a nonnegative integer')
    seconds = value.get('estimated_device_seconds')
    if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
        raise ValueError(label + ' estimated_device_seconds must be finite and nonnegative')
    for field in ('network', 'model_download'):
        if field in value and type(value[field]) is not bool:
            raise ValueError(label + ' ' + field + ' must be a boolean')
        if value.get(field) is True:
            raise ValueError('offline validation cannot require ' + field)
    return value


def read_plan(path, name):
    body = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(body, dict) or type(body.get('schema')) is not int or body['schema'] != 1
            or not isinstance(body.get('plans'), dict) or name not in body['plans']):
        raise ValueError('unknown validation schema/plan')
    entries, defaults = body.get('tests'), body.get('defaults')
    if not isinstance(defaults, dict) or not isinstance(entries, list) or not entries:
        raise ValueError('test defaults and nonempty declarations are required')
    declared = {}
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError('test declarations must be objects')
        metadata = {**defaults, **row}
        for field, choices in ENUMS.items():
            if not isinstance(metadata.get(field), str) or metadata[field] not in choices:
                raise ValueError('invalid or missing test metadata: ' + field)
        if not isinstance(metadata.get('target'), str) or not metadata['target'].strip():
            raise ValueError('test target must be a nonempty string')
        if ((metadata['result_type'] == 'numeric' or metadata['type'] == 'physical')
                and metadata['result_detail'] == 'summary'):
            raise ValueError('numeric or physical results require metrics or samples detail')
        node = row.get('test_id')
        if not isinstance(node, str) or not node or '\\' in node or '[' in node or ']' in node:
            raise ValueError('test_id must name a canonical unparameterized pytest source node')
        parts = node.split('::')
        relative = Path(parts[0])
        source = (ROOT / relative).resolve()
        if (len(parts) < 2 or any(not part.isidentifier() for part in parts[1:])
                or relative.is_absolute() or '..' in relative.parts or relative.suffix != '.py'
                or not source.is_relative_to((ROOT / 'tests').resolve()) or not source.is_file()):
            raise ValueError('missing or invalid required test source: ' + node)
        if node in declared:
            raise ValueError('test declarations must be unique')
        tags = metadata.get('tags')
        if (not isinstance(tags, list) or not tags or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
                or len(tags) != len(set(tags)) or len({'offline_cpu', 'offline_cuda'} & set(tags)) != 1):
            raise ValueError('test tags must be nonempty, unique and resource-consistent')
        resources = _resources(metadata.get('resources', {}), node)
        cuda = 'offline_cuda' in tags or resources.get('gpu_count', 0) > 0
        if cuda and metadata['cost_class'] not in ('c3', 'c4'):
            raise ValueError('real CUDA tests require cost_class c3 or c4')
        if ('offline_cpu' in tags and resources.get('gpu_count', 0) > 0
                or 'offline_cuda' in tags and resources.get('gpu_count') == 0):
            raise ValueError('test tags contradict declared GPU resources')
        declared[node] = (metadata, cuda)
    plan = body['plans'][name]
    if not isinstance(plan, dict) or type(plan.get('cpu_only')) is not bool:
        raise ValueError('cpu_only must be an explicit boolean')
    budget = plan.get('budget_seconds')
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('plan budget_seconds must be finite and positive')
    included = plan.get('include')
    if included is not None:
        if (not isinstance(included, list) or not included or any(not isinstance(node, str) for node in included)
                or len(included) != len(set(included)) or not set(included) <= declared.keys()):
            raise ValueError('unknown/duplicate/empty inclusion')
    exclusions = plan.get('exclude')
    if not isinstance(exclusions, list):
        raise ValueError('plan exclusions must be an explicit list')
    excluded = set()
    for row in exclusions:
        if (not isinstance(row, dict) or not isinstance(row.get('test_id'), str)
                or row['test_id'] not in declared or row['test_id'] in excluded):
            raise ValueError('unknown/duplicate exclusion')
        if not isinstance(row.get('reason'), str) or not row['reason'].strip():
            raise ValueError('exclusions require nonempty string reasons')
        excluded.add(row['test_id'])
    if included is not None and set(included) & excluded:
        raise ValueError('included test must not also be excluded')
    selected = (set(included) if included is not None else set(declared)) - excluded
    if not selected:
        raise ValueError('validation plan selects no tests')
    resources = _resources(plan.get('resources', {}), 'plan')
    if plan['cpu_only']:
        if resources.get('gpu_count', 0) > 0 or any(declared[node][1] for node in selected):
            raise ValueError('CPU-only plan cannot select real CUDA requirements')
    else:
        seconds = resources.get('estimated_device_seconds')
        if (resources.get('gpu_count', 0) < 1 or type(seconds) not in (int, float) or seconds <= 0
                or any(not isinstance(resources.get(field), str) or not resources[field].strip()
                       for field in ('estimate_basis', 'memory_requirement'))):
            raise ValueError('CUDA plan requires GPU count, device-time estimate, basis and memory requirement')
        if any(declared[node][0].get('resources', {}).get('gpu_count', 0) > resources['gpu_count'] for node in selected):
            raise ValueError('CUDA plan has fewer GPUs than a selected test requires')
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
        required = set(self.plan.get('include', self.entries))
        kept: list[Any] = []
        dropped: list[Any] = []
        found = set()
        for item in items:
            base = item.nodeid.partition('[')[0]
            if base not in self.entries:
                raise pytest.UsageError('undeclared collected test: ' + item.nodeid)
            found.add(base)
            (dropped if base in excluded or base not in required else kept).append(item)
        missing = required - found
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
            pytest.exit('validation plan budget exhausted; remaining cases are missing', returncode=1)

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
            'run_id': self.environment['run_id'], 'plan_id': self.environment.get('plan_id', 'cpu-pr'),
            'attempt': 1, **metadata, 'test_id': report.nodeid, 'environment': self.environment,
            'phase': report.when, 'status': report.outcome, 'verdict': verdict,
            'native_status': native_status,
            **({'wasxfail': report.wasxfail} if hasattr(report, 'wasxfail') else {}),
            'duration': report.duration, 'metrics': dict(getattr(report, 'user_properties', [])), 'cost': {},
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
    cpu_only = plan.get('cpu_only', True)
    env = {'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
           'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUTF8':'1'}
    if cpu_only:
        env['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.update(env)
    os.environ['PYTHONPATH'] = os.pathsep.join((str(ROOT/'src'), str(ROOT)))
    sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
    def git(*params):
        return subprocess.check_output(['git','-c','safe.directory='+str(ROOT),*params],
            cwd=ROOT, text=True, timeout=20).strip()
    environment = {'run_id': output.name, 'python': sys.version, 'platform': platform.platform(),
        'head': git('rev-parse','HEAD'), 'dirty': bool(git('status','--porcelain')),
        'cpu_only': cpu_only, 'plan_id': args.plan,
        'budget_seconds': plan['budget_seconds'], 'requested_resources': plan.get('resources', {})}
    if not cpu_only:
        import torch
        environment.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_devices=[{'index': index, 'name': torch.cuda.get_device_name(index),
                'capability': list(torch.cuda.get_device_capability(index)),
                'total_memory': torch.cuda.get_device_properties(index).total_memory}
                for index in range(torch.cuda.device_count())])
    import pytest
    recorder = Recorder(body, plan, output, environment)
    recorder.save('environment.json', environment)
    requested_gpus = plan.get('resources', {}).get('gpu_count', 1)
    if not cpu_only and (not environment['cuda_available'] or len(environment['cuda_devices']) < requested_gpus):
        reason = ('CUDA required but unavailable; no CPU fallback' if not environment['cuda_available'] else
                  'visible CUDA GPU count is below the declared requirement; no CPU fallback')
        recorder.save('summary.json', {'exit_code': 1, 'reason': reason,
                                      'environment': environment})
        return 1
    sources = sorted({str(ROOT / node.partition('::')[0]) for node in plan['include']}) if 'include' in plan else [str(ROOT/'tests')]
    junit_options = [] if cpu_only else ['-o', 'junit_family=xunit1']
    return int(pytest.main([*sources, '--basetemp', str(output/'pytest'),
        '-o', 'cache_dir='+str(output/'cache'), '--junitxml='+str(output/'junit.xml'),
        *junit_options, '-ra'], plugins=[recorder]))


if __name__ == '__main__':
    raise SystemExit(main())
