"""Run a declared offline validation plan and save per-case evidence outside the source tree.

Usage: python tests/run.py --plan cpu-pr --output-root /external/new-validation-root
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
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


def _needs_measurement(metadata):
    return metadata['result_type'] == 'numeric' or metadata['result_detail'] in ('metrics', 'samples')


def _metric_specs(metadata):
    specs = metadata.get('metric_specs')
    if (not isinstance(specs, list) or not specs
            or any(not isinstance(spec, dict) or any(not isinstance(spec.get(field), str) or not spec[field].strip()
                   for field in ('name', 'unit', 'statistic')) for spec in specs)):
        raise ValueError('measurement requires explicit metric_specs names, units and statistics')
    if len({spec['name'] for spec in specs}) != len(specs):
        raise ValueError('metric_specs names must be unique')
    return {spec['name']: spec for spec in specs}


def _measurement_error(metadata, properties):
    specs = _metric_specs(metadata)
    metrics = properties.get('metrics')
    if not isinstance(metrics, list) or not metrics:
        return 'measurement requires a nonempty metrics list'
    names = set()
    for metric in metrics:
        if not isinstance(metric, dict) or not isinstance(metric.get('name'), str):
            return 'metric must be an object with a declared name'
        name = metric['name']
        if name not in specs or name in names:
            return 'metric names must match declarations without duplicates'
        names.add(name)
        if any(metric.get(field) != specs[name][field] for field in ('unit', 'statistic')):
            return 'metric unit or statistic differs from its declaration'
        value = metric.get('value')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return 'metric value must be a finite real number, excluding booleans'
        count, warmup = metric.get('samples'), metric.get('warmup_iterations')
        if type(count) is not int or count < 1 or type(warmup) is not int or warmup < 0:
            return 'metric sample and warmup counts are invalid'
        raw = metric.get('raw_samples')
        if (not isinstance(raw, list) or len(raw) != count or any(isinstance(sample, bool)
                or not isinstance(sample, (int, float)) or not math.isfinite(sample) for sample in raw)):
            return 'metric raw_samples must contain exactly the declared finite observations'
        context = metric.get('context')
        if (not isinstance(context, dict) or not isinstance(context.get('data_size'), dict)
                or not context['data_size'] or any(type(size) is not int or size < 1 for size in context['data_size'].values())
                or not isinstance(context.get('environment'), dict) or not context['environment']
                or not isinstance(context.get('measurement_scope'), str) or not context['measurement_scope'].strip()):
            return 'metric requires data size, actual environment and measurement scope'
    return None if names == specs.keys() else 'measurement is missing declared metrics'


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
        if metadata['type'] == 'location' and metadata['level'] != 'unit':
            raise ValueError('location checks are only valid at unit level')
        if not isinstance(metadata.get('target'), str) or not metadata['target'].strip():
            raise ValueError('test target must be a nonempty string')
        if ((metadata['result_type'] == 'numeric' or metadata['type'] == 'physical')
                and metadata['result_detail'] == 'summary'):
            raise ValueError('numeric or physical results require metrics or samples detail')
        if _needs_measurement(metadata):
            _metric_specs(metadata)
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


def _identity(environment):
    return {key: environment.get(key) for key in ('run_id', 'plan_id', 'head')}


def _evidence_paths(root):
    paths = sorted({*(root/name for name in ('manifest.json', 'environment.json', 'selection.json', 'summary.pending.json')
                      if (root/name).is_file()), *root.glob('*.result.json'),
                   *(path for phase in ('setup', 'call', 'teardown') for path in root.glob('*.' + phase + '.json'))})
    if any(path.resolve().parent != root.resolve() or path.resolve() != path.absolute() for path in paths):
        raise ValueError('report evidence escapes the run root or uses a leaf alias')
    return paths


def render_report(root):
    """Rebuild a report from this run's saved evidence, without reading current declarations."""
    root = Path(root).resolve()
    def read(name, default=None):
        path = root/name
        if not path.is_file():
            return default
        if path.resolve().parent != root or path.resolve() != path:
            raise ValueError('report evidence escapes the run root or uses a leaf alias')
        return json.loads(path.read_text(encoding='utf-8'))
    def text(value):
        if value is None:
            return '未知'
        return html.escape(str(value), quote=False).replace('|', '&#124;').replace('`', '&#96;').replace(
            '[', '&#91;').replace(']', '&#93;').replace('\n', ' ').replace('\r', ' ')
    environment, manifest = read('environment.json', {}), read('manifest.json', {})
    final = read('summary.json')
    pending = read('summary.pending.json', {})
    summary = final if final is not None else pending
    if final is not None and (not pending or any(final.get(key) != value for key, value in pending.items()
            if key not in ('exit_code', 'error_type')) or (final.get('exit_code') != pending.get('exit_code')
            and final.get('exit_code') != 1) or final.get('exit_code') == 0 and pending.get('exit_code') != 0):
        raise ValueError('final summary differs from its saved candidate')
    for source in (manifest.get('environment', {}), summary.get('environment', {})):
        if _identity(source) != _identity(environment):
            raise ValueError('saved report run/plan/HEAD identity differs')
    if final and 'evidence_sha256' in final:
        actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _evidence_paths(root)}
        if actual != final['evidence_sha256']:
            raise ValueError('saved report evidence digest differs')
        report = root/'report.md'
        if report.exists() and (report.resolve().parent != root or report.resolve() != report):
            raise ValueError('saved Markdown escapes the run root or uses a leaf alias')
        if report.is_file() and hashlib.sha256(report.read_bytes()).hexdigest() != final.get('report_sha256'):
            raise ValueError('saved Markdown report digest differs')
    entries = {row['test_id']: row for row in manifest.get('tests', [])}
    selection = read('selection.json')
    selected = set(selection.get('selected', [])) if selection else set()
    if selection and _identity(selection.get('environment', {})) != _identity(environment):
        raise ValueError('saved selection identity differs')
    cases, metrics, durations = {}, [], {}
    for path in root.glob('*.result.json'):
        result = read(path.name); node = result['test_id']; metadata = entries.get(node.partition('[')[0])
        if node in cases or node not in selected or metadata is None or _identity(result['environment']) != _identity(environment):
            raise ValueError('saved terminal identity differs')
        key = hashlib.sha256(node.encode()).hexdigest()
        phases = [read(key + '.' + phase + '.json') for phase in ('setup', 'call', 'teardown')]
        if any(phase is None or phase['test_id'] != node or _identity(phase['environment']) != _identity(environment)
               for phase in phases):
            raise ValueError('saved terminal lacks matching phase identities')
        call = phases[1]
        complete = not result['evidence_errors'] and all(phase['status'] == 'passed' for phase in phases)
        numeric = metadata['result_type'] == 'numeric'
        expected_status = 'measured' if complete and numeric else 'passed' if complete else 'failed' if call['status'] == 'failed' else 'error'
        if (result['result_type'] != metadata['result_type'] or result['status'] != expected_status
                or result['quality_verdict'] != (None if numeric else 'passed' if complete else 'failed')
                or result['metrics'] != call['metrics']):
            raise ValueError('saved terminal semantics differ from declarations or phases')
        if complete and _needs_measurement(metadata) and _measurement_error(metadata, call['metrics']):
            raise ValueError('saved terminal contains invalid measurement')
        cases[node] = result
        values = [phase.get('duration') for phase in phases]
        durations[node] = sum(values) if all(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 for value in values) else None
        observations = call['metrics'].get('metrics', []) if _needs_measurement(metadata) else []
        for metric in observations if isinstance(observations, list) else []:
            if isinstance(metric, dict):
                metrics.append('| ' + ' | '.join(text(value) for value in (node, metric.get('name'), metric.get('value'),
                    metric.get('unit'), metric.get('statistic'), metric.get('samples'))) + ' |')
    measured = sum(case['status'] in ('passed', 'measured') and _needs_measurement(entries[node.partition('[')[0]])
                   for node, case in cases.items())
    quality = sum(case['quality_verdict'] == 'passed' for case in cases.values())
    if (any(field in summary and (field != 'selected' or selection is not None) and summary[field] != value for field, value in
            (('selected', len(selected)), ('measurement_completed', measured), ('quality_passed', quality)))
            or summary.get('exit_code') == 0 and (not selected or not selection or set(cases) != selected
                or any(case['status'] not in ('passed', 'measured') for case in cases.values()))):
        raise ValueError('saved batch counts or success disagree with case evidence')
    lines = ['# 测试批次报告', '', '本报告只汇总已保存证据；正式完成状态以 `summary.json` 为准；候选汇总不是正式终态。', '',
        '| 运行与范围 | 实际记录 |', '| --- | --- |']
    for field in ('run_id', 'plan_id', 'head', 'dirty', 'python', 'platform', 'torch', 'cuda_runtime', 'budget_seconds'):
        lines.append('| ' + field + ' | ' + text(environment.get(field)) + ' |')
    lines += ['', '| 执行与判定 | 数量/状态 |', '| --- | --- |',
        '| 退出码 | ' + text(summary.get('exit_code')) + ' |',
        '| 已选独立参数 | ' + (str(len(selected)) if selection else '未知（未保存selection）') + ' |',
        '| 已保存逐用例终态/尝试 | ' + str(len(cases)) + ' / ' + str(len(cases)) + '（每用例attempt=1） |',
        '| 测量完成 | ' + str(measured) + ' |', '| 质量通过 | ' + str(quality) + ' |',
        '| 批次墙钟秒 | ' + text(summary.get('duration')) + ' |', '',
        'Recorder计量开始/结束UTC：' + text(summary.get('started_at')) + ' / ' + text(summary.get('ended_at')), '',
        '测量完成与质量通过独立，可重叠；numeric 不计质量通过。缺收集时参数总数未知，未执行不算通过。', '',
        '| 声明维度 | 分组 | 已选参数 |', '| --- | --- | --- |']
    from collections import Counter
    for field in ('level', 'type', 'priority', 'cost_class', 'target'):
        counts = Counter(entries[node.partition('[')[0]][field] for node in selected if node.partition('[')[0] in entries)
        lines += ['| ' + field + ' | ' + text(group) + ' | ' + str(count) + ' |' for group, count in sorted(counts.items())]
    resources = environment.get('requested_resources', {})
    lines += ['', '成本边界：Recorder计时自其构造至生成批次候选汇总，含pytest准备、采集与清理；不含Python冷启动、此前Git/CUDA环境探测及报告生成/发布，不是GPU kernel时间。设备实际时长/峰值及费用未统一采集，保持未知，不填零。',
        '设备时长估计（秒）：' + text(resources.get('estimated_device_seconds')) + '；依据：' + text(resources.get('estimate_basis')),
        '请求GPU数：' + text(resources.get('gpu_count')) + '；显存要求：' + text(resources.get('memory_requirement')),
        '', '| 成本类 | 完整用例阶段秒合计 |', '| --- | --- |']
    for cost in ('c1', 'c2', 'c3', 'c4'):
        values = [value for node, value in durations.items() if entries[node.partition('[')[0]]['cost_class'] == cost]
        lines.append('| ' + cost + ' | ' + text(sum(values) if values and all(value is not None for value in values) else None) + ' |')
    lines += ['', '逐用例阶段秒为setup/call/teardown实际duration之和，不重复计入整批墙钟，也不估算未知项。',
        '', '| 用例/参数 | 终态 | 错误类别 | 证据 |', '| --- | --- | --- | --- |']
    for node in sorted(selected):
        case = cases.get(node); key = hashlib.sha256(node.encode()).hexdigest()
        category = ('缺终态' if case is None else '证据无效' if case['evidence_errors'] else
                    '无' if case['status'] in ('passed', 'measured') else case['status'])
        links = ['[' + phase + '](' + key + '.' + phase + '.json)' for phase in ('setup', 'call', 'teardown')
                 if (root/(key + '.' + phase + '.json')).is_file()]
        if case:
            links.append('[终态](' + key + '.result.json)')
        lines.append('| ' + text(node) + ' | ' + text(case['status'] if case else None) + ' | ' + text(category) + ' | ' + ' '.join(links) + ' |')
    lines += ['', '## 数值采集', '', '| 参数 | 指标 | 数值 | 单位 | 统计口径 | 样本数 |', '| --- | --- | --- | --- | --- | --- |', *metrics,
        '', '单位/样本条件、inline原样本与阈值以对应call证据为准；本报告不推断性能结论。', '', '## 未验证范围', '']
    if not selection:
        lines.append('收集清单未知；不得从已保存子集推定全部完成。')
    else:
        lines += ['- 缺收集声明：' + text(node) for node in selection.get('missing_required', [])]
        lines += ['- 未选：' + text(row['test_id']) + '；原因：' + text(row['reason']) for row in selection.get('unselected', [])]
        lines += ['- 缺终态参数：' + text(node) for node in sorted(selected - cases.keys())]
    if summary.get('error_type'):
        lines.append('批次错误类别：' + text(summary['error_type']))
    evidence_links = [('环境', 'environment.json'), ('本次声明与计划', 'manifest.json'), ('选择', 'selection.json')]
    lines += ['', '原始证据：' + '、'.join('[' + label + '](' + name + ')' for label, name in evidence_links if (root/name).is_file()) + '。', '']
    return '\n'.join(lines)


class Recorder:
    def __init__(self, body, plan, output, environment):
        self.entries = {row['test_id']: {**body['defaults'], **row} for row in body['tests']}
        self.plan, self.output, self.environment = plan, Path(output), environment
        self.selected, self.finished = set(), set()
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        self.failed = False
        self.phases: dict[str, dict[str, dict[str, Any]]] = {}
        self.evidence_errors: dict[str, list[str]] = {}

    def save(self, name, value):
        path = self.output / name
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)

    def snapshot(self):
        for name, value in (('environment.json', self.environment), ('manifest.json', {
                'environment': _identity(self.environment), 'plan': self.plan, 'tests': list(self.entries.values())})):
            path = self.output/name
            if path.exists():
                if path.resolve() != path.absolute():
                    raise ValueError('saved snapshot uses a leaf alias')
                if json.loads(path.read_text(encoding='utf-8')) != value:
                    raise ValueError('saved run snapshot differs: ' + name)
            else:
                self.save(name, value)

    def publish_summary(self, summary):
        self.snapshot()
        self.save('summary.pending.json', summary)
        try:
            evidence = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _evidence_paths(self.output)}
            rendered = render_report(self.output)
            if evidence != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _evidence_paths(self.output)}:
                raise ValueError('saved report inputs changed while rendering')
            with (self.output/'report.md').open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(rendered)
            if evidence != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _evidence_paths(self.output)}:
                raise ValueError('saved report inputs changed while writing Markdown')
            summary = {**summary, 'report_sha256': hashlib.sha256((self.output/'report.md').read_bytes()).hexdigest(),
                       'evidence_sha256': evidence}
        except Exception as error:
            summary = {**summary, 'exit_code': 1, 'error_type': type(error).__name__}
        self.save('summary.json', summary)
        return int(summary['exit_code'])

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
        self.selected = {item.nodeid for item in kept}
        reasons = {row['test_id']: row['reason'] for row in self.plan['exclude']}
        self.save('selection.json', {'environment': _identity(self.environment), 'required': sorted(required),
            'selected': sorted(self.selected), 'missing_required': sorted(missing), 'excluded': self.plan['exclude'],
            'unselected': [{'test_id': node, 'reason': reasons.get(node, 'not included in this explicit plan')}
                           for node in sorted(self.entries.keys() - required | set(reasons))],
            'excluded_parameter_ids': [item.nodeid for item in dropped]})
        if missing or not kept:
            self.failed = True
            raise pytest.UsageError('required collection missing: ' + repr(sorted(missing)))
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
        errors = self.evidence_errors.setdefault(report.nodeid, [])
        phases = self.phases.setdefault(report.nodeid, {})
        if report.when not in ('setup', 'call', 'teardown') or report.when in phases:
            errors.append('unknown or repeated test phase: ' + report.when)
            self.failed = True
            return
        if ((report.when == 'setup' and phases) or (report.when != 'setup' and 'setup' not in phases)
                or (report.when == 'call' and 'teardown' in phases)):
            errors.append('test phases arrived out of order')
        properties = {}
        for name, value in getattr(report, 'user_properties', []):
            if not isinstance(name, str) or not name.strip() or name in properties:
                errors.append('invalid or duplicate result property: ' + str(name))
            else:
                properties[name] = value
        if _needs_measurement(metadata) and report.when == 'call':
            error = _measurement_error(metadata, properties)
            if error:
                errors.append(error)
        try:
            properties = json.loads(json.dumps(properties, allow_nan=False))
        except (TypeError, ValueError):
            errors.append('result properties are not finite JSON data')
            properties = {}  # Invalid data cannot be serialized as a successful measurement.
        self.failed |= verdict == 'failed' or bool(errors)
        if errors:
            verdict = 'failed'
        key = hashlib.sha256(report.nodeid.encode()).hexdigest()
        event = {
            'run_id': self.environment['run_id'], 'plan_id': self.environment.get('plan_id', 'cpu-pr'),
            'attempt': 1, **metadata, 'test_id': report.nodeid, 'environment': self.environment,
            'phase': report.when, 'status': report.outcome,
            'verdict': None if metadata['result_type'] == 'numeric' else verdict,
            'native_status': native_status,
            **({'wasxfail': report.wasxfail} if hasattr(report, 'wasxfail') else {}),
            'duration': report.duration, 'metrics': properties, 'cost': {}, 'evidence_errors': list(errors),
            'failure': str(report.longrepr) if report.failed or report.skipped else None, 'artifacts': []}
        self.save(key + '.' + report.when + '.json', event)
        phases[report.when] = event
        if report.when == 'teardown':
            self.finished.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        missing = {node for node in self.selected if set(self.phases.get(node, {})) != {'setup', 'call', 'teardown'}}
        measured = quality_passed = 0
        for node in sorted(self.selected - missing):
            phases = self.phases[node]
            metadata = self.entries[node.partition('[')[0]]
            complete = not self.evidence_errors[node] and all(event['status'] == 'passed' for event in phases.values())
            numeric = metadata['result_type'] == 'numeric'
            status = 'measured' if complete and numeric else 'passed' if complete else 'error'
            if not complete and phases['call']['status'] == 'failed':
                status = 'failed'
            verdict = None if numeric else 'passed' if complete else 'failed'
            measured += int(complete and _needs_measurement(metadata))
            quality_passed += int(complete and not numeric)
            self.save(hashlib.sha256(node.encode()).hexdigest() + '.result.json', {
                'test_id': node, 'attempt': 1, 'result_type': metadata['result_type'],
                'status': status, 'verdict': verdict, 'quality_verdict': verdict,
                'metrics': phases['call']['metrics'], 'evidence_errors': self.evidence_errors[node],
                'environment': self.environment})
        if missing or not self.selected or self.failed:
            session.exitstatus = 1
        session.exitstatus = self.publish_summary({'exit_code': int(session.exitstatus), 'selected': len(self.selected),
            'finished': len(self.finished), 'missing': sorted(missing), 'failed_or_skipped': self.failed,
            'measurement_completed': measured, 'quality_passed': quality_passed,
            'evidence_errors': {node: errors for node, errors in self.evidence_errors.items() if errors},
            'started_at': self.started_at, 'ended_at': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            'environment': self.environment, 'duration': time.monotonic() - self.started})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='cpu-pr')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    if output.is_relative_to(ROOT):
        parser.error('validation output must be outside the source tree')
    output.mkdir(parents=True, exist_ok=False)
    try:
        body, plan = read_plan(ROOT / 'tests/manifest.json', args.plan)
    except Exception as error:
        environment = {'run_id': output.name, 'plan_id': args.plan, 'python': sys.version, 'platform': platform.platform()}
        recorder = Recorder({'defaults': {}, 'tests': []}, {}, output, environment)
        return recorder.publish_summary({'exit_code': 1, 'error_type': type(error).__name__, 'environment': environment})
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
    recorder.snapshot()
    requested_gpus = plan.get('resources', {}).get('gpu_count', 1)
    if not cpu_only and (not environment['cuda_available'] or len(environment['cuda_devices']) < requested_gpus):
        reason = ('CUDA required but unavailable; no CPU fallback' if not environment['cuda_available'] else
                  'visible CUDA GPU count is below the declared requirement; no CPU fallback')
        return recorder.publish_summary({'exit_code': 1, 'reason': reason, 'error_type': 'CUDAAdmissionError',
                                         'environment': environment})
    sources = sorted({str(ROOT / node.partition('::')[0]) for node in plan['include']}) if 'include' in plan else [str(ROOT/'tests')]
    junit_options = [] if cpu_only else ['-o', 'junit_family=xunit1']
    return int(pytest.main([*sources, '--basetemp', str(output/'pytest'),
        '-o', 'cache_dir='+str(output/'cache'), '--junitxml='+str(output/'junit.xml'),
        *junit_options, '-ra'], plugins=[recorder]))


if __name__ == '__main__':
    raise SystemExit(main())
