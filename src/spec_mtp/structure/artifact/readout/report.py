"""Write Experiment and Run number tables from process.json. Does not write a conclusion."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote
from typing import Any

from spec_mtp.structure.artifact.errors import CorruptArtifactError
from spec_mtp.structure.artifact.asset.tree import diagnostic_files
from spec_mtp.structure.artifact.index import index_path, load_index
from spec_mtp.structure.artifact.paths import PROCESS_NAME, validate_output_root
from spec_mtp.structure.artifact.result.io import load_result
from spec_mtp.structure.artifact.readout.status import log_path


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return '-'
    return f'{float(value):.4f}'


def _factors(factors: dict[str, Any]) -> str:
    parts = [f'{key}={value}' for key, value in sorted((factors or {}).items())]
    return ', '.join(parts) or '-'


def _planned_runs(body: dict[str, Any]) -> dict[str, Any]:
    experiments = body.get('experiments')
    if not isinstance(experiments, list):
        raise ValueError('experiments must be a list')
    planned: dict[str, Any] = {}
    for exp in experiments:
        if not isinstance(exp, dict) or not isinstance(exp.get('runs'), list):
            raise ValueError('each experiment must contain a runs list')
        factors = exp.get('factors', {})
        if not isinstance(factors, dict):
            raise ValueError('experiment factors must be an object')
        for run in exp['runs']:
            if not isinstance(run, dict) or not isinstance(run.get('id'), str) or not run['id']:
                raise ValueError('each planned Run must have a nonempty string id')
            if run['id'] in planned:
                raise ValueError(f'duplicate planned Run: {run["id"]}')
            planned[run['id']] = {'seed': run.get('seed'), 'factors': factors}
    return planned


def _log_links(index: dict[str, Any], study_dir: Path, output: Path | None = None) -> dict[str, str]:
    links: dict[str, str] = {}
    for exp in index.get('experiments') or []:
        for run in exp.get('runs') or []:
            run_id = str(run.get('id') or '')
            if not run_id:
                continue
            path = log_path(study_dir, run, str(run.get('run_dir') or run_id))
            if path is not None:
                path = diagnostic_files(path)[-1]
                relative = path.resolve().relative_to(study_dir.resolve())
                if output is None:
                    links[run_id] = '../' + relative.as_posix()
                else:
                    try:
                        links[run_id] = quote(Path(os.path.relpath(path.resolve(), output.parent)).as_posix(), safe='/.:')
                    except ValueError:
                        links[run_id] = path.resolve().as_uri()
    return links


def render_numbers(study_dir: Path | str, *, output: Path | None = None) -> str:
    study_dir = Path(study_dir)
    try:
        body = load_result(study_dir / PROCESS_NAME)
    except CorruptArtifactError as exc:
        raise ValueError(f'Cannot report invalid process: {exc}') from exc
    try:
        index = load_index(study_dir)
        planned = _planned_runs(index)
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f'Cannot report without a usable current index: {index_path(study_dir)}; '
                         f'check the plan and rebuild with make: {exc}') from exc
    try:
        if ((index.get('study') is not None and body.get('study') != index['study'])
                or _planned_runs(body) != planned):
            raise ValueError('Study name or planned Run identity differs')
    except (TypeError, ValueError) as exc:
        raise ValueError(f'process does not match the current index: {exc}; '
                         'check the plan and rerun process before report') from exc
    links = _log_links(index, study_dir, output)
    lines = [
        f'# Numbers: {body.get("study") or study_dir.name}',
        '',
        'Derived from the saved process.json and index.json. Research interpretation belongs in the project records.',
        '',
        '## Experiments',
        '',
        '| factors | metric | mean | std | min | max | n |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for exp in body.get('experiments') or []:
        factors = _factors(dict(exp.get('factors') or {}))
        metrics = exp.get('metrics') or {}
        if not metrics:
            lines.append(f'| {factors} | - | - | - | - | - | {exp.get("n", 0)} |')
            continue
        for name in sorted(metrics):
            stat = metrics[name] if isinstance(metrics[name], dict) else {}
            lines.append(
                '| '
                + ' | '.join(
                    [
                        factors,
                        str(name),
                        _fmt(stat.get('mean')),
                        _fmt(stat.get('std')),
                        _fmt(stat.get('min')),
                        _fmt(stat.get('max')),
                        str(stat.get('n', exp.get('n', 0))),
                    ]
                )
                + ' |'
            )
    lines.extend(
        [
            '',
            '## Runs',
            '',
            '| factors | seed | id | metrics | log |',
            '|---|---:|---|---|---|',
        ]
    )
    for exp in body.get('experiments') or []:
        factors = _factors(dict(exp.get('factors') or {}))
        for run in exp.get('runs') or []:
            run_id = str(run.get('id') or '')
            metrics = run.get('metrics') or {}
            metric_text = ', '.join(f'{key}={_fmt(metrics[key])}' for key in sorted(metrics)) or '-'
            href = links[run_id]
            seed = '' if run.get('seed') is None else str(run.get('seed'))
            lines.append(
                f'| {factors} | {seed} | `{run_id}` | {metric_text} | [log]({href}) |'
            )
    lines.append('')
    return '\n'.join(lines)


def write_numbers(study_dir: Path | str, output: Path | str) -> Path:
    """Export to an explicit, new target separate from inputs and source roots."""
    study = Path(study_dir).resolve()
    requested = Path(output)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(requested)
    target = requested.resolve()
    validate_output_root(target.parent, study)
    if target.exists():
        raise FileExistsError(target)
    text = render_numbers(study, output=target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
    return target
