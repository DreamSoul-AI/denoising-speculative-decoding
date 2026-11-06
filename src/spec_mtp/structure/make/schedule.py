"""GPU round-robin and ``&`` / ``wait`` launch scripts."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from spec_mtp.structure.algorithm.resume import sibling_train_dependency
from spec_mtp.structure.artifact.asset import tree as diagnostic_assets
from spec_mtp.structure.artifact.config.io import load_config
from spec_mtp.structure.artifact.index import load_index
from spec_mtp.structure.artifact.layout import validate_run_directory, validate_study_root
from spec_mtp.structure.artifact.provenance import check_frozen, file_sha256
from spec_mtp.structure.make.expand import load_study_yaml
from spec_mtp.structure.artifact.result.format import STATUS_SUCCEEDED, decode_result
from spec_mtp.structure.control.mapping import get_setting

SCRIPTS_DIRNAME = 'scripts'
JOBS_NAME = 'jobs.json'


def repo_root_from(start: Path) -> Path:
    cur = start.resolve()
    for candidate in (cur, *cur.parents, *Path(__file__).resolve().parents):
        pyproject = candidate / 'pyproject.toml'
        if pyproject.is_file():
            return candidate
    return Path.cwd().resolve()


def gpu_ids(init_gpu: int, num_gpus: int) -> list[str]:
    if num_gpus <= 0:
        return []
    return [str(init_gpu + i) for i in range(num_gpus)]


def bash_path(path: Path) -> str:
    text = str(path.resolve())
    if len(text) >= 2 and text[1] == ':':
        return '/' + text[0].lower() + text[2:].replace('\\', '/')
    return text.replace('\\', '/')


def _run_result(study_dir: Path, run_id: str) -> dict[str, Any]:
    validate_run_directory(run_id)
    result_path = Path(study_dir) / 'runs' / run_id / 'result.json'
    try:
        return decode_result(result_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def run_succeeded(study_dir: Path, run_id: str) -> bool:
    if _run_result(study_dir, run_id).get('status') != STATUS_SUCCEEDED:
        return False
    parent = sibling_train_dependency(study_dir, run_id)
    if parent is None:
        return True
    if _run_result(study_dir, parent).get('status') != STATUS_SUCCEEDED:
        return False
    # ponytail: conservative local freshness; copied/backdated artifacts need explicit re-evaluation.
    try:
        runs = Path(study_dir) / 'runs'
        return (runs / parent / 'result.json').stat().st_mtime_ns <= (runs / run_id / 'result.json').stat().st_mtime_ns
    except OSError:
        return False


def _job_fields(config_path: Path) -> tuple[str, str]:
    cfg = load_config(config_path)
    algo = cfg.get('algorithm')
    system = cfg.get('system')
    mode = str(algo.get('mode') or 'train') if isinstance(algo, dict) else 'train'
    device = str(get_setting(system, 'device') or 'cpu') if isinstance(system, dict) else 'cpu'
    return mode, device.lower()


def job_mode(job: dict[str, Any]) -> str:
    return str(job.get('mode') or 'train')


def filter_jobs_by_mode(
    jobs: list[dict[str, Any]],
    modes: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    """Keep jobs whose ``mode`` is in ``modes``. Empty/None = no filter."""
    if not modes:
        return list(jobs)
    wanted = {str(mode).strip().lower() for mode in modes if str(mode).strip()}
    if not wanted:
        return list(jobs)
    return [job for job in jobs if job_mode(job) in wanted]


def filter_batches_by_mode(
    batches: list[list[dict[str, Any]]] | None,
    modes: list[str] | tuple[str, ...] | None,
) -> list[list[dict[str, Any]]] | None:
    if batches is None:
        return None
    if not modes:
        return [list(group) for group in batches if group]
    return [kept for kept in (filter_jobs_by_mode(list(group), modes) for group in batches) if kept]


def job_waves(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Train jobs first, then eval. Eval waits until the train wave finishes."""
    trains = [job for job in jobs if job.get('mode') != 'eval']
    evals = [job for job in jobs if job.get('mode') == 'eval']
    if trains and evals:
        return [trains, evals]
    return [jobs] if jobs else []


def _assign_gpus(jobs: list[dict[str, Any]], init_gpu: int, num_gpus: int) -> None:
    gpus = gpu_ids(init_gpu, num_gpus)
    gpu_index = 0
    for job in jobs:
        if str(job.get('device') or 'cuda').lower() == 'cpu' or not gpus:
            job.pop('gpu', None)
            continue
        job['gpu'] = gpus[gpu_index % len(gpus)]
        gpu_index += 1


def plan_jobs(
    study_dir: Path,
    config_paths: list[Path],
    *,
    init_gpu: int = 0,
    num_gpus: int = 1,
    include_done: bool = False,
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for path in config_paths:
        run_id = path.parent.name
        if not include_done and run_succeeded(study_dir, run_id):
            continue
        mode, device = _job_fields(path)
        pending.append(
            {
                'run_id': run_id,
                'config': str(path.as_posix()),
                'mode': mode,
                'device': device,
            }
        )
    ordered: list[dict[str, Any]] = []
    for wave in job_waves(pending):
        ordered.extend(wave)
    _assign_gpus(ordered, init_gpu, num_gpus)
    return ordered


def scripts_dir(study_dir: Path) -> Path:
    path = study_dir / SCRIPTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_launch_jobs(study_dir: Path, jobs: list[dict[str, Any]]) -> None:
    """Bind every persisted dispatch to the current index and canonical config."""
    study = Path(study_dir).resolve()
    index = load_index(study)
    known = {str(run.get('run_dir') or run.get('id')): run
             for exp in index.get('experiments') or [] for run in exp.get('runs') or []}
    seen = set()
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get('run_id'), str):
            raise ValueError('launch job requires a Run identity')
        run_id = job['run_id']
        validate_run_directory(run_id)
        if run_id not in known or run_id in seen:
            raise ValueError('unknown or duplicate Run in launch plan')
        seen.add(run_id)
        expected = study / 'runs' / run_id / 'config.yaml'
        if (not expected.resolve().is_relative_to(study) or not expected.is_file()
                or not job.get('config') or Path(str(job['config'])).resolve() != expected.resolve()
                or (study / str(known[run_id].get('config', ''))).resolve() != expected.resolve()):
            raise ValueError('launch config must match the indexed canonical Run config')
        config = load_config(expected)
        if config.get('id') != known[run_id].get('id'):
            raise ValueError('launch config identity differs from the index')
        mode, device = _job_fields(expected)
        if mode != job.get('mode') or device != job.get('device'):
            raise ValueError('launch job mode/device differs from its config')
        gpu = job.get('gpu')
        if device == 'cpu' and gpu is not None:
            raise ValueError('CPU jobs cannot carry a CUDA assignment')
        if device != 'cpu' and (type(gpu) not in (int, str) or not str(gpu).isdigit()):
            raise ValueError('CUDA jobs require an explicit GPU assignment')


def load_launch_plan(
    study_dir: Path | str,
    *,
    init_gpu: int,
    num_gpus: int,
    round_size: int,
    include_done: bool = False,
) -> dict[str, Any] | None:
    """Resume a saved plan; absent means make, malformed or changed means reject."""
    study_dir = Path(study_dir).resolve()
    path = study_dir / SCRIPTS_DIRNAME / JOBS_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('invalid saved launch plan') from exc
    if (not isinstance(payload, dict) or not isinstance(payload.get('jobs'), list)
            or Path(str(payload.get('study_dir', ''))).resolve() != study_dir):
        raise ValueError('launch plan must identify this runtime and a job list')
    if (any(type(payload.get(key)) is not int for key in ('init_gpu', 'num_gpus', 'round'))
            or payload['init_gpu'] < 0 or payload['num_gpus'] < 0):
        raise ValueError('launch resource fields must be nonnegative integers')
    if int(payload.get('init_gpu', 0)) != int(init_gpu) or int(payload.get('num_gpus', 1)) != int(num_gpus):
        raise ValueError('launch GPU assignment changed; use an explicit remake')
    stored_round = int(payload.get('round') or 1)
    if stored_round < 1 or (int(round_size) > 0 and stored_round != int(round_size)):
        raise ValueError('launch round changed; use an explicit remake')
    validate_launch_jobs(study_dir, payload['jobs'])
    assigned = set(gpu_ids(int(payload['init_gpu']), int(payload['num_gpus'])))
    for job in payload['jobs']:
        if job['device'] != 'cpu' and str(job['gpu']) not in assigned:
            raise ValueError('job GPU differs from the saved allocation')

    def keep(rows):
        return list(rows) if include_done else [job for job in rows if not run_succeeded(study_dir, job['run_id'])]

    jobs = keep(payload['jobs'])
    batches = None
    raw_batches = payload.get('batches')
    if raw_batches is not None:
        if not isinstance(raw_batches, list) or any(not isinstance(group, list) for group in raw_batches):
            raise ValueError('invalid launch batches')
        flat = [job for group in raw_batches for job in group]
        validate_launch_jobs(study_dir, flat)
        if sorted(json.dumps(job, sort_keys=True) for job in flat) != sorted(
                json.dumps(job, sort_keys=True) for job in payload['jobs']):
            raise ValueError('launch batches differ from the saved job list')
        batches = [kept for group in raw_batches if (kept := keep(group))]
    return {'jobs_json': path, 'job_list': jobs, 'round': stored_round, 'batches': batches,
            'n_jobs': len(jobs),
            'expand': {'study_dir': study_dir, 'index': study_dir / 'index.json', 'configs': []}}


def write_jobs_json(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    round_size: int,
    init_gpu: int,
    num_gpus: int,
    extra: dict[str, Any] | None = None,
    batches: list[list[dict[str, Any]]] | None = None,
) -> Path:
    payload = {
        'study_dir': str(study_dir.resolve()),
        'round': int(round_size),
        'init_gpu': int(init_gpu),
        'num_gpus': int(num_gpus),
        'jobs': jobs,
    }
    if batches:
        payload['batches'] = batches
    if extra:
        payload.update(extra)
    dest = scripts_dir(study_dir) / JOBS_NAME
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return dest


def wait_groups(
    jobs: list[dict[str, Any]],
    round_size: int,
    batches: list[list[dict[str, Any]]] | None = None,
) -> list[list[dict[str, Any]]]:
    if batches:
        return [group for group in batches if group]
    groups: list[list[dict[str, Any]]] = []
    for wave in job_waves(jobs):
        resources: dict[str, list[dict[str, Any]]] = {}
        for job in wave:
            device = str(job.get('device') or 'cuda').lower()
            resources.setdefault(device, []).append(job)
        for rows in resources.values():
            for start in range(0, len(rows), round_size):
                groups.append(rows[start : start + round_size])
    return groups


def render_bash(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    python_exe: str,
    round_size: int,
    split_round: int,
    batches: list[list[dict[str, Any]]] | None = None,
) -> list[str]:
    if round_size < 1:
        raise ValueError('round must be >= 1')
    if split_round < 1:
        raise ValueError('split_round must be >= 1')
    repo = repo_root_from(study_dir)
    py_b = bash_path(Path(python_exe))
    repo_b = bash_path(repo)
    chunks: list[str] = []
    groups = wait_groups(jobs, round_size, batches)
    partitions = [groups[start:start + split_round] for start in range(0, len(groups), split_round)] or [[]]
    for index, part in enumerate(partitions):
        body = [
            'from pathlib import Path',
            'from spec_mtp.structure.make.schedule import launch_jobs, run_succeeded',
            'from studies._shared.flow.cli import check_study_inputs',
            f'study = Path({str(study_dir.resolve())!r})',
            'check_study_inputs(study)',
            f'groups = {part!r}',
            'jobs = [job for group in groups for job in group]',
            f"codes = launch_jobs(study, jobs, round_size={round_size}, batches=groups, console='shared')",
            "failed = any(codes) or any(not run_succeeded(study, str(job['run_id'])) for job in jobs)",
        ]
        if index == len(partitions) - 1:
            body.extend([
                'from spec_mtp.flow.process import run_study',
                'result = run_study(study)',
                "print(study / 'process.json', flush=True)",
                "failed = failed or not result.get('complete', False)",
            ])
        body.append('raise SystemExit(int(failed))')
        chunks.append('\n'.join([
            '#!/bin/bash', f'cd {shlex.quote(repo_b)}', f'{shlex.quote(py_b)} - <<\'RPIPE_LAUNCH\'',
            *body, 'RPIPE_LAUNCH', '',
        ]))
    return chunks


def powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_launch_scripts(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    round_size: int = 4,
    split_round: int = 65535,
    python_exe: str | None = None,
    init_gpu: int = 0,
    num_gpus: int = 1,
    extra: dict[str, Any] | None = None,
    batches: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    py = python_exe or sys.executable
    dest = scripts_dir(study_dir)
    jobs_path = write_jobs_json(
        study_dir,
        jobs,
        round_size=round_size,
        init_gpu=init_gpu,
        num_gpus=num_gpus,
        extra=extra,
        batches=batches,
    )
    chunks = render_bash(
        study_dir,
        jobs,
        python_exe=py,
        round_size=round_size,
        split_round=split_round,
        batches=batches,
    )
    sh_paths: list[Path] = []
    stem = 'launch'
    for k, text in enumerate(chunks, start=1):
        name = f'{stem}_{k}.sh' if len(chunks) > 1 else f'{stem}.sh'
        path = dest / name
        path.write_text(text, encoding='utf-8', newline='\n')
        sh_paths.append(path)
    ps1 = dest / 'launch.ps1'
    study_win = str(study_dir.resolve())
    py_win = str(Path(py).resolve())
    ps1.write_text(
        '\n'.join(
            [
                '$ErrorActionPreference = "Stop"',
                f'Set-Location -LiteralPath {powershell_literal(str(repo_root_from(study_dir)))}',
                f'& {powershell_literal(py_win)} -m studies._shared.flow.cli launch {powershell_literal(study_win)} --console shared --init-gpu {int(init_gpu)} --num-gpus {int(num_gpus)} --round {int(round_size)}',
                'exit $LASTEXITCODE',
                '',
            ]
        ),
        encoding='utf-8-sig',
        newline='\n',
    )
    return {
        'scripts_dir': dest,
        'jobs_json': jobs_path,
        'bash': sh_paths,
        'ps1': ps1,
        'n_jobs': len(jobs),
    }


def job_popen_kwargs(console: str) -> dict[str, Any]:
    """Windows ``new`` = one console window per run-one (same idea as Start-Process)."""
    if console == 'new' and os.name == 'nt':
        flag = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0x00000010)
        return {'creationflags': flag}
    return {}


def resolve_console(mode: str = 'auto') -> str:
    raw = str(mode or 'auto').strip().lower()
    if raw in ('', 'auto'):
        return 'new' if os.name == 'nt' else 'shared'
    if raw in ('new', 'window', 'windows', 'separate'):
        return 'new'
    if raw in ('shared', 'same', 'mix'):
        return 'shared'
    raise ValueError('console must be auto, new, or shared')


def launch_job_env(job: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    owned = [str(Path(__file__).resolve().parents[3]), str(repo_root_from(Path(__file__)))]
    inherited = env.get('PYTHONPATH', '').split(os.pathsep)
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys([*owned, *(p for p in inherited if p)]))
    gpu = job.get('gpu')
    if gpu is not None and str(job.get('device') or 'cuda').lower() != 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def _write_interruption_receipt(study_dir: Path, body: dict[str, Any]) -> Path:
    """Publish once, after child cleanup; partial evidence is never auto-repaired."""
    study_dir = Path(study_dir).absolute()
    if study_dir.resolve() != study_dir or body.get('study') != str(study_dir):
        raise ValueError('interruption receipt requires its admitted canonical Study')
    attempt_id = uuid.uuid4().hex
    payload = json.dumps({**body, 'attempt_id': attempt_id}, ensure_ascii=True,
                         allow_nan=False, separators=(',', ':')) + '\n'
    if len(payload) > 65536:
        raise ValueError('interruption receipt exceeds 64 KiB')
    namespace = study_dir / SCRIPTS_DIRNAME / 'launch-attempts'
    for directory in (namespace.parent, namespace):
        if directory.resolve() != directory:
            raise ValueError('interruption receipt namespace must not be an alias')
        directory.mkdir(exist_ok=True)
        if directory.resolve() != directory:
            raise ValueError('interruption receipt namespace changed')
    # shortcut: append-only evidence has no automatic retention; deletion needs an ownership policy.
    attempt = namespace / attempt_id
    attempt.mkdir()
    if attempt.resolve() != attempt:
        raise ValueError('interruption attempt must not be an alias')
    partial, receipt = attempt / 'partial.json', attempt / 'receipt.json'
    with partial.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if attempt.resolve() != attempt or partial.resolve() != partial:
        raise ValueError('interruption attempt changed before publication')
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError('interruption receipt already exists')
    os.rename(partial, receipt)
    return receipt


def launch_jobs(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    round_size: int = 4,
    python_exe: str | None = None,
    cwd: Path | None = None,
    batches: list[list[dict[str, Any]]] | None = None,
    retry_failed: bool = False,
    console: str = 'auto',
) -> list[int]:
    if round_size < 1:
        raise ValueError('round must be >= 1')
    study_dir = validate_study_root(study_dir)
    diagnostic_assets.check_diagnostic_budget(study_dir)
    declared = load_study_yaml(study_dir) if (study_dir / 'study.yaml').is_file() else {}
    check_frozen(study_dir, declared)
    validate_launch_jobs(study_dir, jobs)
    if batches is not None:
        flattened = [job for group in batches for job in group]
        validate_launch_jobs(study_dir, flattened)
        if sorted(json.dumps(job, sort_keys=True) for job in flattened) != sorted(
                json.dumps(job, sort_keys=True) for job in jobs):
            raise ValueError('dispatch batches differ from the admitted jobs')
    py = python_exe or sys.executable
    root = cwd or repo_root_from(study_dir)
    groups = wait_groups(jobs, round_size, batches)
    run_ids = [str(job['run_id']) for group in groups for job in group]
    for run_id in run_ids:
        validate_run_directory(run_id)
    if len(run_ids) != len(set(run_ids)):
        raise ValueError('duplicate Run in launch plan; use a new version for repeated computation')
    for run_id in run_ids:
        diagnostic_assets.check_diagnostic_budget(study_dir, run_id)
    limits: dict[str, float] = {}
    limit_hashes: dict[str, str] = {}
    missing = object()
    for job in jobs:
        path = Path(job['config'])
        system = load_config(path).get('system')
        value = get_setting(system, 'worker_timeout_seconds', missing) if isinstance(system, dict) else missing
        if value is missing:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('worker_timeout_seconds must be a positive finite number')
        try:
            seconds = float(value)
        except OverflowError as exc:
            raise ValueError('worker_timeout_seconds must be finite') from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('worker_timeout_seconds must be a positive finite number')
        if declared.get('freeze') is not True:
            raise ValueError('worker_timeout_seconds requires freeze: true and frozen provenance')
        run_id = str(job['run_id'])
        limits[run_id], limit_hashes[run_id] = seconds, file_sha256(path)
    codes: list[int] = []
    console_mode = resolve_console(console)
    popen_extra = job_popen_kwargs(console_mode)
    try:
        index = load_index(study_dir)
    except (OSError, TypeError, ValueError):
        index = {}
    dependencies = {
        run_id: parent
        for exp in index.get('experiments') or []
        for run in exp.get('runs') or []
        if (run_id := str(run.get('run_dir') or run.get('id') or ''))
        if (parent := sibling_train_dependency(study_dir, run_id)) is not None
    }
    # A supplied batch must not bypass the train/eval barrier either.
    waves = [
        [kept for group in groups if (kept := [job for job in group if (job_mode(job) == 'eval') == is_eval])]
        for is_eval in (False, True)
    ]
    total_waits = sum(len(wave) for wave in waves)
    wait_index = 0

    def _group_mode(group: list[dict[str, Any]]) -> str:
        modes: list[str] = []
        for job in group:
            mode = str(job.get('mode') or 'train')
            if mode not in modes:
                modes.append(mode)
        return '+'.join(modes) or 'train'

    def _run_groups(
        groups: list[list[dict[str, Any]]],
        *,
        kind: str,
    ) -> list[tuple[dict[str, Any], int]]:
        nonlocal wait_index
        pairs: list[tuple[dict[str, Any], int]] = []
        total = len(groups)
        for index, group in enumerate(groups, start=1):
            if kind == 'wait':
                wait_index += 1
            position = f'{wait_index}/{total_waits}' if kind == 'wait' else f'{index}/{total}'
            print(f'launch: {kind} {position} mode={_group_mode(group)}', flush=True)
            index_sha = file_sha256(study_dir / 'index.json')
            admitted_jobs = [{'run_id': str(job['run_id']),
                              'config_sha256': file_sha256(Path(job['config']))} for job in group]
            if any(row['run_id'] in limit_hashes and row['config_sha256'] != limit_hashes[row['run_id']]
                   for row in admitted_jobs):
                raise ValueError('timed Run config changed since deadline admission')
            procs: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
            monitored = any(str(job['run_id']) in limits for job in group)
            started: dict[str, float] = {}
            observed: dict[str, int] = {}
            deadline_report: dict[str, Any] | None = None

            def observe() -> None:
                nonlocal deadline_report
                for selected, process in procs:
                    name = str(selected['run_id'])
                    if name not in observed and process.poll() is not None:
                        observed[name] = int(process.wait(timeout=0))
                now = time.monotonic()
                expired = [name for name in started if name not in observed
                           and now >= started[name] + limits[name]]
                if expired:
                    name = min(expired, key=lambda selected: started[selected] + limits[selected])
                    elapsed = now - started[name]
                    deadline_report = {'run_id': name, 'worker_timeout_seconds': limits[name],
                                       'elapsed_seconds': elapsed, 'clock': 'monotonic',
                                       'start': 'before_popen', 'poll_interval_seconds': .05}
                    failure = subprocess.TimeoutExpired(['run-one', name], limits[name])
                    setattr(failure, 'run_id', name)
                    setattr(failure, 'worker_timeout_seconds', limits[name])
                    setattr(failure, 'elapsed_seconds', elapsed)
                    raise failure

            try:
                for job in group:
                    if monitored:
                        observe()
                    run_id = str(job['run_id'])
                    parent = dependencies.get(run_id)
                    if parent is not None and not run_succeeded(study_dir, parent):
                        message = f'blocked: sibling train {parent} did not succeed'
                        print(f'error {run_id} {message}', flush=True)
                        pairs.append((job, 1))
                        continue
                    cmd = [py, '-m', 'studies._shared.flow.cli', 'run-one', str(study_dir), run_id]
                    where = 'window' if console_mode == 'new' and os.name == 'nt' else 'here'
                    resource = f'gpu={job["gpu"]}' if job.get('gpu') is not None and str(job.get('device') or 'cuda').lower() != 'cpu' else 'cpu'
                    print(f'+ {resource} {job["run_id"]} ({where})', flush=True)
                    if run_id in limits:
                        started[run_id] = time.monotonic()
                    procs.append(
                        (
                            job,
                            subprocess.Popen(
                                cmd,
                                cwd=str(root),
                                env=launch_job_env(job),
                                **popen_extra,
                            ),
                        )
                    )
                if monitored:
                    while len(observed) < len(procs):
                        observe()
                        if len(observed) < len(procs):
                            remaining = [started[name] + limits[name] - time.monotonic()
                                         for name in started if name not in observed]
                            delay = min(.05, min(remaining, default=.05))
                            if delay > 0:
                                time.sleep(delay)
                for job, proc in procs:
                    code = observed[str(job['run_id'])] if monitored else int(proc.wait())
                    pairs.append((job, code))
                    if code != 0:
                        print(f'error {job["run_id"]} exit={code}', flush=True)
            except BaseException as primary:
                workers: list[dict[str, Any]] = []
                for job, proc in procs:
                    row: dict[str, Any] = {'run_id': str(job['run_id']), 'terminate_requested': False,
                                          'kill_requested': False, 'direct_exit_confirmed': False,
                                          'cleanup_error_types': []}
                    pid = getattr(proc, 'pid', None)
                    if type(pid) is int and pid > 0:
                        row['pid'] = pid
                    if str(job['run_id']) in started:
                        row['worker_timeout_seconds'] = limits[str(job['run_id'])]
                        row['elapsed_seconds'] = time.monotonic() - started[str(job['run_id'])]
                    workers.append(row)
                for (_job, proc), row in zip(procs, workers):
                    try:
                        if proc.poll() is None:
                            row['terminate_requested'] = True
                            proc.terminate()
                    except BaseException as cleanup:
                        row['cleanup_error_types'].append(type(cleanup).__name__)
                        setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
                for (_job, proc), row in zip(procs, workers):
                    try:
                        code = proc.wait(timeout=5)
                        if type(code) is not int:
                            raise TypeError('worker wait did not return an integer exit code')
                        row.update(direct_exit_confirmed=True, returncode=code)
                    except subprocess.TimeoutExpired:
                        row['wait_timed_out'] = True
                        try:
                            row['kill_requested'] = True
                            proc.kill()
                            code = proc.wait(timeout=5)
                            if type(code) is not int:
                                raise TypeError('worker wait did not return an integer exit code')
                            row.update(direct_exit_confirmed=True, returncode=code)
                        except BaseException as cleanup:
                            row['cleanup_error_types'].append(type(cleanup).__name__)
                            setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
                    except BaseException as cleanup:
                        row['cleanup_error_types'].append(type(cleanup).__name__)
                        setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
                try:
                    if hasattr(primary, 'cleanup_receipt_path'):
                        delattr(primary, 'cleanup_receipt_path')
                    receipt_body: dict[str, Any] = {
                        'schema': 1, 'scope': 'owned_current_group_only', 'study': str(study_dir),
                        'observed_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                        'launcher_pid': os.getpid(), 'primary_exception_type': type(primary).__name__,
                        'cleanup_wait_seconds': 5, 'index_sha256': index_sha, 'jobs': admitted_jobs,
                        'workers': workers, 'descendants': 'unknown', 'writer_stop': 'unknown',
                        'unreturned_popen': 'unknown',
                    }
                    if deadline_report is not None:
                        receipt_body['deadline'] = deadline_report
                    receipt = _write_interruption_receipt(study_dir, receipt_body)
                    setattr(primary, 'cleanup_receipt_path', str(receipt))
                except BaseException as cleanup:
                    setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
                raise
        return pairs

    for wave in waves:
        pairs = _run_groups(wave, kind='wait')
        codes.extend(code for _, code in pairs)
        failed = [
            job for job, code in pairs
            if (code != 0 or not run_succeeded(study_dir, str(job['run_id'])))
            and (dependencies.get(str(job['run_id'])) is None
                 or run_succeeded(study_dir, dependencies[str(job['run_id'])]))
        ]
        if retry_failed and failed:
            print(f'retry {len(failed)} failed jobs (resume latest)', flush=True)
            retry_pairs = _run_groups([[job] for job in failed], kind='retry')
            codes.extend(code for _, code in retry_pairs)
    return codes
