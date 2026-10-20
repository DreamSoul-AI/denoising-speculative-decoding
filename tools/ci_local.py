"""Run .github/workflows/cpu-checks.yml locally, step by step, without Docker.

Command-level replay of this workflow: a clean ``git worktree`` of HEAD
stands in for actions/checkout, a fresh Python venv of the workflow's
``python-version`` (via ``uv``) stands in for actions/setup-python, and every
``run:`` block is executed verbatim with ``bash -e`` and the step ``env``.
GitHub-hosted runner image, caches and the PR merge commit are not reproduced.

    python tools/ci_local.py [--workflow .github/workflows/cpu-checks.yml] [--keep]

Work dirs live under ``.tmp/ci_local/<run-id>/`` (gitignored). Concurrent
invocations each get their own directory (timestamp + pid) so two runs do not
share a worktree or venv; an advisory lock on ``.tmp/ci_local/ci_local.lock``
serialises the git worktree add/remove against the same repo.
"""

from __future__ import annotations

import argparse
import json
import locale
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__:
    from ._process_tree import ProcessTree
else:
    from _process_tree import ProcessTree

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd, *, timeout=300, check=True, capture_output=False, **kw):
    """Bound a command, collect output and stop owned workers on every exit."""
    print('+', cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)), flush=True)
    if not math.isfinite(timeout) or timeout <= 0:
        raise subprocess.TimeoutExpired(cmd, timeout)
    if capture_output:
        kw.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tree = ProcessTree()
    captures = {}
    failures = []
    primary = None
    stdout = stderr = None
    before = None
    deadline = time.monotonic() + timeout
    try:
        # Files avoid pipe backpressure and inherited-pipe EOF waits after the
        # direct child exits. The owned tree is stopped before reading them.
        for stream in ('stdout', 'stderr'):
            if kw.get(stream) == subprocess.PIPE:
                captures[stream] = tempfile.TemporaryFile()
                kw[stream] = captures[stream]
        proc = tree.start(cmd, **kw)
        if getattr(proc, 'stdin', None) is not None:
            proc.stdin.close()  # Match communicate() with no input: deliver EOF.
        code = tree.wait(max(0, deadline - time.monotonic()))
        if check and code:
            raise subprocess.CalledProcessError(code, cmd)
    except BaseException as error:
        primary = error
    finally:
        proc = tree.proc
        if proc is not None:
            try:
                before = tree.terminate(timeout=10)
            except BaseException as cleanup:
                failures.append({'phase': 'process-tree', 'error': repr(cleanup)})
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=10)
            except BaseException as cleanup:
                failures.append({'phase': 'direct-process', 'error': repr(cleanup)})
        if tree.job is not None:
            failures.extend(tree.job.cleanup_errors)
        try:
            tree.close()
        except BaseException as cleanup:
            failures.append({'phase': 'ownership-handle', 'error': repr(cleanup)})
        output = {}
        for stream, handle in captures.items():
            try:
                handle.seek(0)
                value = handle.read(os.fstat(handle.fileno()).st_size)
                if kw.get('text') or kw.get('universal_newlines') or kw.get('encoding') or kw.get('errors'):
                    value = value.decode(kw.get('encoding') or locale.getpreferredencoding(False),
                                         errors=kw.get('errors') or 'strict')
                    value = value.replace('\r\n', '\n').replace('\r', '\n')
                output[stream] = value
            except BaseException as cleanup:
                failures.append({'phase': 'output-drain', 'error': repr(cleanup)})
            finally:
                try:
                    handle.close()
                except BaseException as cleanup:
                    failures.append({'phase': 'capture-close', 'error': repr(cleanup)})
        stdout, stderr = output.get('stdout'), output.get('stderr')
    if primary is None and failures:
        primary = (subprocess.CalledProcessError(proc.returncode, cmd) if proc is not None and proc.returncode
                   else RuntimeError('Owned process cleanup or output collection unconfirmed'))
    result = primary or subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    result.cleanup_errors = failures
    result.cleanup_confirmed = not failures and (proc is None or proc.returncode is not None)
    result.process_id = proc.pid if proc is not None else None
    result.direct_process_returncode = proc.returncode if proc is not None else None
    result.owned_processes_before_cleanup = before
    if primary is not None:
        if failures and hasattr(primary, 'add_note'):
            primary.add_note(f'Owned process cleanup or output collection unconfirmed: {failures}')
        if isinstance(primary, subprocess.SubprocessError):
            primary.output, primary.stderr = stdout, stderr
        raise primary
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow', default='.github/workflows/cpu-checks.yml')
    parser.add_argument('--keep', action='store_true', help='keep the worktree and venv')
    args = parser.parse_args()
    workflow = yaml.safe_load((ROOT / args.workflow).read_text(encoding='utf-8'))
    base = ROOT / '.tmp' / 'ci_local'
    base.mkdir(parents=True, exist_ok=True)
    run = base / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    run.mkdir()
    # Serialise worktree mutations: two overlapping runs used to share
    # ``.tmp/ci_local/cpu-checks/`` and race on pip install / worktree remove.
    lock_path = base / 'ci_local.lock'
    lock_file = lock_path.open('a+', encoding='utf-8')
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    results = []
    keep = args.keep
    artifacts = False
    primary = None
    try:
        for job_name, job in workflow['jobs'].items():
            job_start = time.monotonic()
            deadline = job_start + float(job.get('timeout-minutes', 360)) * 60
            work = run / job_name / 'repo'
            venv = run / job_name / 'venv'
            try:
                sh(['git', 'worktree', 'add', '--detach', str(work), 'HEAD'], cwd=ROOT,
                   timeout=deadline - time.monotonic())
                for step in job['steps']:
                    uses = step.get('uses', '')
                    if uses.startswith('actions/checkout'):
                        results.append((job_name, 'checkout (git worktree HEAD)', 'ok',
                                        time.monotonic() - job_start))
                        continue
                    if uses.startswith('actions/setup-python'):
                        version = str(step['with']['python-version'])
                        start = time.monotonic()
                        sh(['uv', 'venv', '-q', '--seed', '-p', version, str(venv)],
                           timeout=deadline - time.monotonic())
                        results.append((job_name, f'setup-python {version} (uv venv)', 'ok',
                                        time.monotonic() - start))
                        continue
                    if uses.startswith('actions/upload-artifact'):
                        start = time.monotonic()
                        source = work / step['with']['path']
                        if source.is_dir():
                            shutil.copytree(source, work.parent / 'artifacts' / step['with']['name'])
                            artifacts = True
                        results.append((job_name, step.get('name', 'artifact'), 'ok', time.monotonic() - start))
                        continue
                    if 'run' not in step:
                        raise SystemExit(f'unsupported step: {step}')
                    env = {**os.environ, **{k: str(v) for k, v in (step.get('env') or {}).items()},
                           'VIRTUAL_ENV': str(venv),
                           'PATH': f'{venv / "bin"}{os.pathsep}{os.environ["PATH"]}'}
                    env.pop('PYTHONPATH', None)
                    start = time.monotonic()
                    print(f'\n=== {job_name}: {step.get("name", "run")} ===', flush=True)
                    log = work.parent / f'step-{len(results):02d}.log'
                    print(f'output: {log}', flush=True)
                    with log.open('w', encoding='utf-8') as output:
                        try:
                            proc = sh(['bash', '-e', '-c', step['run']], cwd=work, env=env,
                                      stdout=output, stderr=subprocess.STDOUT, check=False,
                                      timeout=min(deadline - time.monotonic(),
                                                  float(step.get('timeout-minutes', 360)) * 60))
                            status = 'ok' if proc.returncode == 0 else f'FAILED ({proc.returncode})'
                        except subprocess.TimeoutExpired as error:
                            failures = getattr(error, 'cleanup_errors', [])
                            status = 'TIMEOUT (cleanup unconfirmed)' if failures else 'TIMEOUT'
                            output.write(json.dumps({'timeout': str(error), 'cleanup_errors': failures,
                                                     'process_id': getattr(error, 'process_id', None)}) + '\n')
                    results.append((job_name, step.get('name', 'run'), status, time.monotonic() - start))
                    if status != 'ok':
                        keep = True
                        break
            except BaseException as error:
                keep = True
                results.append((job_name, 'setup/execution', f'{type(error).__name__}: {error}',
                                time.monotonic() - job_start))
                raise
            finally:
                if not keep:
                    try:
                        sh(['git', 'worktree', 'remove', '--force', str(work)], cwd=ROOT, timeout=30)
                    except BaseException as error:
                        keep = True
                        results.append((job_name, 'cleanup', f'{type(error).__name__}: {error}',
                                        time.monotonic() - job_start))
                        raise
    except BaseException as error:
        primary = error
        keep = True
        raise
    finally:
        final_errors = []
        finalizers = [
            ('unlock', lambda: fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN) if fcntl is not None else None),
            ('lock-close', lock_file.close),
            ('summary', lambda: (run / 'summary.json').write_text(
                json.dumps(results, indent=2) + '\n', encoding='utf-8')),
        ]
        for phase, finalize in finalizers:
            try:
                finalize()
            except BaseException as error:
                keep = True
                final_errors.append((phase, error))
        if not keep and not artifacts:
            shutil.rmtree(run, ignore_errors=True)
        else:
            try:
                print(f'kept local CI evidence: {run}', flush=True)
            except BaseException as error:
                final_errors.append(('diagnostic', error))
        if final_errors:
            failure = primary if primary is not None else final_errors[0][1]
            failure.finalization_errors = [{'phase': phase, 'error': repr(error)} for phase, error in final_errors]
            if hasattr(failure, 'add_note'):
                failure.add_note(f'Local CI finalization failed: {failure.finalization_errors}')
            if primary is None:
                raise failure

    print('\nsummary:')
    for job_name, name, status, seconds in results:
        print(f'  {job_name:12s} {status:12s} {seconds:7.1f}s  {name}')
    return 0 if all(r[2] == 'ok' for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
