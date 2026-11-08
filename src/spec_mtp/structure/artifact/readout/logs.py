"""Study diagnostic readouts and explicit opt-in capture retention."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any

from spec_mtp.structure.artifact.asset import tree as asset_tree
from spec_mtp.structure.artifact.asset.tree import diagnostic_inventory
from spec_mtp.structure.artifact.index import load_index
from spec_mtp.structure.artifact.paths import RESULT_NAME, RUNS_DIRNAME
from spec_mtp.structure.artifact.layout import validate_run_directory, validate_study_root
from spec_mtp.structure.artifact.readout.status import iter_diagnostic_lines, log_path

_STAMP = re.compile(r'^(\d{4}-\d{2}-\d{2}T\S+)')
_EVENT = re.compile(r'\[(epoch|error|flow|warn|ckpt|resume)\]\s*(.*)$')


def event_lines(study_dir: Path | str) -> list[str]:
    study_dir = Path(study_dir).resolve()
    index = load_index(study_dir)
    rows: list[tuple[float, str]] = []
    for exp in index.get('experiments') or []:
        for run in exp.get('runs') or []:
            run_dir = str(run.get('run_dir') or run.get('id') or '')
            path = log_path(study_dir, run, run_dir)
            if path is None or not path.is_file():
                continue
            start = len(rows)
            try:
                for line in iter_diagnostic_lines(path):
                    match = _EVENT.search(line)
                    if match is None:
                        continue
                    payload = match.group(2).strip()
                    if match.group(1) == 'error' and (
                        payload.startswith('Traceback') or payload.startswith('File ')
                    ):
                        continue
                    stamp = _STAMP.match(line)
                    try:
                        moment = datetime.fromisoformat(stamp.group(1).replace('Z', '+00:00')) if stamp else None
                        instant = moment.timestamp() if moment is not None and moment.tzinfo is not None else float('inf')
                    except ValueError:
                        instant = float('inf')
                    rows.append((instant, line))
            except OSError:
                del rows[start:]
    rows.sort(key=lambda item: item[0])
    return [line for _, line in rows]


def format_logs(study_dir: Path | str) -> str:
    lines = event_lines(study_dir)
    if not lines:
        return ''
    return '\n'.join(lines) + '\n'


def _indexed_log_roots(study_dir: Path) -> list[tuple[str, Path]]:
    """Admit all indexed identities before selecting a Run for a mutation."""
    index = load_index(study_dir)
    roots: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    seen_roots: set[Path] = set()
    for experiment in index.get('experiments') or []:
        for run in experiment.get('runs') or []:
            run_dir = run.get('run_dir') or run.get('id')
            validate_run_directory(run_dir)
            run_id = run.get('id') or run_dir
            validate_run_directory(run_id)
            run_root = study_dir / RUNS_DIRNAME / run_dir
            logs_dir = run_root / 'assets' / 'logs'
            if run_root.resolve() != run_root or logs_dir.resolve() != logs_dir:
                raise ValueError('inventory Run paths must retain their canonical identity')
            if run_id in seen_ids or run_root in seen_roots:
                raise ValueError('duplicate inventory Run identity')
            seen_ids.add(run_id)
            seen_roots.add(run_root)
            raw = run.get('log')
            if raw is not None:
                if not isinstance(raw, str) or not raw:
                    raise ValueError('inventory log reference must be a nonempty relative path')
                reference = Path(raw)
                expected = Path(RUNS_DIRNAME) / run_dir / 'assets' / 'logs'
                if (reference.is_absolute() or reference.drive or '..' in reference.parts
                        or reference not in (expected / 'events.jsonl', expected / 'run.log')):
                    raise ValueError('inventory log reference must belong to this exact Run')
            roots.append((run_id, run_root))
    return roots


def inventory_logs(study_dir: Path | str) -> dict[str, Any]:
    """Read only each indexed Run's canonical diagnostic family."""
    study_dir = Path(study_dir).resolve()
    now = datetime.now(timezone.utc)
    rows = [{'run_id': run_id, **diagnostic_inventory(root / 'assets/logs/events.jsonl', now)}
            for run_id, root in _indexed_log_roots(study_dir)]
    return {'scope': 'run_diagnostics_only',
            'observed_at': now.isoformat().replace('+00:00', 'Z'), 'runs': rows}


def retention_logs(
    study_dir: Path | str,
    run_id: str,
    action: str,
    *,
    seal_sha256: str | None = None,
) -> dict[str, Any]:
    """Explicitly manage only one registered Run's opt-in diagnostic capture."""
    validate_run_directory(run_id)
    if action not in ('prepare', 'seal', 'purge'):
        raise ValueError('unknown diagnostic retention action')
    if ((action == 'purge' and (not isinstance(seal_sha256, str)
                               or re.fullmatch(r'[0-9a-f]{64}', seal_sha256) is None))
            or (action != 'purge' and seal_sha256 is not None)):
        raise ValueError('a seal SHA256 is required only for diagnostic purge')
    study = validate_study_root(study_dir)
    roots = dict(_indexed_log_roots(study))
    if run_id not in roots:
        raise ValueError('diagnostic retention requires an exact registered Run ID')
    root = roots[run_id]
    if not root.is_dir():
        raise FileNotFoundError('registered Run directory is absent')
    result = root / RESULT_NAME
    if action == 'prepare' and (result.exists() or result.is_symlink()):
        raise ValueError('diagnostic capture preparation requires an unexecuted Run')
    anchor = root / 'assets/logs/events.jsonl'
    if action == 'prepare':
        detail = asset_tree.prepare_diagnostic_capture(anchor)
    elif action == 'seal':
        detail = {'state': 'sealed', 'seal_sha256': asset_tree.seal_diagnostic_capture(anchor)}
    else:
        assert seal_sha256 is not None
        detail = asset_tree.purge_sealed_diagnostics(anchor, seal_sha256)
    return {'scope': 'run_diagnostics_only', 'run_id': run_id, 'action': action, **detail}
