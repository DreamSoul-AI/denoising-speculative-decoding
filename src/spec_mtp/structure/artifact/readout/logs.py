"""Print Study diagnostic event lines in timestamp order. Does not write a file."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any

from spec_mtp.structure.artifact.asset.tree import diagnostic_inventory
from spec_mtp.structure.artifact.index import load_index
from spec_mtp.structure.artifact.paths import RUNS_DIRNAME
from spec_mtp.structure.artifact.layout import validate_run_directory
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


def inventory_logs(study_dir: Path | str) -> dict[str, Any]:
    """Read only each indexed Run's canonical diagnostic family."""
    study_dir = Path(study_dir).resolve()
    index = load_index(study_dir)
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
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
            rows.append({'run_id': run_id, **diagnostic_inventory(logs_dir / 'events.jsonl', now)})
    return {'scope': 'run_diagnostics_only',
            'observed_at': now.isoformat().replace('+00:00', 'Z'), 'runs': rows}
