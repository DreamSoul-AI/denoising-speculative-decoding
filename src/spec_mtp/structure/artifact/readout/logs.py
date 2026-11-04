"""Print Study diagnostic event lines in timestamp order. Does not write a file."""

from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path

from spec_mtp.structure.artifact.index import load_index
from spec_mtp.structure.artifact.readout.status import diagnostic_lines, log_path

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
            try:
                lines = diagnostic_lines(path)
            except OSError:
                continue
            for line in lines:
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
    rows.sort(key=lambda item: item[0])
    return [line for _, line in rows]


def format_logs(study_dir: Path | str) -> str:
    lines = event_lines(study_dir)
    if not lines:
        return ''
    return '\n'.join(lines) + '\n'
