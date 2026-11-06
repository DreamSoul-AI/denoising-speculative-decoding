"""UTF-8 diagnostic JSONL; the console renders the same sanitized record."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import threading
import traceback
from typing import Any, Iterator
import unicodedata
import uuid

from spec_mtp import __version__
from spec_mtp.structure.artifact.asset import kinds
from spec_mtp.structure.artifact.asset.tree import diagnostic_files

_RUNTIME_ID = uuid.uuid4().hex
_WRITE_LOCK = threading.Lock()
_SEGMENT_BYTES = 4 * 1024 * 1024

_EVENTS = {'flow': 'task.progress', 'warn': 'operation.warning', 'error': 'task.failed',
           'epoch': 'algorithm.epoch_completed', 'split': 'algorithm.split_completed',
           'ckpt': 'checkpoint.saved', 'resume': 'checkpoint.resumed'}
_PRIVATE = re.compile(r'(?i)\b(password|passwd|token|authorization|api[_-]?key|secret|credential|prompt|input|content|document|reasoning|path|email)\s*[=:]\s*[\s\S]*')
_PATH = re.compile(r'(?i)(?:[a-z]:[\\/]|(?<!\w)/)[^\s;,]*')
_URL = re.compile(r'(?i)\b(?:https?|ftp)://[^\s]+')
_EMAIL = re.compile(r'[^\s@]+@[^\s@]+\.[^\s@]+')


def _text(message: str) -> str:
    message = _PRIVATE.sub(lambda match: match[1] + '=[omitted]', message)
    for pattern in (_PATH, _URL, _EMAIL):
        message = pattern.sub('[omitted]', message)
    message = ''.join(char if not unicodedata.category(char).startswith('C') and char not in '\u2028\u2029' else '\\u%04x' % ord(char)
                      for char in message)
    return message[:2048]


class Logger:
    def __init__(self, assets_dir: Path | str, *, managed: bool = False) -> None:
        self.managed = managed
        self.failure_count = 0
        self.last_error_type: str | None = None
        self._reported_failure = False
        self.assets_dir = Path(assets_dir)
        parts = self.assets_dir.parts
        self.run_id = parts[-2] if len(parts) >= 3 and parts[-1] == 'assets' and parts[-3] == 'runs' else None
        self.runtime_id = f'{os.getpid()}:{_RUNTIME_ID}'
        self.path = self.assets_dir / kinds.RUN_LOG
        with self._guard():
            with _WRITE_LOCK:
                self.path.parent.resolve().relative_to(self.assets_dir.resolve())
                diagnostic_files(self.path)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if not self.path.exists():
                    with self.path.open('x', encoding='utf-8'):
                        pass

    @contextmanager
    def _guard(self) -> Iterator[None]:
        try:
            yield
        except Exception as exc:
            if not self.managed:
                raise
            self.failure_count += 1
            self.last_error_type = type(exc).__name__
            if not self._reported_failure:
                self._reported_failure = True
                category = self.last_error_type if self.last_error_type.isidentifier() else 'Exception'
                try:
                    print(f'spec_mtp diagnostic failure: {category}; completeness unknown',
                          file=sys.stderr, flush=True)
                except Exception:
                    pass

    def _emit(self, level: str, event: str, message: str, **fields: Any) -> None:
        with self._guard():
            record = {
                'timestamp': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
                'level': level, 'event': _EVENTS.get(event, 'logging.unstructured_record'),
                'logger': 'spec_mtp', 'message': _text(message), 'app_name': 'spec_mtp',
                'app_version': __version__, 'runtime_id': self.runtime_id,
                'process_name': multiprocessing.current_process().name, 'pid': os.getpid(),
            }
            if self.run_id is not None:
                record['task_id'] = self.run_id
            record.update(fields)
            payload = json.dumps(record, ensure_ascii=True, allow_nan=False)
            line = payload + '\n'
            if len(line) > 65536:
                raise ValueError('diagnostic record exceeds 65536 characters including LF')
            if len(line) > _SEGMENT_BYTES:
                raise ValueError('diagnostic record exceeds segment byte limit')
            with _WRITE_LOCK:
                parts = diagnostic_files(self.path)
                path = parts[-1]
                with path.open('rb') as handle:
                    size = handle.seek(0, 2)
                    if size:
                        handle.seek(-1, 2)
                        if handle.read(1) != b'\n':
                            raise ValueError('diagnostic tail must end in LF')
                mode = 'a'
                if self.path.name == 'events.jsonl' and size + len(line) > _SEGMENT_BYTES:
                    path = path.with_name(f'events.{len(parts):06d}.jsonl')
                    mode = 'x'
                with path.open(mode, encoding='utf-8', newline='\n') as handle:
                    handle.write(line)
                    handle.flush()
            error_type = f' [{record["exception_type"]}]' if 'exception_type' in record else ''
            print(f"{record['timestamp']} {level} [{record['event']}] {record['message']}{error_type}", flush=True)

    def info(self, message: str, *, event: str = 'flow') -> None:
        self._emit('INFO', event, message)

    def warning(self, message: str) -> None:
        self._emit('WARNING', 'warn', message)

    def error(self, message: str) -> None:
        self._emit('ERROR', 'error', message)

    def exception(self, message: str, exc: BaseException) -> None:
        # Exception text and source lines may contain credentials or prompt payloads.
        with self._guard():
            frames = traceback.extract_tb(exc.__traceback__)
            stack = '\n'.join(f'{frame.name}:{frame.lineno}' for frame in frames)[-4096:]
            self._emit('ERROR', 'error', message, exception_type=type(exc).__name__, traceback=stack)

    def report(self, tracker: Any, split: str, extra: dict[str, Any] | None = None) -> None:
        with self._guard():
            extra = extra or {}
            epoch = extra.get('epoch')
            event = 'epoch' if epoch is not None else 'split'
            parts: list[str] = []
            if epoch is not None:
                parts.append(str(epoch))
            parts.append(f'[split] {split}')
            means = {}
            lasts = {}
            if tracker is not None:
                display = getattr(tracker, 'display_mean', None)
                means = display(split) if callable(display) else tracker.mean(split)
                lasts = tracker.last(split)
            metrics: list[str] = []
            for name in sorted(set(means) | set(lasts)):
                mean = means.get(name)
                last = lasts.get(name)
                if mean is not None:
                    metrics.append(f'{name}={mean:.4f}')
                elif last is not None:
                    metrics.append(f'{name}={last:.4f}')
            if metrics:
                parts.append('[metric] ' + ' '.join(metrics))
            time_bits: list[str] = []
            if extra.get('elapsed') is not None:
                time_bits.append(f"elapsed={extra['elapsed']}")
            if extra.get('eta') is not None:
                time_bits.append(f"eta={extra['eta']}")
            if time_bits:
                parts.append('[time] ' + ' '.join(time_bits))
            skip = {'epoch', 'elapsed', 'eta'}
            for key, value in extra.items():
                if key in skip:
                    continue
                parts.append(f'{key}={value}')
            self._emit('INFO', event, ' '.join(parts))

    def state_dict(self) -> dict[str, Any]:
        return {'path': str(self.path)}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        del state
