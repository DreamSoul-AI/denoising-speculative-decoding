"""R1 local backport: OS-owned, non-reentrant Run admission and attempt identity."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import sys
import uuid

from rpipe.structure.artifact._atomic import atomic_write_text

RUN_ATTEMPT_ENV = 'RPIPE_RUN_ATTEMPT_TOKEN'
RUN_BUSY_EXIT = 75


class RunLockedError(TimeoutError):
    """Another execution already owns this Run."""


@contextmanager
def run_lock(root: Path | str):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / '.rpipe.lock').open('a+b')
    try:
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            raise RunLockedError(f'Run already executing: {root}') from exc
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b'0')
                handle.flush()
            yield
        finally:
            primary = sys.exc_info()[1]
            try:
                if os.name == 'nt':
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as cleanup:
                if primary is None:
                    raise
                if hasattr(primary, 'add_note'):
                    primary.add_note(f'Run unlock failed: {cleanup!r}')
                else:
                    primary.__cause__ = cleanup
    finally:
        primary = sys.exc_info()[1]
        try:
            handle.close()
        except OSError as cleanup:
            if primary is None:
                raise
            if hasattr(primary, 'add_note'):
                primary.add_note(f'Run lock close failed: {cleanup!r}')
            else:
                primary.__cause__ = cleanup


def start_execution(root: Path | str, token: str | None = None) -> str:
    token = token or uuid.uuid4().hex
    atomic_write_text(Path(root) / 'execution.json', json.dumps({
        'attempt_token': token, 'pid': os.getpid(),
    }, ensure_ascii=True) + '\n')
    return token


def owns_execution(root: Path | str, token: str) -> bool:
    try:
        body = json.loads((Path(root) / 'execution.json').read_text(encoding='utf-8'))
        return isinstance(body, dict) and body.get('attempt_token') == token
    except (OSError, ValueError, RecursionError):
        return False
