"""Real lock handles with controlled acquisition, marker and close I/O failures."""

import errno
import os
from pathlib import Path

import pytest

from rpipe.structure.artifact.execution import run_lock


@pytest.mark.parametrize('stage,kind', [
    ('body', RuntimeError), ('body', KeyboardInterrupt), ('body', OSError),
    ('acquire', OSError), ('marker', OSError), ('none', None),
])
def test_lock_close_preserves_primary_and_exposes_secondary(tmp_path, monkeypatch, stage, kind):
    primary = kind(errno.EIO, 'original lock operation') if kind is OSError else kind('original body') if kind else None
    secondary = OSError('controlled owned lock close failure')
    original_open = Path.open
    closed = []
    class Handle:
        def __init__(self, handle):
            self.handle = handle
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def write(self, value):
            if stage == 'marker':
                raise primary
            return self.handle.write(value)
        def close(self):
            self.handle.close()
            closed.append(self.handle.closed)
            raise secondary
    def owned_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return Handle(handle) if path.name == '.rpipe.lock' else handle
    monkeypatch.setattr(Path, 'open', owned_open)
    if stage == 'acquire':
        if os.name == 'nt':
            import msvcrt
            def unavailable(*args):
                raise primary
            monkeypatch.setattr(msvcrt, 'locking', unavailable)
        else:
            import fcntl
            def unavailable(*args):
                raise primary
            monkeypatch.setattr(fcntl, 'flock', unavailable)
    with pytest.raises(BaseException) as caught:
        with run_lock(tmp_path):
            if stage == 'body':
                raise primary
    assert caught.value is (secondary if primary is None else primary)
    assert closed == [True]
    if primary is not None:
        assert primary.__cause__ is secondary or any(
            'controlled owned lock close failure' in note for note in getattr(primary, '__notes__', []))


def test_unknown_lock_close_programming_error_is_not_suppressed(tmp_path, monkeypatch):
    original_open = Path.open
    secondary = ValueError('unexpected close programming error')
    class Handle:
        def __init__(self, handle):
            self.handle = handle
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def close(self):
            self.handle.close()
            raise secondary
    def owned_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return Handle(handle) if path.name == '.rpipe.lock' else handle
    monkeypatch.setattr(Path, 'open', owned_open)
    with pytest.raises(ValueError) as caught:
        with run_lock(tmp_path):
            raise RuntimeError('body still in progress')
    assert caught.value is secondary


@pytest.mark.parametrize('kind', [RuntimeError, KeyboardInterrupt, None])
def test_unlock_io_error_is_visible_without_replacing_primary(tmp_path, monkeypatch, kind):
    primary = kind('original business failure') if kind else None
    secondary = OSError('controlled unlock failure')
    if os.name == 'nt':
        import msvcrt
        original = msvcrt.locking
        def release(*args):
            result = original(*args)
            if args[1] == msvcrt.LK_UNLCK:
                raise secondary
            return result
        monkeypatch.setattr(msvcrt, 'locking', release)
    else:
        import fcntl
        original = fcntl.flock
        def release(*args):
            result = original(*args)
            if args[1] == fcntl.LOCK_UN:
                raise secondary
            return result
        monkeypatch.setattr(fcntl, 'flock', release)
    with pytest.raises(BaseException) as caught:
        with run_lock(tmp_path):
            if primary is not None:
                raise primary
    assert caught.value is (secondary if primary is None else primary)
    if primary is not None:
        assert primary.__cause__ is secondary or any(
            'controlled unlock failure' in note for note in getattr(primary, '__notes__', []))
