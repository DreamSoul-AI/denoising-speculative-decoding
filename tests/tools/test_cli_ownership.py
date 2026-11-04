"""The local test CLI helper cleans only descendants that it launched."""
import json
import os
import subprocess
from types import SimpleNamespace

import psutil
import pytest

from conftest import cli, _process_tree


def test_successful_cli_helper_reaps_owned_descendant(tmp_path, monkeypatch):
    record = tmp_path / 'owned-child.json'
    child = "import os,time,json,psutil,pathlib; pathlib.Path(" + repr(str(record)) + ").write_text(json.dumps({'pid':os.getpid(),'created':psutil.Process().create_time()})); time.sleep(60)"
    (tmp_path / 'owned_probe.py').write_text(
        'import subprocess,sys,time,pathlib\n'
        f'subprocess.Popen([sys.executable,"-c",{child!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n'
        f'path=pathlib.Path({str(record)!r})\n'
        'deadline=time.monotonic()+5\n'
        'while not path.exists() and time.monotonic()<deadline: time.sleep(.01)\n'
        'assert path.exists()\n', encoding='utf8')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    try:
        result = cli(entry='owned_probe')
        assert result.returncode == 0, result.stdout + result.stderr
        identity = json.loads(record.read_text())
        try:
            psutil.Process(identity['pid']).wait(timeout=2)
        except psutil.NoSuchProcess:
            pass
    finally:
        if record.exists():
            identity = json.loads(record.read_text())
            try:
                owned = psutil.Process(identity['pid'])
                if owned.create_time() == identity['created'] and str(record) in ' '.join(owned.cmdline()):
                    owned.kill()
                    owned.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass


def test_timeout_confirms_owned_process_cleanup(tmp_path, monkeypatch):
    (tmp_path / 'timeout_probe.py').write_text('import time\ntime.sleep(60)\n', encoding='utf8')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        cli(entry='timeout_probe', timeout=1)
    assert caught.value.cleanup_confirmed and not caught.value.cleanup_errors
    assert caught.value.direct_process_returncode is not None
    try:
        psutil.Process(caught.value.process_id).wait(timeout=2)
    except psutil.NoSuchProcess:
        pass


def test_unsupported_process_ownership_refuses_before_launch(monkeypatch):
    monkeypatch.setattr(_process_tree, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setattr(_process_tree, 'sys', SimpleNamespace(platform='darwin'))
    calls = []
    monkeypatch.setattr(_process_tree.subprocess, 'Popen', lambda *a, **kw: calls.append(a))
    with pytest.raises(NotImplementedError, match='Windows Job or Linux'):
        _process_tree.ProcessTree().start(['never'])
    assert not calls
