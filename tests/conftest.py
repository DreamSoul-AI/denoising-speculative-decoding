"""Shared helpers: copy the smoke Study into a temp dir and run it through RPipe."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'studies' / 'smoke'
# Child interpreters must verify this checkout, not a different globally editable SD.
os.environ['PYTHONPATH'] = os.pathsep.join((str(ROOT / 'src'), str(ROOT)))


def copy_study(tmp_path, patch=None, name='smoke_copy'):
    """Copy declarations, recipe and flow (no runs) and apply a study.yaml patch."""
    dest = tmp_path / name
    shutil.copytree(SMOKE, dest, ignore=shutil.ignore_patterns(
        'runs', 'shared', 'scripts', 'index.json', 'process.json', 'provenance.json',
        'spec_mtp_*.json', 'activity.json', '__pycache__', 'docs'))
    study = yaml.safe_load((dest / 'study.yaml').read_text(encoding='utf-8'))
    study['study'] = name
    study['fixed']['algorithm']['config']['version'] = name
    if patch:
        patch(study)
    (dest / 'study.yaml').write_text(yaml.safe_dump(study, sort_keys=False), encoding='utf-8')
    return dest


def cli(*args, entry='studies._shared.flow.cli', check=False):
    env = {**os.environ, 'PYTHONUTF8': '1', 'MPLBACKEND': 'Agg'}
    proc = subprocess.run([sys.executable, '-m', entry, *map(str, args)],
                          capture_output=True, text=True, env=env, cwd=ROOT, timeout=90)
    if check and proc.returncode != 0:
        raise AssertionError(proc.stdout[-4000:] + proc.stderr[-4000:])
    return proc


@pytest.fixture
def study_copy(tmp_path):
    return lambda patch=None, name='smoke_copy': copy_study(tmp_path, patch, name)


@pytest.fixture(scope='session')
def historical_file(tmp_path_factory):
    """Read immutable historical inputs, never import or execute archived source."""
    import hashlib

    commit = 'b8b20b5ada7fd860894114cbf224ce749af430cf'
    root = tmp_path_factory.mktemp('historical-inputs')

    def materialize(relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('historical member must stay inside the archive')
        command = ['git', '-c', 'safe.directory=' + str(ROOT), '-C', str(ROOT)]
        oid = subprocess.check_output(command + ['rev-parse', commit + ':' + relative.as_posix()],
                                      text=True, timeout=20).strip()
        data = subprocess.check_output(command + ['cat-file', 'blob', oid], timeout=20)
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == oid
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)
        assert path.read_bytes() == data, 'historical input was changed: ' + str(relative)
        return path

    return materialize
