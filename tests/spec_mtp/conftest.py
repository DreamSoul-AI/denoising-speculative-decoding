"""Shared helpers: copy the smoke Study into a temp dir and run it through RPipe."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / 'studies' / 'smoke'


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


def cli(*args, entry='spec_mtp', check=False):
    env = {**os.environ, 'PYTHONUTF8': '1', 'MPLBACKEND': 'Agg'}
    proc = subprocess.run([sys.executable, '-m', entry, *map(str, args)],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    if check and proc.returncode != 0:
        raise AssertionError(proc.stdout[-4000:] + proc.stderr[-4000:])
    return proc


@pytest.fixture
def study_copy(tmp_path):
    return lambda patch=None, name='smoke_copy': copy_study(tmp_path, patch, name)
