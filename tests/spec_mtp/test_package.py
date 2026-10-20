"""Packaging and entry: RPipe lives in src/rpipe; spec_mtp is a thin entry over it."""

import importlib.metadata as metadata
import json
import re
from pathlib import Path

import pytest
import rpipe

from spec_mtp import cli
from spec_mtp.structure.artifact import sources

tomllib = pytest.importorskip('tomllib')  # Python >= 3.11

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


def _vendored_commit():
    text = (ROOT / 'src' / 'rpipe' / 'ORIGIN').read_text(encoding='utf-8').splitlines()[0]
    match = re.match(r'[0-9a-f]{40}$', text)
    if not match:
        raise AssertionError('src/rpipe/ORIGIN must be the full upstream commit')
    return text


def test_rpipe_is_vendored_from_the_recorded_commit():
    commit = _vendored_commit()
    assert (ROOT / 'src' / 'rpipe' / '__init__.py').is_file()
    assert 'rpipe*' in PYPROJECT['tool']['setuptools']['packages']['find']['include']
    assert Path(rpipe.__file__).resolve().parent == (ROOT / 'src' / 'rpipe').resolve()
    origin = sources.rpipe_origin()
    assert origin.get('vendored') is True
    assert origin.get('commit') == commit


def test_console_scripts_expose_vendored_rpipe_and_spec_mtp():
    assert PYPROJECT['project']['scripts'] == {
        'rpipe': 'rpipe.flow.cli:main',
        'spec-mtp': 'spec_mtp.cli:main',
    }
    dist = metadata.distribution('denoise-spec-mtp')
    assert {ep.name: ep.value for ep in dist.entry_points if ep.group == 'console_scripts'} == {
        'rpipe': 'rpipe.flow.cli:main',
        'spec-mtp': 'spec_mtp.cli:main',
    }


def test_manifest_covers_package_sources():
    body = sources.manifest()
    assert {'spec_mtp/recipe.py', 'spec_mtp/flow/collect/run.py',
            'spec_mtp/structure/artifact/validation.py',
            'spec_mtp/structure/algorithm/decoding.py'} <= set(body['files'])
    assert not any(name.startswith('rpipe/') for name in body['files'])
    assert body['digest'] == sources.files_digest(body['files'])


def test_cli_lock_only_for_make_run_launch(tmp_path):
    study = tmp_path / 's'
    study.mkdir()
    (study / 'study.yaml').write_text('study: s\n')
    assert cli.write_lock_for(['status', str(study)]) is None
    assert cli.write_lock_for(['make', str(study)]) == study / sources.LOCK_NAME
    first = json.loads((study / sources.LOCK_NAME).read_text())['created_at']
    (study / 'scripts').mkdir()
    (study / 'scripts' / 'jobs.json').write_text('{}')
    assert cli.write_lock_for(['launch', str(study)]) is None  # keep the make-time lock
    assert json.loads((study / sources.LOCK_NAME).read_text())['created_at'] == first
    assert cli.write_lock_for(['launch', str(study), '--remake']) is not None
    assert cli.write_lock_for(['study', 'run', str(study)]) is None  # run must not refresh freeze lock


def test_run_does_not_create_lock_when_freeze(tmp_path):
    study = tmp_path / 'frozen'
    study.mkdir()
    (study / 'study.yaml').write_text('study: s\nfreeze: true\n', encoding='utf-8')
    assert cli.write_lock_for(['run', str(study)]) is None
    assert not (study / sources.LOCK_NAME).exists()


def test_run_writes_lock_when_missing_and_unfrozen(tmp_path):
    study = tmp_path / 'open'
    study.mkdir()
    (study / 'study.yaml').write_text('study: s\n', encoding='utf-8')
    assert cli.write_lock_for(['run', str(study)]) == study / sources.LOCK_NAME


def test_load_lock_absent_is_none(tmp_path):
    assert sources.load_lock(tmp_path) is None


def test_load_lock_rejects_corrupt_and_non_object(tmp_path):
    path = tmp_path / sources.LOCK_NAME
    path.write_text('{not json', encoding='utf-8')
    with pytest.raises(sources.SpecMtpSourceChangedError, match='not valid JSON'):
        sources.load_lock(tmp_path)
    path.write_text('[1, 2]', encoding='utf-8')
    with pytest.raises(sources.SpecMtpSourceChangedError, match='JSON object'):
        sources.load_lock(tmp_path)
    path.write_text('{"digest": "abc"}\n', encoding='utf-8')
    assert sources.load_lock(tmp_path) == {'digest': 'abc'}


def test_package_initializers_do_not_define_business_functions():
    import ast

    package = ROOT / 'src/spec_mtp'
    violations = []
    for path in package.rglob('__init__.py'):
        body = ast.parse(path.read_text(encoding='utf-8'))
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in body.body):
            violations.append(path.relative_to(package).as_posix())
    assert not violations, violations
