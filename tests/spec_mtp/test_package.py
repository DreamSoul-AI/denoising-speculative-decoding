"""Packaging and entry: spec_mtp owns the study flow; no embedded RPipe package."""

import importlib.metadata as metadata
from pathlib import Path

import pytest

from studies._shared.structure import sources

tomllib = pytest.importorskip('tomllib')  # Python >= 3.11

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


def test_rpipe_copy_is_gone_and_the_template_is_recorded():
    assert not (ROOT / 'src' / 'rpipe').exists()
    assert not (ROOT / 'tools').exists()
    include = PYPROJECT['tool']['setuptools']['packages']['find']['include']
    assert 'rpipe*' not in include and 'spec_mtp*' in include
    origin = sources.rpipe_origin()
    assert origin == {'vendored': False, 'template': 'https://github.com/DreamSoul-AI/RPipe'}
    with pytest.raises(ModuleNotFoundError):
        import rpipe  # noqa: F401


def test_console_scripts_expose_spec_mtp():
    assert PYPROJECT['project']['scripts'] == {'spec-mtp': 'spec_mtp.cli:main'}
    dist = metadata.distribution('denoise-spec-mtp')
    assert {ep.name: ep.value for ep in dist.entry_points if ep.group == 'console_scripts'} == {
        'spec-mtp': 'spec_mtp.cli:main',
    }
