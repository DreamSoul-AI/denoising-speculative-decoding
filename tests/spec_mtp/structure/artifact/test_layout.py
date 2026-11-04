"""External Study snapshots preserve their original byte identity and boundaries."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from spec_mtp.structure.artifact import layout


def _tree(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def source_study(tmp_path):
    checkout = tmp_path / 'checkout'
    (checkout / '.git').mkdir(parents=True)
    source = checkout / 'studies' / 'tiny'
    members = {
        'study.yaml': (b'study: tiny\nrecipe: structure/recipe.py\n'
                       b'flow:\n  study_phases: true\n'
                       b'provenance:\n  include: [inputs/static.bin]\n'),
        'experiment_config.yaml': b'seed: 7\n',
        'structure/__init__.py': b'',
        'structure/recipe.py': b'def register(ctx):\r\n    return None\r\n',
        'flow/process/__init__.py': b'from .run import run\n',
        'flow/process/run.py': b'def run(ctx):\n    return None\n',
        'inputs/static.bin': b'actual\x00static\r\ninput',
        'docs/old-result.md': b'old output\n',
        'runs/old/assets/raw.bin': b'old Run evidence',
        'unselected.bin': b'not declared',
    }
    for relative, data in members.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return source


def test_materialization_copies_declared_study_bytes_without_outputs_or_host_library(source_study, tmp_path):
    source = source_study
    before = _tree(source)
    output = tmp_path / 'runtime'
    assert layout.materialize_study(source, output) == output.resolve()
    layout.check_study_snapshot(output)
    copied = _tree(output)
    selected = {name: data for name, data in before.items()
                if name not in ('docs/old-result.md', 'runs/old/assets/raw.bin', 'unselected.bin')}
    assert all(copied[name] == data for name, data in selected.items())
    assert not (set(before) - set(selected)) & set(copied)
    assert not (output / 'spec_mtp').exists()
    metadata = json.loads(copied['study_source.json'])
    assert metadata['schema'] == 1
    assert metadata['source_dir'] == str(source.resolve())
    assert isinstance(metadata['git'], dict)
    assert all(metadata['files'][name] == hashlib.sha256(data).hexdigest()
               for name, data in selected.items())
    assert before == _tree(source)


def test_repeated_materialization_preserves_original_metadata_and_run_outputs(source_study, tmp_path):
    output = layout.materialize_study(source_study, tmp_path / 'runtime')
    result = output / 'runs' / 'existing' / 'result.json'
    result.parent.mkdir(parents=True)
    result.write_bytes(b'{"status":"failed"}\n')
    metadata = output / 'study_source.json'
    os.utime(metadata, ns=(1_000_000_000, 1_000_000_000))
    stamp = metadata.stat().st_mtime_ns
    before = _tree(output)
    assert layout.materialize_study(source_study, output) == output
    assert _tree(output) == before
    assert metadata.stat().st_mtime_ns == stamp


@pytest.mark.parametrize('location,change', [
    ('source', 'modified'), ('source', 'new'),
    ('snapshot', 'modified'), ('snapshot', 'new'),
])
def test_materialization_rejects_changed_source_or_snapshot_without_overwriting(source_study, tmp_path,
                                                                            location, change):
    output = layout.materialize_study(source_study, tmp_path / 'runtime')
    root = source_study if location == 'source' else output
    target = root / 'structure' / ('recipe.py' if change == 'modified' else 'new.py')
    target.write_bytes(b'changed source bytes\n')
    before_source, before_output = _tree(source_study), _tree(output)
    with pytest.raises((ValueError, OSError)):
        layout.materialize_study(source_study, output)
    with pytest.raises((ValueError, OSError)):
        layout.check_study_snapshot(output)
    assert _tree(source_study) == before_source
    assert _tree(output) == before_output


@pytest.mark.parametrize('damage', ['invalid_json', 'schema', 'bool_schema', 'source_dir', 'files', 'digest', 'missing_git', 'bad_git'])
def test_materialization_rejects_damaged_metadata_without_repair(source_study, tmp_path, damage):
    output = layout.materialize_study(source_study, tmp_path / 'runtime')
    metadata = output / 'study_source.json'
    body = json.loads(metadata.read_text(encoding='utf-8'))
    if damage == 'invalid_json':
        metadata.write_bytes(b'{')
    else:
        if damage == 'schema':
            body['schema'] = 999
        elif damage == 'bool_schema':
            body['schema'] = True
        elif damage == 'missing_git':
            del body['git']
        elif damage == 'bad_git':
            body['git'] = []
        elif damage == 'source_dir':
            body['source_dir'] = str(tmp_path / 'different-source')
        elif damage == 'files':
            body['files'] = []
        else:
            body['files']['structure/recipe.py'] = '0' * 64
        metadata.write_text(json.dumps(body), encoding='utf-8')
    before = _tree(output)
    with pytest.raises((ValueError, OSError)):
        layout.check_study_snapshot(output)
    with pytest.raises((ValueError, OSError)):
        layout.materialize_study(source_study, output)
    assert _tree(output) == before


@pytest.mark.parametrize('placement', ['source', 'inside', 'checkout', 'ancestor'])
def test_materialization_rejects_source_overlap_before_creating_directories(source_study, tmp_path, placement):
    targets = {'source': source_study, 'inside': source_study / 'new-runtime',
               'checkout': tmp_path / 'checkout' / 'new-runtime', 'ancestor': tmp_path}
    output = targets[placement]
    before = _tree(tmp_path)
    with pytest.raises((ValueError, OSError)):
        layout.materialize_study(source_study, output)
    assert _tree(tmp_path) == before
    assert not (source_study / 'new-runtime').exists()
    assert not (tmp_path / 'checkout' / 'new-runtime').exists()


@pytest.mark.parametrize('populated', [False, True])
def test_materialization_refuses_an_existing_unbound_directory(source_study, tmp_path, populated):
    output = tmp_path / 'unknown'
    output.mkdir()
    if populated:
        (output / 'unowned.bin').write_bytes(b'preserve another task')
    before = _tree(output)
    with pytest.raises((ValueError, OSError)):
        layout.materialize_study(source_study, output)
    assert _tree(output) == before
    assert not (output / 'study_source.json').exists()


def test_materialization_rejects_resolved_member_escape_without_output(source_study, tmp_path, monkeypatch):
    escaped = tmp_path / 'outside.py'
    escaped.write_bytes(b'not Study source')
    member = source_study / 'structure' / 'recipe.py'
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        return original(escaped if path == member else path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', resolve)
    output = tmp_path / 'runtime'
    with pytest.raises((ValueError, OSError)):
        layout.materialize_study(source_study, output)
    assert not output.exists()
    assert escaped.read_bytes() == b'not Study source'


@pytest.mark.parametrize('limit', ['single', 'total'])
def test_materialization_rejects_static_size_limits_before_creating_output(source_study, tmp_path, limit):
    members = [('large.bin', 16 * 1024 * 1024 + 1)] if limit == 'single' else [
        (f'large-{index}.bin', 11 * 1024 * 1024) for index in range(3)]
    for name, size in members:
        with (source_study / 'inputs' / name).open('wb') as stream:
            stream.truncate(size)
    study = source_study / 'study.yaml'
    study.write_bytes(study.read_bytes().replace(b'[inputs/static.bin]', b'[inputs/*.bin]'))
    output = tmp_path / 'runtime'
    with pytest.raises((ValueError, OSError)):
        layout.materialize_study(source_study, output)
    assert not output.exists()
    assert all((source_study / 'inputs' / name).stat().st_size == size for name, size in members)


def test_ensure_study_layout_rejects_a_source_checkout_before_mkdir(source_study):
    before = _tree(source_study)
    with pytest.raises(ValueError):
        layout.ensure_study_layout(source_study)
    assert _tree(source_study) == before
    assert not (source_study / 'shared').exists()


@pytest.mark.parametrize('already_bound', [False, True])
def test_runtime_root_cannot_use_another_checkout_even_with_a_binding(source_study, tmp_path, already_bound):
    checkout = tmp_path / 'other-checkout'
    output = checkout / 'runtime'
    if already_bound:
        layout.materialize_study(source_study, output)
    (checkout / '.git').mkdir(parents=True)
    before = _tree(checkout)
    with pytest.raises(ValueError):
        layout.materialize_study(source_study, output)
    if already_bound:
        with pytest.raises(ValueError):
            layout.ensure_study_layout(output)
    assert _tree(checkout) == before


def test_source_change_before_copy_leaves_no_new_runtime(source_study, tmp_path, monkeypatch):
    original = layout._study_files
    first = True
    def selected(root):
        nonlocal first
        files = original(root)
        if first:
            first = False
            (source_study / 'structure/recipe.py').write_bytes(b'changed after selection')
        return files
    monkeypatch.setattr(layout, '_study_files', selected)
    output = tmp_path / 'runtime'
    with pytest.raises(ValueError, match='changed'):
        layout.materialize_study(source_study, output)
    assert not output.exists()
