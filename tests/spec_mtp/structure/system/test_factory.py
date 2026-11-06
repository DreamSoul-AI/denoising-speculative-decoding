"""Checkpoint selection is read-only and prefers an existing committed bundle."""

from types import SimpleNamespace

from pathlib import Path

import pytest

from spec_mtp.structure.system.factory import System, SystemFactory
from spec_mtp.structure.system.logger import Logger
from spec_mtp.structure.system.config import SystemConfig

def _system(root):
    return System(device='cuda:0', logger=SimpleNamespace(), assets_dir=root, meta={})

def test_missing_checkpoint_read_does_not_create_assets(tmp_path):
    assets = tmp_path / 'absent-assets'
    system = _system(assets)
    assert system.checkpoint_path('latest') == assets / 'checkpoints/latest.pt'
    assert system.load_checkpoint('latest') is None
    assert not assets.exists()

@pytest.mark.parametrize('reference', ['latest', 'directory', 'bundle'])
def test_checkpoint_path_prefers_committed_bundle_without_touching_mirror(tmp_path, reference):
    folder = tmp_path / 'checkpoints/latest'
    folder.mkdir(parents=True)
    (folder / 'model.pt').write_bytes(b'older diagnostic mirror')
    bundle = folder.with_suffix('.pt')
    bundle.write_bytes(b'committed bundle identity fixture')
    (folder.parent / '.latest.incomplete').touch()
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    requested = {'latest': 'latest', 'directory': str(folder), 'bundle': str(bundle)}[reference]
    assert _system(tmp_path).checkpoint_path(requested) == bundle
    assert before == {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}

def test_checkpoint_path_rejects_incomplete_mirror_without_bundle(tmp_path):
    root = tmp_path / 'checkpoints'
    root.mkdir()
    (root / '.latest.incomplete').touch()
    with pytest.raises(OSError, match='incomplete checkpoint'):
        _system(tmp_path).checkpoint_path('latest')
    assert sorted(path.name for path in root.iterdir()) == ['.latest.incomplete']

@pytest.mark.parametrize('relative', ['.latest.unique.pt.writing', '.latest.unique.writing/model.pt', 'alias.pt'])
def test_checkpoint_path_rejects_explicit_unpublished_staging_material(tmp_path, monkeypatch, relative):
    temporary = tmp_path / 'checkpoints' / relative
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(b'unpublished encoding fixture')
    if relative == 'alias.pt':
        original_resolve = Path.resolve
        staged = temporary.with_name('.latest.unique.pt.writing')
        monkeypatch.setattr(Path, 'resolve', lambda path, *args, **options:
                            original_resolve(staged if path == temporary else path, *args, **options))
    for action in (_system(tmp_path).checkpoint_path, _system(tmp_path).load_checkpoint):
        with pytest.raises(OSError, match='unpublished checkpoint'):
            action(str(temporary))
    assert temporary.read_bytes() == b'unpublished encoding fixture'

@pytest.mark.parametrize('reference', ['latest', 'directory'])
def test_checkpoint_path_rechecks_selected_bundle_for_unpublished_alias(tmp_path, monkeypatch, reference):
    folder = tmp_path / 'checkpoints/latest'
    folder.mkdir(parents=True)
    bundle = folder.with_suffix('.pt')
    bundle.write_bytes(b'selected bundle alias fixture')
    staged = bundle.with_name('.latest.unique.pt.writing')
    original_resolve = Path.resolve
    monkeypatch.setattr(Path, 'resolve', lambda path, *args, **options:
                        original_resolve(staged if path == bundle else path, *args, **options))
    requested = 'latest' if reference == 'latest' else str(folder)
    with pytest.raises(OSError, match='unpublished checkpoint'):
        _system(tmp_path).load_checkpoint(requested)
    assert bundle.read_bytes() == b'selected bundle alias fixture'


@pytest.mark.parametrize('owner', ['same_run', 'different_run', 'spy'])
def test_native_system_reuses_only_the_admitted_run_logger(tmp_path, monkeypatch, owner):
    assets = tmp_path / 'runs/r1/assets'
    injected = (SimpleNamespace(info=lambda *a, **k: None) if owner == 'spy' else
                Logger(assets if owner == 'same_run' else tmp_path / 'runs/r2/assets', managed=True))
    def forbidden(*args, **kwargs): pytest.fail('injected logger must not trigger another Logger construction')
    monkeypatch.setattr(Logger, '__init__', forbidden)
    if owner == 'different_run':
        with pytest.raises(ValueError): SystemFactory.build(SystemConfig(), assets, logger=injected)
    else:
        system = SystemFactory.build(SystemConfig(), assets, logger=injected)
        assert system.logger is injected and system.assets_dir == assets
        assert system.to_result_snapshot()['ready'] is True


def test_native_system_without_injection_isolates_logger_construction_failure(tmp_path, monkeypatch):
    assets = tmp_path / 'assets'
    path = assets / 'logs/events.jsonl'
    primary = PermissionError('diagnostic directory not writable')
    original_open = Path.open
    def opened(candidate, *args, **kwargs):
        if candidate == path: raise primary
        return original_open(candidate, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', opened)
        system = SystemFactory.build(SystemConfig(), assets)
        system.logger.info('still a diagnostic operation')
    assert system.assets_dir == assets and system.to_result_snapshot()['ready'] is True
    assert system.logger.failure_count >= 1 and system.logger.last_error_type == 'PermissionError'
    assert system.logger.state_dict() == {'path': str(path)}
