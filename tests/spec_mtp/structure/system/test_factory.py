"""Checkpoint selection is read-only and prefers an existing committed bundle."""

from types import SimpleNamespace

import pytest

from spec_mtp.structure.system.factory import System


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
