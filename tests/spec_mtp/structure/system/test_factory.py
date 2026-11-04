"""Checkpoint selection is read-only and prefers an existing committed bundle."""

from types import SimpleNamespace
from pathlib import Path

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


@pytest.fixture
def cuda_checkpoint(tmp_path):
    import torch

    assert torch.cuda.is_available(), 'checkpoint tensor integration requires an available CUDA device'
    device = torch.device('cuda', torch.cuda.current_device())
    print(f'checkpoint persistence device={device} name={torch.cuda.get_device_name(device)}')
    system = System(device=str(device), logger=SimpleNamespace(), assets_dir=tmp_path, meta={})

    def payload(value, *, tensor_meta=False):
        body = {'model': {'weight': torch.tensor([value, value + 1.], device=device)},
                'optimizer': {'step': value, 'buffer': torch.tensor([2. * value], device=device)},
                'scheduler': {'last_epoch': value}, 'tracker': {'history': [value]},
                'logger': {'fixture': True}, 'epoch': value}
        if tensor_meta:
            body['metadata_tensor'] = torch.tensor([value], device=device)
        return body

    return SimpleNamespace(system=system, payload=payload, device=device)


def _failure(monkeypatch, system, phase):
    import os
    import tempfile
    import torch
    from spec_mtp.structure.artifact import _atomic

    primary = (TypeError if phase == 'meta_writer_type' else OSError)(f'injected checkpoint {phase} failure')
    original_save, original_text, original_replace = torch.save, _atomic.atomic_write_text, _atomic.atomic_replace
    bundle, folder = system.assets_dir / 'checkpoints/latest.pt', system.assets_dir / 'checkpoints/latest'

    def save(value, target, **options):
        path = Path(target) if isinstance(target, (str, Path)) else None
        if phase == 'serialization' and path is None:
            target.write(b'partial uncommitted bundle')
            raise primary
        if path is not None and ((phase == 'piece' and path.name == 'model.pt')
                                 or (phase == 'meta_pt' and path.name == 'meta.pt')):
            path.write_bytes(b'partial diagnostic piece')
            raise primary
        return original_save(value, target, **options)

    def text(path, *args, **options):
        if Path(path).name == {'tracker': 'tracker.json', 'meta_json': 'meta.json',
                              'meta_writer_type': 'meta.json'}.get(phase):
            raise primary
        return original_text(path, *args, **options)

    def replace(source, target):
        if ((phase == 'mirror_publish' and Path(target) == folder / 'optimizer.pt')
                or (phase == 'bundle_publish' and Path(target) == bundle)):
            raise primary
        return original_replace(source, target)

    monkeypatch.setattr(torch, 'save', save)
    monkeypatch.setattr(_atomic, 'atomic_write_text', text)
    monkeypatch.setattr(_atomic, 'atomic_replace', replace)
    if phase == 'fsync':
        monkeypatch.setattr(os, 'fsync', lambda fd: (_ for _ in ()).throw(primary))
    if phase == 'flush':
        original_temporary = tempfile.NamedTemporaryFile

        class FailedFlush:
            def __init__(self, *args, **options):
                self.stream = original_temporary(*args, **options)

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def flush(self):
                raise primary

        monkeypatch.setattr(tempfile, 'NamedTemporaryFile', FailedFlush)
    return primary


@pytest.mark.parametrize('phase', ['serialization', 'flush', 'fsync', 'piece', 'tracker',
                                  'meta_json', 'meta_writer_type', 'meta_pt', 'mirror_publish', 'bundle_publish'])
def test_checkpoint_precommit_failure_preserves_committed_cuda_bundle(cuda_checkpoint, monkeypatch, phase):
    import torch

    case = cuda_checkpoint
    previous = case.payload(1)
    bundle = case.system.save_checkpoint(previous, 'latest')
    original_bytes = bundle.read_bytes()
    failure = _failure(monkeypatch, case.system, phase)
    with pytest.raises(type(failure)) as caught:
        case.system.save_checkpoint(case.payload(2, tensor_meta=phase == 'meta_pt'), 'latest')
    assert caught.value is failure and bundle.read_bytes() == original_bytes
    for reference in ('latest', str(bundle.with_suffix('')), str(bundle)):
        assert case.system.checkpoint_path(reference) == bundle
        restored = case.system.load_checkpoint(reference)
        assert restored['epoch'] == 1 and restored['model']['weight'].device == case.device
        assert torch.equal(restored['model']['weight'], previous['model']['weight'])


@pytest.mark.parametrize('phase', ['mirror_publish', 'bundle_publish'])
def test_first_write_partial_mirror_cannot_be_selected_as_committed_checkpoint(cuda_checkpoint, monkeypatch, phase):
    case = cuda_checkpoint
    _failure(monkeypatch, case.system, phase)
    with pytest.raises(OSError):
        case.system.save_checkpoint(case.payload(1), 'latest')
    bundle = case.system.assets_dir / 'checkpoints/latest.pt'
    assert not bundle.exists() and (bundle.parent / '.latest.incomplete').is_file()
    for reference in ('latest', str(bundle.with_suffix('')), str(bundle)):
        with pytest.raises(OSError, match='incomplete checkpoint'):
            case.system.load_checkpoint(reference)


@pytest.mark.parametrize('phase', ['serialization', 'meta_json', 'mirror_publish', 'bundle_publish', 'none'])
def test_failed_write_never_promotes_unknown_legacy_mirror_to_bundle(cuda_checkpoint, monkeypatch, phase):
    import json
    import torch

    case = cuda_checkpoint
    folder = case.system.assets_dir / 'checkpoints/latest'
    folder.mkdir(parents=True)
    torch.save(case.payload(1)['model'], folder / 'model.pt')
    (folder / 'meta.json').write_text(json.dumps({'epoch': 1}), encoding='utf-8')
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    if phase != 'none':
        _failure(monkeypatch, case.system, phase)
    with pytest.raises(FileExistsError, match='explicit migration'):
        case.system.save_checkpoint(case.payload(2), 'latest')
    assert not folder.with_suffix('.pt').exists()
    assert not (folder.parent / '.latest.incomplete').exists()
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


@pytest.mark.parametrize('cleanup', ['marker', 'staging'])
@pytest.mark.parametrize('error_kind', [OSError, RuntimeError])
def test_committed_cuda_checkpoint_survives_diagnostic_cleanup_failure(cuda_checkpoint, monkeypatch, cleanup, error_kind):
    case = cuda_checkpoint
    case.system.save_checkpoint(case.payload(1), 'latest')
    original_unlink, original_rmdir = Path.unlink, Path.rmdir

    def unlink(path, *args, **options):
        if cleanup == 'marker' and path.name == '.latest.incomplete':
            raise error_kind('injected committed marker cleanup failure')
        return original_unlink(path, *args, **options)

    def rmdir(path, *args, **options):
        if cleanup == 'staging' and path.name.startswith('.latest.') and path.name.endswith('.writing'):
            raise error_kind('injected committed staging cleanup failure')
        return original_rmdir(path, *args, **options)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', unlink)
        patch.setattr(Path, 'rmdir', rmdir)
        bundle = case.system.save_checkpoint(case.payload(2), 'latest')
    assert bundle.is_file() and case.system.load_checkpoint('latest')['epoch'] == 2


def test_checkpoint_bundle_is_fsynced_before_publication_and_tensor_meta_roundtrips(cuda_checkpoint, monkeypatch):
    import os
    import torch
    from spec_mtp.structure.artifact import _atomic

    case = cuda_checkpoint
    events = []
    original_sync, original_replace = os.fsync, _atomic.atomic_replace

    def sync(fd):
        events.append('fsync')
        return original_sync(fd)

    def replace(source, target):
        if Path(target).name == 'latest.pt':
            assert events == ['fsync']
            events.append('commit')
        return original_replace(source, target)

    monkeypatch.setattr(os, 'fsync', sync)
    monkeypatch.setattr(_atomic, 'atomic_replace', replace)
    payload = case.payload(3, tensor_meta=True)
    bundle = case.system.save_checkpoint(payload, 'latest')
    restored = case.system.load_checkpoint(str(bundle))
    assert events == ['fsync', 'commit']
    assert torch.equal(restored['metadata_tensor'], payload['metadata_tensor'])
    assert (bundle.with_suffix('') / 'meta.pt').is_file()


def test_failed_checkpoint_cleanup_keeps_primary_error_and_previous_cuda_bundle(cuda_checkpoint, monkeypatch):
    case = cuda_checkpoint
    bundle = case.system.save_checkpoint(case.payload(1), 'latest')
    before = bundle.read_bytes()
    primary = _failure(monkeypatch, case.system, 'serialization')
    cleanup = OSError('injected uncommitted temporary cleanup failure')
    original_unlink = Path.unlink

    def unlink(path, *args, **options):
        if path.name.startswith('.latest.') and path.name.endswith('.pt.writing'):
            raise cleanup
        return original_unlink(path, *args, **options)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', unlink)
        with pytest.raises(OSError) as caught:
            case.system.save_checkpoint(case.payload(2), 'latest')
    assert caught.value is primary and primary.cleanup_errors == (cleanup,)
    assert bundle.read_bytes() == before and case.system.load_checkpoint('latest')['epoch'] == 1


def test_complete_but_unpublished_cuda_temporary_never_becomes_resume_source(cuda_checkpoint, monkeypatch):
    import torch

    case = cuda_checkpoint
    _failure(monkeypatch, case.system, 'fsync')
    original_unlink = Path.unlink

    def unlink(path, *args, **options):
        if path.name.startswith('.latest.') and path.name.endswith('.pt.writing'):
            raise OSError('injected retained temporary')
        return original_unlink(path, *args, **options)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', unlink)
        with pytest.raises(OSError, match='fsync'):
            case.system.save_checkpoint(case.payload(4), 'latest')
    temporary, = (case.system.assets_dir / 'checkpoints').glob('.latest.*.pt.writing')
    assert torch.load(temporary, map_location=case.device, weights_only=False)['epoch'] == 4
    with pytest.raises(OSError, match='unpublished checkpoint'):
        case.system.load_checkpoint(str(temporary))
    assert case.system.load_checkpoint('latest') is None


def test_checkpoint_serialization_error_survives_a_second_stream_close_failure(cuda_checkpoint, monkeypatch):
    import tempfile
    import torch

    case = cuda_checkpoint
    bundle = case.system.save_checkpoint(case.payload(1), 'latest')
    before = bundle.read_bytes()
    original_temporary = tempfile.NamedTemporaryFile
    primary, cleanup = OSError('primary serialization failure'), OSError('secondary stream close failure')

    class FailedClose:
        def __init__(self, *args, **options):
            self.stream = original_temporary(*args, **options)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            self.stream.close()
            raise cleanup

    def fail(value, stream, **options):
        stream.write(b'partial serialization')
        raise primary

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, 'NamedTemporaryFile', FailedClose)
        patch.setattr(torch, 'save', fail)
        with pytest.raises(OSError) as caught:
            case.system.save_checkpoint(case.payload(2), 'latest')
    assert caught.value is primary and cleanup in primary.cleanup_errors
    assert bundle.read_bytes() == before
    assert case.system.load_checkpoint('latest')['epoch'] == 1
