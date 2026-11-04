"""Runtime System object, factory, and Logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spec_mtp.structure.system.config import SystemConfig
from spec_mtp.structure.system.logger import Logger


class System:
    def __init__(
        self,
        *,
        device: str,
        logger: Logger,
        assets_dir: Path,
        meta: dict[str, Any],
    ) -> None:
        self.device = device
        self.logger = logger
        self.assets_dir = assets_dir
        self.meta = meta

    def place_module(self, module: Any) -> Any:
        if module is None:
            return None
        import torch

        return module.to(torch.device(self.device))

    def checkpoint_dir(self) -> Path:
        from spec_mtp.structure.artifact.asset import kinds

        directory = Path(self.assets_dir) / kinds.CHECKPOINTS
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_checkpoint(self, payload: dict[str, Any], name: str) -> Path:
        """Publish an atomic bundle after updating its diagnostic piece mirror."""
        import json
        import os
        import shutil
        import tempfile
        import torch

        from spec_mtp.structure.artifact._atomic import atomic_replace, atomic_write_text

        stem = str(name).replace('/', '_').replace('\\', '_')
        if not stem or stem in ('.', '..') or Path(stem).name != stem:
            raise ValueError(f'invalid checkpoint name: {name!r}')
        root = self.checkpoint_dir()
        bundle = root / f'{stem}.pt'
        folder = root / stem
        incomplete = root / f'.{stem}.incomplete'
        if not bundle.is_file() and folder.exists() and not incomplete.exists():
            raise FileExistsError('legacy checkpoint mirror requires explicit migration before overwrite')
        pending_bundle: Path | None = None
        tmp: Path | None = None
        try:
            handle = tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{stem}.', suffix='.pt.writing',
                                                 dir=root, delete=False)
            pending_bundle = Path(handle.name)
            try:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException as primary:
                try:
                    handle.close()
                except BaseException as cleanup:
                    setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
                raise
            else:
                handle.close()
            tmp = Path(tempfile.mkdtemp(prefix=f'.{stem}.', suffix='.writing', dir=root))
            torch_keys = ('model', 'optimizer', 'scheduler')
            for key in torch_keys:
                value = payload.get(key)
                if value is not None:
                    torch.save(value, tmp / f'{key}.pt')
            if payload.get('tracker') is not None:
                atomic_write_text(tmp / 'tracker.json', json.dumps(payload['tracker'], indent=2, ensure_ascii=False))
            if payload.get('logger') is not None:
                atomic_write_text(tmp / 'logger.json', json.dumps(payload['logger'], indent=2, ensure_ascii=False))
            meta = {key: value for key, value in payload.items()
                    if key not in torch_keys and key not in ('tracker', 'logger')}
            try:
                serialized_meta = json.dumps(meta, indent=2, ensure_ascii=False)
            except TypeError:
                torch.save(meta, tmp / 'meta.pt')
            else:
                atomic_write_text(tmp / 'meta.json', serialized_meta)
            # One writer per Run/stem; the bundle replacement is the sole commit point.
            incomplete.touch()
            folder.mkdir(exist_ok=True)
            for part in ('model.pt', 'optimizer.pt', 'scheduler.pt', 'tracker.json', 'logger.json', 'meta.json', 'meta.pt', 'payload.pt'):
                source, target = tmp / part, folder / part
                if source.is_file():
                    atomic_replace(source, target)
                else:
                    target.unlink(missing_ok=True)
            atomic_replace(pending_bundle, bundle)
        except BaseException as primary:
            for path in (pending_bundle, tmp):
                if path is not None:
                    try:
                        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
                    except BaseException as cleanup:
                        setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
            raise
        assert tmp is not None
        for path in (incomplete, tmp):
            try:
                path.rmdir() if path.is_dir() else path.unlink(missing_ok=True)
            except Exception:
                pass  # Committed bundle identity survives diagnostic cleanup failure.
        return bundle

    def checkpoint_path(self, name: str) -> Path:
        """Resolve committed checkpoint material without creating output directories."""
        raw = str(name)

        def published(path: Path) -> Path:
            if any(part.startswith('.') and part.endswith('.writing')
                   for value in (path, path.resolve()) for part in value.parts):
                raise OSError('unpublished checkpoint staging material is not a committed checkpoint')
            return path

        candidate = published(Path(raw))
        path = candidate
        if not (candidate.suffix.lower() == '.pt' or candidate.is_file() or candidate.is_dir() or '/' in raw or '\\' in raw):
            stem = raw.replace('/', '_').replace('\\', '_')
            from spec_mtp.structure.artifact.asset import kinds

            root = Path(self.assets_dir) / kinds.CHECKPOINTS
            bundle = root / f'{stem}.pt'
            folder = root / stem
            if bundle.is_file():
                path = bundle
            elif folder.is_dir():
                path = folder
            else:
                path = bundle
        if not path.is_file():
            folder = path if path.is_dir() or path.suffix.lower() != '.pt' else path.with_suffix('')
            bundle = folder.with_name(f'{folder.name}.pt')
            if bundle.is_file():
                path = bundle
            elif (folder.parent / f'.{folder.name}.incomplete').exists():
                raise OSError(f'incomplete checkpoint without a committed bundle: {folder}')
        return published(path)

    def load_checkpoint(self, name: str) -> dict[str, Any] | None:
        """Read a ``.pt`` bundle, a checkpoint directory, or a path."""
        import json
        import torch

        path = self.checkpoint_path(name)
        if path.is_dir():
            payload_path = path / 'payload.pt'
            if payload_path.is_file():
                loaded = torch.load(payload_path, map_location=self.device, weights_only=False)
                return loaded if isinstance(loaded, dict) else {'model': loaded}
            assembled: dict[str, Any] = {}
            meta_json = path / 'meta.json'
            if meta_json.is_file():
                assembled.update(json.loads(meta_json.read_text(encoding='utf-8')))
            meta_pt = path / 'meta.pt'
            if meta_pt.is_file():
                extra = torch.load(meta_pt, map_location=self.device, weights_only=False)
                if isinstance(extra, dict):
                    assembled.update(extra)
            for key in ('model', 'optimizer', 'scheduler'):
                piece = path / f'{key}.pt'
                if piece.is_file():
                    assembled[key] = torch.load(piece, map_location=self.device, weights_only=False)
            tracker_json = path / 'tracker.json'
            if tracker_json.is_file():
                assembled['tracker'] = json.loads(tracker_json.read_text(encoding='utf-8'))
            logger_json = path / 'logger.json'
            if logger_json.is_file():
                assembled['logger'] = json.loads(logger_json.read_text(encoding='utf-8'))
            return assembled or None
        if not path.is_file():
            return None
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(loaded, dict):
            return loaded
        return {'model': loaded}

    def to_result_snapshot(self) -> dict[str, Any]:
        return {'device': self.device, **self.meta}


class SystemFactory:
    @staticmethod
    def build(system_config: SystemConfig, assets_dir: Path | str) -> System:
        return _build_native(system_config, Path(assets_dir))


def _build_native(system_config: SystemConfig, assets_dir: Path) -> System:
    device = str(system_config.setting('device', 'cpu'))
    logger = Logger(assets_dir)
    deterministic = bool(system_config.setting('deterministic', False))
    return System(
        device=device,
        logger=logger,
        assets_dir=assets_dir,
        meta={
            'ready': True,
            'source': system_config.source or 'native',
            'deterministic': deterministic,
            'cudnn_benchmark': system_config.setting('cudnn_benchmark'),
            'cudnn_deterministic': system_config.setting('cudnn_deterministic'),
        },
    )
