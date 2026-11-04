"""Algorithm base class (no factory import)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spec_mtp.structure.algorithm.config import AlgorithmConfig
from spec_mtp.structure.algorithm.hook import AlgorithmHook
from spec_mtp.structure.algorithm.optim import make_optimizer as _make_optimizer
from spec_mtp.structure.algorithm.optim import make_scheduler as _make_scheduler
from spec_mtp.structure.algorithm.resume import (
    apply_module_state,
    resolve_checkpoint_ref,
    resume_stem,
    sibling_train_checkpoint,
    partial_resume_warning,
)
from spec_mtp.structure.algorithm.tracker import AlgorithmTracker


class Algorithm(AlgorithmHook):
    """Computation for one mode. Optimizer / scheduler / resume live here."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.mode = config.mode or 'train'

    def make_optimizer(self, module: Any) -> Any:
        return _make_optimizer(module, self.config)

    def make_scheduler(self, optimizer: Any, t_max: int) -> Any:
        return _make_scheduler(optimizer, self.config, t_max)

    def resume(
        self,
        data: Any,
        model: Any,
        system: Any,
        tracker: AlgorithmTracker,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del data
        extra = extra if extra is not None else {}
        payload = self.load_resume(system, extra)
        self.apply_resume(payload, model, system, tracker, extra)
        return payload

    def load_resume(self, system: Any, extra: dict[str, Any]) -> dict[str, Any] | None:
        """Read the selected checkpoint without applying scientific state."""
        stem = resume_stem(self.config, mode=self.mode)
        extra['resume_stem'] = stem
        if stem is None:
            return None
        target = resolve_checkpoint_ref(stem, system, mode=self.mode)
        extra['resume_path'] = target
        loader = getattr(system, 'load_checkpoint', None)
        resolver = getattr(system, 'checkpoint_path', None)

        def read(reference):
            from spec_mtp.structure.artifact.provenance import file_sha256

            selected = Path(resolver(reference)) if callable(resolver) else None
            sha = file_sha256(selected) if selected is not None and selected.is_file() else None
            value = loader(reference) if callable(loader) else None
            if selected is not None and sha is not None:
                if file_sha256(selected) != sha:
                    raise ValueError('checkpoint bytes changed during reading')
                extra['resume_sources'] = {'path': str(selected.resolve()), 'sha256': sha}
            return value

        payload = read(target)
        if payload is None and self.mode == 'eval':
            fallback = sibling_train_checkpoint(system, stem=Path(str(target)).stem)
            if fallback is not None and callable(loader):
                payload = read(str(fallback))
                extra['resume_path'] = str(fallback)
        if payload is None:
            if self.mode == 'eval' or self.config.setting('resume_from'):
                raise FileNotFoundError(f'resume missing checkpoint {stem!r}')
            extra['resume_skipped'] = stem
            return None
        return payload

    def apply_resume(self, payload, model, system, tracker, extra) -> None:
        """Apply a checkpoint only after the caller's admission has completed."""
        if payload is None:
            return
        logger = getattr(system, 'logger', None)
        apply_module_state(getattr(model, 'module', None), payload)
        extra['resume_payload'] = payload
        if self.mode == 'train':
            if payload.get('training_state') is None or self.config.source == 'transformers_trainer':
                partial_resume_warning(logger)
            tracker_state = payload.get('tracker')
            if tracker_state:
                tracker.load_state_dict(tracker_state)
            logger_state = payload.get('logger')
            if logger is not None and logger_state is not None and hasattr(logger, 'load_state_dict'):
                logger.load_state_dict(logger_state)
        if logger is not None:
            logger.info(
                f"target={extra.get('resume_path')} epoch={payload.get('epoch')} step={payload.get('step')}",
                event='resume',
            )

    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        raise NotImplementedError
