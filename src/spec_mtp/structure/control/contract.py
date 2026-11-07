"""Light contract checks for RunConfig / Control."""

from __future__ import annotations

from spec_mtp.structure.control.control import Control
from spec_mtp.structure.control.run_config import RunConfig

_VALID_MODES = frozenset({'train', 'eval', 'inference'})


class ContractError(ValueError):
    """RunConfig / Control failed contract checks."""


def validate_run_config(run: RunConfig, *, require_mode: bool = False) -> None:
    if not run.id:
        raise ContractError('run id is required (content hash)')
    if require_mode:
        if not run.algorithm.mode:
            raise ContractError('algorithm.mode is required')
        if run.algorithm.mode not in _VALID_MODES:
            raise ContractError(f'invalid algorithm.mode: {run.algorithm.mode!r}')
    elif run.algorithm.mode is not None and run.algorithm.mode not in _VALID_MODES:
        raise ContractError(f'invalid algorithm.mode: {run.algorithm.mode!r}')


def validate_control(control: Control, *, require_mode: bool = False) -> None:
    validate_run_config(control.run, require_mode=require_mode)
