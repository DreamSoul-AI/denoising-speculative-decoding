"""Compatibility shim. Definitions live in spec_mtp.structure.api.decode_api."""

from spec_mtp.structure.api.decode_api import (
    AuditResult,
    Cancel,
    DecodeRequest,
    DecodeResult,
    HiddenMaskPolicy,
    MaskPolicy,
    ModelSession,
    ObservationSink,
    Prompt,
)

__all__ = [
    'AuditResult',
    'Cancel',
    'DecodeRequest',
    'DecodeResult',
    'HiddenMaskPolicy',
    'MaskPolicy',
    'ModelSession',
    'ObservationSink',
    'Prompt',
]
