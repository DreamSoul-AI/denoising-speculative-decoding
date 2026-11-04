"""Re-export shim; helpers live in sampling_diagnostics.

    python studies/_shared/structure/diagnose_rare_events.py
"""
try:
    from sampling_diagnostics import detectability
except ImportError:
    from studies._shared.structure.sampling_diagnostics import detectability
__all__ = ['detectability']
