"""Re-export shim; helpers live in sampling_diagnostics.

    python studies/_shared/structure/diagnose_sampling_paths.py
"""
try:
    from sampling_diagnostics import conditional_acceptance
except ImportError:
    from studies._shared.structure.sampling_diagnostics import conditional_acceptance
__all__ = ['conditional_acceptance']
