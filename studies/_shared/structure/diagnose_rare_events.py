"""Re-export shim; helpers live in sampling_diagnostics."""
try:
    from sampling_diagnostics import detectability
except ImportError:
    from studies._shared.structure.sampling_diagnostics import detectability
__all__ = ['detectability']
