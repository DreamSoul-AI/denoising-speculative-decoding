"""Re-export shim; helpers live in sampling_diagnostics."""
try:
    from sampling_diagnostics import conditional_acceptance
except ImportError:
    from studies._shared.structure.sampling_diagnostics import conditional_acceptance
__all__ = ['conditional_acceptance']
