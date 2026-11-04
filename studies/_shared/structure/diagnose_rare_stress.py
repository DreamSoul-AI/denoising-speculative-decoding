"""Re-export shim; helpers live in sampling_diagnostics."""
try:
    from sampling_diagnostics import event_summary
except ImportError:
    from studies._shared.structure.sampling_diagnostics import event_summary
__all__ = ['event_summary']
