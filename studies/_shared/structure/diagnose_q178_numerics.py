"""Re-export shim; helpers live in q178_diagnostics."""
try:
    from q178_diagnostics import canonical_states, synchronized_transfer, tensor_equal
except ImportError:
    from studies._shared.structure.q178_diagnostics import canonical_states, synchronized_transfer, tensor_equal
__all__ = ['canonical_states', 'synchronized_transfer', 'tensor_equal']
