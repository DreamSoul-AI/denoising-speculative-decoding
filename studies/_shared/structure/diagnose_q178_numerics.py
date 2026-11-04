"""Re-export shim; helpers live in q178_diagnostics.

    python studies/_shared/structure/diagnose_q178_numerics.py
"""
try:
    from q178_diagnostics import canonical_states, synchronized_transfer, tensor_equal
except ImportError:
    from studies._shared.structure.q178_diagnostics import canonical_states, synchronized_transfer, tensor_equal
__all__ = ['canonical_states', 'synchronized_transfer', 'tensor_equal']
