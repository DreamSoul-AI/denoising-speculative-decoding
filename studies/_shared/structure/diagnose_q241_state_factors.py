"""Re-export shim; helpers live in q241_diagnostics.

    python studies/_shared/structure/diagnose_q241_state_factors.py
"""
try:
    from q241_diagnostics import selective_transfer
except ImportError:
    from studies._shared.structure.q241_diagnostics import selective_transfer
__all__ = ['selective_transfer']
