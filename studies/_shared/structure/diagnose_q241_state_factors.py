"""Re-export shim; helpers live in q241_diagnostics."""
try:
    from q241_diagnostics import selective_transfer
except ImportError:
    from studies._shared.structure.q241_diagnostics import selective_transfer
__all__ = ['selective_transfer']
