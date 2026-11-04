"""Re-export shim; helpers live in window_diagnostics.

    python studies/_shared/structure/diagnose_proxy_transfer.py
"""
try:
    from window_diagnostics import transfer
except ImportError:
    from studies._shared.structure.window_diagnostics import transfer
__all__ = ['transfer']
