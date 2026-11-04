"""Re-export shim; helpers live in extension_diagnostics.

    python studies/_shared/structure/diagnose_extension.py
"""
try:
    from extension_diagnostics import choose_window
except ImportError:
    from studies._shared.structure.extension_diagnostics import choose_window
__all__ = ['choose_window']
