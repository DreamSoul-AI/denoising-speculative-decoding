"""Re-export shim; helpers live in extension_diagnostics."""
try:
    from extension_diagnostics import choose_window
except ImportError:
    from studies._shared.structure.extension_diagnostics import choose_window
__all__ = ['choose_window']
