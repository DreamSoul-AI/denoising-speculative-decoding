"""Re-export shim; helpers live in window_diagnostics."""
try:
    from window_diagnostics import select_window, compare_window
except ImportError:
    from studies._shared.structure.window_diagnostics import select_window, compare_window
__all__ = ['select_window', 'compare_window']
