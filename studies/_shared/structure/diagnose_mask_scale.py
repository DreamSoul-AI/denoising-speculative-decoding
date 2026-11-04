"""Re-export shim; helpers live in window_diagnostics.

    python studies/_shared/structure/diagnose_mask_scale.py
"""
try:
    from window_diagnostics import crossed_masks
except ImportError:
    from studies._shared.structure.window_diagnostics import crossed_masks
__all__ = ['crossed_masks']
