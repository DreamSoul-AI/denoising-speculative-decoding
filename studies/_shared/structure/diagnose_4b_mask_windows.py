"""Re-export shim; helpers live in geometry_4b."""
import os  # noqa: F401
import subprocess  # noqa: F401
try:
    from geometry_4b import VARIANTS, TEMPLATE_VARIANTS, RECIPROCAL_VARIANTS, DIRECTION_VARIANTS, EARLY_ROOTS, admission__diagnose_4b_mask_windows as admission, intervention_masks, walk, active_predecessor, input_control, content_start_of, WindowCaptured, capture_window, capture_input_window, probe_window
except ImportError:
    from studies._shared.structure.geometry_4b import VARIANTS, TEMPLATE_VARIANTS, RECIPROCAL_VARIANTS, DIRECTION_VARIANTS, EARLY_ROOTS, admission__diagnose_4b_mask_windows as admission, intervention_masks, walk, active_predecessor, input_control, content_start_of, WindowCaptured, capture_window, capture_input_window, probe_window
__all__ = ['VARIANTS', 'TEMPLATE_VARIANTS', 'RECIPROCAL_VARIANTS', 'DIRECTION_VARIANTS', 'EARLY_ROOTS', 'admission', 'intervention_masks', 'walk', 'active_predecessor', 'input_control', 'content_start_of', 'WindowCaptured', 'capture_window', 'capture_input_window', 'probe_window']
